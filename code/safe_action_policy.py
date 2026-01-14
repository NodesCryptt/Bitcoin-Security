"""
CP1 Safe Action Policy Engine
=============================
Maps ML scores to conservative actions with safety guarantees.

This policy engine ensures that:
1. We never auto-REJECT valid transactions solely on ML score
2. Actions are mapped based on FPR-budget thresholds
3. REJECT actions require corroboration from CP2/CP3 before blocking

Threshold Selection (from cp1_threshold_analysis.csv):
- t1 = 0.12: ACCEPT threshold (FPR < 1%, high confidence safe)
- t2 = 0.65: REJECT threshold (FPR < 0.1%, very high confidence)

Action Mapping:
- ACCEPT (score < t1): Full relay/accept to mempool
- FLAG (t1 <= score < t2): Shadow/quarantine, notify analyst
- REJECT (score >= t2): Quarantine + alert, do NOT auto-drop
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Action(Enum):
    """Actions that can be taken on a transaction."""
    ACCEPT = "accept"       # Full relay, accept to mempool
    FLAG = "flag"          # Shadow/quarantine, notify analyst
    REJECT = "reject"       # Quarantine + alert (do not auto-drop)


class ActionSeverity(Enum):
    """Severity levels for actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PolicyDecision:
    """Result of policy evaluation."""
    action: Action
    score: float
    severity: ActionSeverity
    reason: str
    require_corroboration: bool = False
    notify_analyst: bool = False
    quarantine: bool = False
    relay_allowed: bool = True
    
    @property
    def is_safe(self) -> bool:
        """Check if the action is safe (ACCEPT)."""
        return self.action == Action.ACCEPT
    
    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "score": round(self.score, 4),
            "severity": self.severity.value,
            "reason": self.reason,
            "require_corroboration": self.require_corroboration,
            "notify_analyst": self.notify_analyst,
            "quarantine": self.quarantine,
            "relay_allowed": self.relay_allowed,
            "timestamp": datetime.utcnow().isoformat()
        }


class SafeActionPolicy:
    """
    Conservative action policy for CP1 decisions.
    
    This policy follows the principle of "do no harm":
    - False positives (blocking legitimate transactions) are worse than
      false negatives (missing illicit transactions)
    - Never auto-drop transactions without corroboration
    - Prefer alerting humans over automated blocking
    
    Usage:
        policy = SafeActionPolicy()
        decision = policy.evaluate(score=0.72)
        if decision.action == Action.REJECT:
            # Quarantine but don't drop
            quarantine_tx(txid)
            notify_analyst(txid, decision)
    """
    
    # Default thresholds (tuned for FPR < 2% in production)
    DEFAULT_ACCEPT_THRESHOLD = 0.12
    DEFAULT_REJECT_THRESHOLD = 0.65
    
    def __init__(
        self,
        accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
        reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
        shadow_mode: bool = True
    ):
        """
        Initialize policy.
        
        Args:
            accept_threshold: Score below which transactions are accepted
            reject_threshold: Score above which transactions are flagged for rejection
            shadow_mode: If True, never actually block (for initial deployment)
        """
        if accept_threshold >= reject_threshold:
            raise ValueError("accept_threshold must be less than reject_threshold")
        
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.shadow_mode = shadow_mode
        
        logger.info(
            f"SafeActionPolicy initialized: "
            f"ACCEPT < {accept_threshold}, "
            f"FLAG {accept_threshold}-{reject_threshold}, "
            f"REJECT >= {reject_threshold}, "
            f"shadow_mode={shadow_mode}"
        )
    
    def evaluate(self, score: float, context: Optional[Dict[str, Any]] = None) -> PolicyDecision:
        """
        Evaluate a transaction score and return the action to take.
        
        Args:
            score: ML model score (0-1, higher = more likely illicit)
            context: Optional context (mempool state, peer info, etc.)
        
        Returns:
            PolicyDecision with action and metadata
        """
        context = context or {}
        
        # Validate score
        if score < 0 or score > 1:
            logger.warning(f"Invalid score: {score}, clamping to [0, 1]")
            score = max(0, min(1, score))
        
        # Determine action based on thresholds
        if score < self.accept_threshold:
            return self._create_accept_decision(score, context)
        elif score < self.reject_threshold:
            return self._create_flag_decision(score, context)
        else:
            return self._create_reject_decision(score, context)
    
    def _create_accept_decision(
        self, score: float, context: Dict[str, Any]
    ) -> PolicyDecision:
        """Create ACCEPT decision."""
        return PolicyDecision(
            action=Action.ACCEPT,
            score=score,
            severity=ActionSeverity.LOW,
            reason=f"Score {score:.4f} below accept threshold {self.accept_threshold}",
            require_corroboration=False,
            notify_analyst=False,
            quarantine=False,
            relay_allowed=True
        )
    
    def _create_flag_decision(
        self, score: float, context: Dict[str, Any]
    ) -> PolicyDecision:
        """Create FLAG decision."""
        # Determine severity based on score within the FLAG range
        range_size = self.reject_threshold - self.accept_threshold
        normalized = (score - self.accept_threshold) / range_size
        
        if normalized < 0.5:
            severity = ActionSeverity.MEDIUM
        else:
            severity = ActionSeverity.HIGH
        
        return PolicyDecision(
            action=Action.FLAG,
            score=score,
            severity=severity,
            reason=f"Score {score:.4f} in FLAG range [{self.accept_threshold}, {self.reject_threshold})",
            require_corroboration=False,
            notify_analyst=True,
            quarantine=True,  # Shadow/quarantine
            relay_allowed=not self.shadow_mode  # In shadow mode, still relay
        )
    
    def _create_reject_decision(
        self, score: float, context: Dict[str, Any]
    ) -> PolicyDecision:
        """Create REJECT decision."""
        return PolicyDecision(
            action=Action.REJECT,
            score=score,
            severity=ActionSeverity.CRITICAL,
            reason=f"Score {score:.4f} above reject threshold {self.reject_threshold}",
            require_corroboration=True,  # NEVER auto-reject without corroboration
            notify_analyst=True,
            quarantine=True,
            relay_allowed=self.shadow_mode  # In shadow mode, still relay
        )
    
    def update_thresholds(
        self,
        accept_threshold: Optional[float] = None,
        reject_threshold: Optional[float] = None
    ):
        """
        Update thresholds dynamically.
        
        Use with caution - threshold changes should be logged and audited.
        """
        if accept_threshold is not None:
            old_accept = self.accept_threshold
            self.accept_threshold = accept_threshold
            logger.info(f"Accept threshold updated: {old_accept} -> {accept_threshold}")
        
        if reject_threshold is not None:
            old_reject = self.reject_threshold
            self.reject_threshold = reject_threshold
            logger.info(f"Reject threshold updated: {old_reject} -> {reject_threshold}")
        
        if self.accept_threshold >= self.reject_threshold:
            raise ValueError("accept_threshold must be less than reject_threshold")


# Convenience instance with default settings
DEFAULT_POLICY = SafeActionPolicy()


def evaluate_score(score: float, context: Optional[Dict[str, Any]] = None) -> PolicyDecision:
    """
    Convenience function to evaluate a score with default policy.
    
    Args:
        score: ML model score (0-1)
        context: Optional context
    
    Returns:
        PolicyDecision
    """
    return DEFAULT_POLICY.evaluate(score, context)
