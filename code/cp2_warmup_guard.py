#!/usr/bin/env python
"""
CP2 Warm-up Gating
==================
Prevents false positive eclipse/sybil scores during IBD and initial startup.

Key behaviors:
1. During IBD: Report observe-only mode, scores = 0
2. Warm-up period: Wait for minimum peers and uptime before scoring
3. After warm-up: Full peer scoring enabled
"""

import logging
import time
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("cp2.warmup")


class CP2Mode(Enum):
    """CP2 operational modes."""
    IBD = "ibd"           # Initial block download - observe only
    WARMUP = "warmup"     # Warming up - building peer baseline
    ACTIVE = "active"     # Full scoring enabled


@dataclass
class WarmupConfig:
    """Configuration for warm-up gating."""
    min_outbound_peers: int = 8
    min_uptime_seconds: int = 1800  # 30 minutes
    min_addr_messages: int = 100
    ibd_check_interval: int = 30


@dataclass
class WarmupState:
    """Current warm-up state."""
    mode: CP2Mode
    reason: str
    outbound_peers: int
    uptime_seconds: int
    can_score: bool


class CP2WarmupGuard:
    """
    Guard that prevents peer scoring until conditions are met.
    
    Usage:
        guard = CP2WarmupGuard(rpc)
        
        while True:
            state = guard.get_state()
            
            if not state.can_score:
                # Log but don't score
                logger.info(f"CP2 {state.mode.value}: {state.reason}")
                record_placeholder_metrics()
                continue
            
            # Safe to score peers
            scores = compute_peer_scores()
    """
    
    def __init__(
        self, 
        rpc, 
        config: WarmupConfig = None
    ):
        """
        Initialize warm-up guard.
        
        Args:
            rpc: Bitcoin RPC client
            config: Warm-up configuration
        """
        self.rpc = rpc
        self.config = config or WarmupConfig()
        
        # Track startup time
        self._start_time = time.time()
        
        # Cache IBD status
        self._is_ibd = True
        self._last_ibd_check = 0
        self._sync_progress = 0.0
        
        # Initial check
        self._update_ibd_status()
        
        logger.info(f"CP2 WarmupGuard initialized")
        logger.info(f"  Min outbound peers: {self.config.min_outbound_peers}")
        logger.info(f"  Min uptime: {self.config.min_uptime_seconds}s")
    
    def _update_ibd_status(self):
        """Update IBD status from Bitcoin Core."""
        try:
            info = self.rpc.getblockchaininfo()
            self._is_ibd = info.get("initialblockdownload", True)
            self._sync_progress = info.get("verificationprogress", 0.0)
            self._last_ibd_check = time.time()
        except Exception as e:
            logger.warning(f"Failed to check IBD status: {e}")
            self._is_ibd = True
    
    @property
    def is_ibd(self) -> bool:
        """Check if node is in Initial Block Download."""
        if time.time() - self._last_ibd_check > self.config.ibd_check_interval:
            self._update_ibd_status()
        return self._is_ibd
    
    @property
    def uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self._start_time
    
    def get_peer_counts(self) -> Dict[str, int]:
        """Get current peer counts."""
        try:
            peers = self.rpc.getpeerinfo()
            
            inbound = sum(1 for p in peers if p.get("inbound", False))
            outbound = sum(1 for p in peers if not p.get("inbound", True))
            total = len(peers)
            
            return {
                "total": total,
                "inbound": inbound,
                "outbound": outbound
            }
        except Exception as e:
            logger.warning(f"Failed to get peer info: {e}")
            return {"total": 0, "inbound": 0, "outbound": 0}
    
    def get_state(self) -> WarmupState:
        """
        Get current warm-up state.
        
        Returns:
            WarmupState with mode and scoring permission
        """
        peer_counts = self.get_peer_counts()
        uptime = self.uptime
        
        # Check 1: IBD
        if self.is_ibd:
            return WarmupState(
                mode=CP2Mode.IBD,
                reason=f"sync_progress={self._sync_progress:.2%}",
                outbound_peers=peer_counts["outbound"],
                uptime_seconds=int(uptime),
                can_score=False
            )
        
        # Check 2: Minimum outbound peers
        if peer_counts["outbound"] < self.config.min_outbound_peers:
            return WarmupState(
                mode=CP2Mode.WARMUP,
                reason=f"peers={peer_counts['outbound']}/{self.config.min_outbound_peers}",
                outbound_peers=peer_counts["outbound"],
                uptime_seconds=int(uptime),
                can_score=False
            )
        
        # Check 3: Minimum uptime
        if uptime < self.config.min_uptime_seconds:
            remaining = int(self.config.min_uptime_seconds - uptime)
            return WarmupState(
                mode=CP2Mode.WARMUP,
                reason=f"uptime={int(uptime)}s, need={remaining}s more",
                outbound_peers=peer_counts["outbound"],
                uptime_seconds=int(uptime),
                can_score=False
            )
        
        # All checks passed
        return WarmupState(
            mode=CP2Mode.ACTIVE,
            reason="all_conditions_met",
            outbound_peers=peer_counts["outbound"],
            uptime_seconds=int(uptime),
            can_score=True
        )
    
    def should_score_peers(self) -> bool:
        """
        Simple check if peer scoring is allowed.
        
        Returns:
            True if safe to score peers
        """
        return self.get_state().can_score


def create_warmup_guard(rpc_url: str = None) -> CP2WarmupGuard:
    """
    Factory to create warm-up guard.
    
    Args:
        rpc_url: Bitcoin RPC URL (uses env var if not provided)
        
    Returns:
        CP2WarmupGuard instance
    """
    from bitcoinrpc.authproxy import AuthServiceProxy
    
    url = rpc_url or os.environ.get(
        "BITCOIN_RPC_URL",
        "http://rpcuser:rpcpass@127.0.0.1:8332"
    )
    
    rpc = AuthServiceProxy(url)
    return CP2WarmupGuard(rpc)


# =============================================================================
# METRICS INTEGRATION
# =============================================================================

class WarmupMetrics:
    """Prometheus metrics for warm-up state."""
    
    def __init__(self, metrics_exporter=None):
        self.exporter = metrics_exporter
    
    def record_state(self, state: WarmupState):
        """Record warm-up state to Prometheus."""
        if not self.exporter:
            return
        
        try:
            # Mode as gauge (0=ibd, 1=warmup, 2=active)
            mode_value = {
                CP2Mode.IBD: 0,
                CP2Mode.WARMUP: 1,
                CP2Mode.ACTIVE: 2
            }.get(state.mode, 0)
            
            self.exporter.set_gauge("cp2_mode", mode_value)
            self.exporter.set_gauge("cp2_outbound_peers", state.outbound_peers)
            self.exporter.set_gauge("cp2_uptime_seconds", state.uptime_seconds)
            self.exporter.set_gauge("cp2_can_score", 1 if state.can_score else 0)
            
        except Exception as e:
            logger.debug(f"Failed to record metrics: {e}")


# =============================================================================
# SAFE SCORING WRAPPER
# =============================================================================

def safe_compute_scores(
    guard: CP2WarmupGuard,
    compute_fn: callable,
    fallback_scores: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Wrapper that only computes scores when safe.
    
    Args:
        guard: Warm-up guard
        compute_fn: Function that computes actual scores
        fallback_scores: Scores to return during warm-up
        
    Returns:
        Computed scores or fallback
    """
    fallback = fallback_scores or {
        "eclipse_score": 0.0,
        "sybil_score": 0.0,
        "diversity_score": 1.0
    }
    
    state = guard.get_state()
    
    if not state.can_score:
        logger.info(f"CP2 {state.mode.value}: {state.reason} - using fallback scores")
        return {**fallback, "_mode": state.mode.value, "_reason": state.reason}
    
    # Safe to compute
    scores = compute_fn()
    scores["_mode"] = "active"
    return scores
