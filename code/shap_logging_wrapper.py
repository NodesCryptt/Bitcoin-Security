#!/usr/bin/env python
"""
CP1 SHAP Logging Wrapper
========================
Production-ready SHAP explanation logging for FLAG/REJECT events.

Features:
- Structured JSON logs: {txid, score, decision, top_features, human_reason}
- SOC viewer format: Condensed triage packets
- File rotation: Daily log files with configurable retention
- Async saving: Non-blocking writes
- Prometheus metrics: Explanation count, save latency

Usage:
    wrapper = SHAPLoggingWrapper(explainer, output_dir="results/explanations")
    
    # Generate and save explanation
    explanation = wrapper.explain_and_log(features, txid, score, decision)
    
    # Get recent explanations for SOC review
    recent = wrapper.get_recent_explanations(limit=10)
"""

import os
import json
import time
import gzip
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Try to import pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "results" / "explanations"
DEFAULT_SOC_OUTPUT_DIR = Path(__file__).parent.parent / "results" / "soc_triage"
MAX_QUEUE_SIZE = 1000
LOG_RETENTION_DAYS = 30


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExplanationLog:
    """Structured explanation log entry."""
    txid: str
    score: float
    decision: str
    timestamp: str
    model_version: str
    top_features: List[Dict[str, Any]]
    human_reason: str
    raw_features: Dict[str, float]
    processing_latency_ms: float
    explanation_latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass 
class SOCTriagePacket:
    """Condensed triage packet for SOC analyst review."""
    txid: str
    score: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    decision: str
    timestamp: str
    
    # Quick summary
    top_3_reasons: List[str]
    recommended_action: str
    
    # Key features
    size_bytes: int
    fee_rate: float
    num_inputs: int
    num_outputs: int
    total_value_btc: float
    
    # Risk indicators
    risk_indicators: List[str]
    
    # Links
    mempool_link: str
    raw_tx_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_explanation(
        cls,
        explanation_log: ExplanationLog,
        raw_features: Dict[str, float] = None
    ) -> "SOCTriagePacket":
        """Create SOC packet from explanation log."""
        features = raw_features or explanation_log.raw_features
        
        # Determine severity
        score = explanation_log.score
        if score >= 0.8:
            severity = "CRITICAL"
            recommended_action = "IMMEDIATE REVIEW - Block if corroborated"
        elif score >= 0.6:
            severity = "HIGH"
            recommended_action = "Priority review within 1 hour"
        elif score >= 0.4:
            severity = "MEDIUM"
            recommended_action = "Review within 4 hours"
        else:
            severity = "LOW"
            recommended_action = "Standard triage queue"
        
        # Extract top 3 reasons
        top_3 = [f["feature_name"] for f in explanation_log.top_features[:3]]
        
        # Build risk indicators
        indicators = []
        if features.get("is_double_spend", 0) > 0:
            indicators.append("DOUBLE_SPEND_DETECTED")
        if features.get("any_illicit_addr", 0) > 0:
            indicators.append("ILLICIT_ADDRESS_INPUT")
        if features.get("fees", 0) == 0:
            indicators.append("ZERO_FEE")
        if features.get("num_input_addresses", 0) > 10:
            indicators.append("HIGH_INPUT_COUNT")
        if features.get("num_output_addresses", 0) > 20:
            indicators.append("HIGH_OUTPUT_COUNT")
        
        return cls(
            txid=explanation_log.txid,
            score=explanation_log.score,
            severity=severity,
            decision=explanation_log.decision,
            timestamp=explanation_log.timestamp,
            top_3_reasons=top_3,
            recommended_action=recommended_action,
            size_bytes=int(features.get("size", 0)),
            fee_rate=features.get("fee_rate", 0.0),
            num_inputs=int(features.get("num_input_addresses", 0)),
            num_outputs=int(features.get("num_output_addresses", 0)),
            total_value_btc=features.get("total_BTC", 0.0),
            risk_indicators=indicators,
            mempool_link=f"https://mempool.space/tx/{explanation_log.txid}",
            raw_tx_hash=explanation_log.txid
        )


# =============================================================================
# SHAP LOGGING WRAPPER
# =============================================================================

class SHAPLoggingWrapper:
    """
    Production wrapper for SHAP explainer with async logging.
    
    Wraps an existing SHAPExplainer and adds:
    - Async file saving (non-blocking)
    - Daily log rotation
    - SOC triage packet generation
    - Prometheus metrics integration
    """
    
    def __init__(
        self,
        explainer,
        output_dir: Path = None,
        soc_output_dir: Path = None,
        metrics = None,
        async_saving: bool = True,
        max_queue_size: int = MAX_QUEUE_SIZE,
        retention_days: int = LOG_RETENTION_DAYS,
        model_version: str = "v1"
    ):
        """
        Initialize wrapper.
        
        Args:
            explainer: SHAPExplainer instance
            output_dir: Directory for explanation logs
            soc_output_dir: Directory for SOC triage packets
            metrics: MetricsExporter instance (optional)
            async_saving: Enable async file writes
            max_queue_size: Max pending saves before blocking
            retention_days: Days to keep logs
            model_version: Model version string
        """
        self.explainer = explainer
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.soc_output_dir = Path(soc_output_dir) if soc_output_dir else DEFAULT_SOC_OUTPUT_DIR
        self.metrics = metrics
        self.async_saving = async_saving
        self.retention_days = retention_days
        self.model_version = model_version
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.soc_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats
        self._stats = {
            "explanations_generated": 0,
            "saves_completed": 0,
            "save_errors": 0,
            "total_latency_ms": 0.0,
        }
        self._lock = threading.Lock()
        
        # Async saving setup
        self._save_queue: Queue = Queue(maxsize=max_queue_size)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        
        if async_saving:
            self._start_save_worker()
        
        logger.info(f"SHAP logging wrapper initialized: {self.output_dir}")
    
    def _start_save_worker(self):
        """Start background save worker."""
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shap_save")
        
        def worker():
            while self._running:
                try:
                    item = self._save_queue.get(timeout=1.0)
                    if item is None:
                        break
                    self._do_save(item)
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Save worker error: {e}")
        
        self._executor.submit(worker)
        logger.debug("SHAP save worker started")
    
    def explain_and_log(
        self,
        features,
        txid: str,
        score: float,
        decision: str,
        processing_latency_ms: float = 0.0,
        top_k: int = 5
    ) -> ExplanationLog:
        """
        Generate explanation and log it.
        
        Args:
            features: DataFrame with feature values
            txid: Transaction ID
            score: Model prediction score
            decision: Decision (FLAG/REJECT)
            processing_latency_ms: Time spent on feature extraction
            top_k: Number of top features to include
            
        Returns:
            ExplanationLog
        """
        start_time = time.perf_counter()
        
        # Generate explanation
        explanation = self.explainer.explain(features, txid, score, decision, top_k)
        
        explanation_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Convert to log format
        log_entry = ExplanationLog(
            txid=txid,
            score=score,
            decision=decision,
            timestamp=datetime.utcnow().isoformat(),
            model_version=self.model_version,
            top_features=[fc.to_dict() for fc in explanation.top_features],
            human_reason=explanation.human_reason,
            raw_features=dict(explanation.raw_features),
            processing_latency_ms=processing_latency_ms,
            explanation_latency_ms=explanation_latency_ms
        )
        
        # Update stats
        with self._lock:
            self._stats["explanations_generated"] += 1
            self._stats["total_latency_ms"] += explanation_latency_ms
        
        # Record to Prometheus
        if self.metrics:
            try:
                self.metrics.increment("cp1_shap_explanations_total")
                self.metrics.observe("cp1_shap_latency_seconds", explanation_latency_ms / 1000)
            except Exception as e:
                logger.debug(f"Metrics error: {e}")
        
        # Queue for saving
        if self.async_saving:
            try:
                self._save_queue.put_nowait(log_entry)
            except:
                # Queue full, save synchronously
                self._do_save(log_entry)
        else:
            self._do_save(log_entry)
        
        return log_entry
    
    def _do_save(self, log_entry: ExplanationLog):
        """Actually save the log entry to disk."""
        try:
            # Get daily log file
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            log_file = self.output_dir / f"explanations_{date_str}.jsonl"
            
            # Append to file
            with open(log_file, "a") as f:
                f.write(log_entry.to_json().replace("\n", " ") + "\n")
            
            # Also save individual JSON for easy lookup
            tx_file = self.output_dir / f"{log_entry.txid[:16]}_{log_entry.timestamp.replace(':', '-')}.json"
            with open(tx_file, "w") as f:
                f.write(log_entry.to_json())
            
            # Generate and save SOC triage packet
            soc_packet = SOCTriagePacket.from_explanation(log_entry)
            soc_file = self.soc_output_dir / f"triage_{date_str}.jsonl"
            with open(soc_file, "a") as f:
                f.write(soc_packet.to_json().replace("\n", " ") + "\n")
            
            with self._lock:
                self._stats["saves_completed"] += 1
            
        except Exception as e:
            logger.error(f"Failed to save explanation: {e}")
            with self._lock:
                self._stats["save_errors"] += 1
    
    def get_recent_explanations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent explanations for SOC dashboard.
        
        Args:
            limit: Max number to return
            
        Returns:
            List of explanation dicts
        """
        explanations = []
        
        # Get today's log file
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.output_dir / f"explanations_{date_str}.jsonl"
        
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                
                # Get last N lines
                for line in reversed(lines[-limit:]):
                    try:
                        explanations.append(json.loads(line.strip()))
                    except:
                        pass
            except Exception as e:
                logger.error(f"Failed to read explanations: {e}")
        
        return explanations
    
    def get_soc_triage_queue(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get SOC triage queue (high severity first).
        
        Returns:
            List of SOC packets sorted by severity
        """
        packets = []
        
        # Get today's triage file
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        triage_file = self.soc_output_dir / f"triage_{date_str}.jsonl"
        
        if triage_file.exists():
            try:
                with open(triage_file, "r") as f:
                    for line in f:
                        try:
                            packets.append(json.loads(line.strip()))
                        except:
                            pass
            except Exception as e:
                logger.error(f"Failed to read triage: {e}")
        
        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        packets.sort(key=lambda x: (severity_order.get(x.get("severity", "LOW"), 4), x.get("timestamp", "")))
        
        return packets[:limit]
    
    def cleanup_old_logs(self, days: int = None):
        """
        Delete logs older than retention period.
        
        Args:
            days: Override retention days
        """
        retention = days or self.retention_days
        cutoff = datetime.utcnow() - timedelta(days=retention)
        
        deleted = 0
        for log_dir in [self.output_dir, self.soc_output_dir]:
            for f in log_dir.glob("*.json*"):
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime < cutoff:
                        f.unlink()
                        deleted += 1
                except Exception as e:
                    logger.debug(f"Failed to cleanup {f}: {e}")
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old log files")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        with self._lock:
            stats = dict(self._stats)
        
        stats["queue_size"] = self._save_queue.qsize()
        stats["avg_latency_ms"] = (
            stats["total_latency_ms"] / stats["explanations_generated"]
            if stats["explanations_generated"] > 0 else 0
        )
        
        return stats
    
    def shutdown(self):
        """Graceful shutdown."""
        self._running = False
        
        if self._executor:
            # Drain queue
            while not self._save_queue.empty():
                try:
                    item = self._save_queue.get_nowait()
                    if item:
                        self._do_save(item)
                except:
                    break
            
            self._executor.shutdown(wait=True)
        
        logger.info(f"SHAP wrapper shutdown. Stats: {self.get_stats()}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_shap_wrapper(
    model,
    feature_names: List[str],
    output_dir: Path = None,
    metrics = None,
    model_version: str = "v1"
) -> SHAPLoggingWrapper:
    """
    Create a SHAP logging wrapper for a model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature column names
        output_dir: Output directory for logs
        metrics: MetricsExporter instance
        model_version: Model version string
        
    Returns:
        SHAPLoggingWrapper instance
    """
    # Import here to avoid circular imports
    from shap_explainer import SHAPExplainer
    
    explainer = SHAPExplainer(model, feature_names, model_version)
    
    return SHAPLoggingWrapper(
        explainer=explainer,
        output_dir=output_dir,
        metrics=metrics,
        model_version=model_version
    )
