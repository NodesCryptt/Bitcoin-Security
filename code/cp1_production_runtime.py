#!/usr/bin/env python
"""
CP1 Production Runtime
======================
Production-ready CP1 runtime with full integration:
- Consensus validation (Bitcoin Core)
- ML inference (XGBoost)
- Safe action policy
- SHAP explainability
- Prometheus metrics
- Redis pub/sub for CP2 integration
- Feature caching

Usage:
    python cp1_production_runtime.py

Environment Variables:
    BITCOIN_RPC_URL: Bitcoin Core RPC URL
    REDIS_URL: Redis URL
    ZMQ_RAWTX: ZMQ rawtx endpoint
    PROMETHEUS_PORT: Prometheus metrics port
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import zmq
import joblib
import pandas as pd
import numpy as np

# CP1 modules
from consensus_validator import ConsensusValidator, ConsensusResult, ValidationResult
from safe_action_policy import SafeActionPolicy, Action, PolicyDecision
from shap_explainer import SHAPExplainer, Explanation
from redis_client import CP1RedisClient, PeerAlert
from metrics_exporter import MetricsExporter
from feature_cache import FeatureCache

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cp1.runtime")

# Bitcoin RPC
try:
    from bitcoinrpc.authproxy import AuthServiceProxy
except ImportError:
    logger.error("python-bitcoinrpc not installed. Run: pip install python-bitcoinrpc")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    # Bitcoin Core
    rpc_url: str = "http://cp1user:CP1SecurePassword123!@127.0.0.1:18443"
    zmq_rawtx: str = "tcp://127.0.0.1:28332"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # Model
    model_path: str = None
    feature_path: str = None
    
    # Thresholds
    accept_threshold: float = 0.12
    reject_threshold: float = 0.65
    
    # Prometheus
    metrics_port: int = 8000
    
    # Modes
    shadow_mode: bool = True  # Don't actually block in shadow mode
    enable_shap: bool = True
    enable_redis: bool = True
    
    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Create config from environment variables."""
        base_dir = Path(__file__).parent.parent
        
        return cls(
            rpc_url=os.environ.get("BITCOIN_RPC_URL", cls.rpc_url),
            zmq_rawtx=os.environ.get("ZMQ_RAWTX", cls.zmq_rawtx),
            redis_url=os.environ.get("REDIS_URL", cls.redis_url),
            model_path=os.environ.get("MODEL_PATH", str(base_dir / "models" / "cp1_static_xgb_v1.joblib")),
            feature_path=os.environ.get("FEATURE_PATH", str(base_dir / "results" / "cp1_static_ellipticpp.csv")),
            metrics_port=int(os.environ.get("PROMETHEUS_PORT", 8000)),
            shadow_mode=os.environ.get("SHADOW_MODE", "true").lower() == "true",
        )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(decoded: Dict[str, Any], feature_cols: list) -> pd.DataFrame:
    """
    Extract features from decoded transaction.
    
    Maps basic tx properties to the expected feature schema.
    Returns DataFrame ready for model inference.
    """
    raw_features = {}
    
    # Basic features
    try:
        raw_features["size"] = float(decoded.get("size", 0))
    except (ValueError, TypeError):
        raw_features["size"] = 0.0
    
    try:
        raw_features["fees"] = float(decoded.get("fee", 0))
    except (ValueError, TypeError):
        raw_features["fees"] = 0.0
    
    try:
        raw_features["num_input_addresses"] = len(decoded.get("vin", []))
    except (ValueError, TypeError):
        raw_features["num_input_addresses"] = 0
    
    try:
        raw_features["num_output_addresses"] = len(decoded.get("vout", []))
    except (ValueError, TypeError):
        raw_features["num_output_addresses"] = 0
    
    try:
        vouts = decoded.get("vout", [])
        total = 0.0
        for v in vouts:
            if isinstance(v, dict):
                val = v.get("value", 0)
                if val is not None:
                    total += float(val)
        raw_features["total_BTC"] = total
    except (ValueError, TypeError):
        raw_features["total_BTC"] = 0.0
    
    # Build row with all expected columns
    row_dict = {c: raw_features.get(c, 0.0) for c in feature_cols}
    row = pd.DataFrame([row_dict])
    
    # Ensure numeric types
    row = row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    return row


# =============================================================================
# CP1 PRODUCTION RUNTIME
# =============================================================================

class CP1ProductionRuntime:
    """
    Production CP1 runtime with full integration.
    
    Processing flow:
    1. Receive raw tx via ZMQ
    2. Consensus validation (FIRST - before ML)
    3. If invalid → REJECT immediately
    4. Feature extraction
    5. Check peer risk from CP2 (via Redis)
    6. ML inference
    7. Safe action policy
    8. If FLAG/REJECT → SHAP explanation
    9. Publish to Redis for CP2 correlation
    10. Emit Prometheus metrics
    """
    
    def __init__(self, config: RuntimeConfig = None):
        """Initialize runtime with configuration."""
        self.config = config or RuntimeConfig.from_env()
        self.running = False
        
        logger.info("=" * 60)
        logger.info("CP1 PRODUCTION RUNTIME")
        logger.info("=" * 60)
        
        # Initialize components
        self._init_bitcoin_rpc()
        self._init_model()
        self._init_zmq()
        self._init_validator()
        self._init_policy()
        self._init_redis()
        self._init_metrics()
        self._init_explainer()
        self._init_cache()
        
        # Stats
        self._stats = {
            "tx_processed": 0,
            "accept_count": 0,
            "flag_count": 0,
            "reject_count": 0,
            "consensus_reject_count": 0,
            "errors": 0,
        }
        
        logger.info("-" * 60)
        logger.info(f"Thresholds: ACCEPT < {self.config.accept_threshold}, "
                   f"FLAG {self.config.accept_threshold}-{self.config.reject_threshold}, "
                   f"REJECT >= {self.config.reject_threshold}")
        logger.info(f"Shadow mode: {self.config.shadow_mode}")
        logger.info("=" * 60)
    
    def _init_bitcoin_rpc(self):
        """Initialize Bitcoin Core RPC connection."""
        logger.info(f"Connecting to Bitcoin Core: {self.config.rpc_url.split('@')[1] if '@' in self.config.rpc_url else self.config.rpc_url}")
        try:
            self.rpc = AuthServiceProxy(self.config.rpc_url)
            info = self.rpc.getblockchaininfo()
            logger.info(f"Connected to Bitcoin Core: chain={info['chain']}, blocks={info['blocks']}")
        except Exception as e:
            logger.error(f"Failed to connect to Bitcoin Core: {e}")
            raise
    
    def _init_model(self):
        """Load ML model and feature schema."""
        logger.info(f"Loading model: {self.config.model_path}")
        self.model = joblib.load(self.config.model_path)
        
        logger.info(f"Loading feature schema: {self.config.feature_path}")
        df = pd.read_csv(self.config.feature_path, nrows=1)
        self.feature_cols = [c for c in df.columns if c != "label"]
        logger.info(f"Feature columns: {len(self.feature_cols)}")
    
    def _init_zmq(self):
        """Initialize ZMQ subscriber for raw transactions only."""
        logger.info(f"Connecting to ZMQ: {self.config.zmq_rawtx}")
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect(self.config.zmq_rawtx)
        # Subscribe ONLY to rawtx topic - ignore rawblock, hashblock, etc.
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "rawtx")
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
    
    def _init_validator(self):
        """Initialize consensus validator."""
        logger.info("Initializing consensus validator")
        self.validator = ConsensusValidator(
            self.rpc, 
            strict_mode=True,
            rpc_url=self.config.rpc_url  # Enable RPC reconnection
        )
    
    def _init_policy(self):
        """Initialize safe action policy."""
        logger.info("Initializing safe action policy")
        self.policy = SafeActionPolicy(
            accept_threshold=self.config.accept_threshold,
            reject_threshold=self.config.reject_threshold,
            shadow_mode=self.config.shadow_mode
        )
    
    def _init_redis(self):
        """Initialize Redis client."""
        if not self.config.enable_redis:
            logger.info("Redis disabled")
            self.redis = None
            return
        
        logger.info(f"Initializing Redis: {self.config.redis_url}")
        self.redis = CP1RedisClient(redis_url=self.config.redis_url)
        
        if self.redis.is_connected:
            # Subscribe to CP2 alerts
            self.redis.subscribe_peer_alerts(self._on_peer_alert)
            logger.info("Subscribed to CP2 peer alerts")
        else:
            logger.warning("Redis not connected, running without CP2 integration")
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        logger.info(f"Initializing Prometheus metrics on port {self.config.metrics_port}")
        self.metrics = MetricsExporter(port=self.config.metrics_port)
        self.metrics.start_server()
    
    def _init_explainer(self):
        """Initialize SHAP explainer."""
        if not self.config.enable_shap:
            logger.info("SHAP disabled")
            self.explainer = None
            return
        
        logger.info("Initializing SHAP explainer")
        self.explainer = SHAPExplainer(self.model, self.feature_cols)
    
    def _init_cache(self):
        """Initialize feature cache."""
        logger.info("Initializing feature cache")
        redis_url = self.config.redis_url if self.config.enable_redis else None
        self.cache = FeatureCache(redis_url=redis_url)
    
    def _on_peer_alert(self, alert: PeerAlert):
        """Handle peer alert from CP2."""
        logger.info(f"CP2 Alert: {alert.event} - peer={alert.peer_addr} score={alert.score}")
        # Store in cache for lookup during tx processing
        if self.cache:
            self.cache._set(f"peer_risk:{alert.peer_addr}", {"risk": alert.score}, 3600)
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    def process_transaction(self, raw_hex: str, peer_addr: str = None) -> PolicyDecision:
        """
        Process a single transaction.
        
        Args:
            raw_hex: Raw transaction hex
            peer_addr: Address of peer that announced this tx
        
        Returns:
            PolicyDecision
        """
        start_time = time.perf_counter()
        
        # 1. Record ingestion
        self.metrics.record_tx_ingested()
        self._stats["tx_processed"] += 1
        
        # 2. Consensus validation (FIRST)
        validation = self.validator.validate(raw_hex)
        
        if not validation.is_valid:
            self._stats["consensus_reject_count"] += 1
            self.metrics.record_consensus_failure(validation.status.value)
            logger.warning(f"Consensus REJECT: {validation.reject_reason}")
            
            # Still return a decision
            return PolicyDecision(
                action=Action.REJECT,
                score=1.0,
                severity=self.policy._create_reject_decision(1.0, {}).severity,
                reason=f"Consensus: {validation.reject_reason}",
                require_corroboration=False,  # Consensus reject doesn't need corroboration
                relay_allowed=False
            )
        
        txid = validation.txid
        decoded = validation.decoded
        
        # 3. Feature extraction
        feature_start = time.perf_counter()
        features = extract_features(decoded, self.feature_cols)
        feature_time = time.perf_counter() - feature_start
        self.metrics.observe("cp1_feature_extraction_seconds", feature_time)
        
        # 4. Get peer risk from CP2 (if available)
        peer_risk = 0.0
        if peer_addr and self.cache:
            cached = self.cache._get(f"peer_risk:{peer_addr}")
            if cached:
                peer_risk = cached.get("risk", 0.0)
        
        # 5. ML inference
        infer_start = time.perf_counter()
        score = float(self.model.predict_proba(features)[0, 1])
        infer_time = time.perf_counter() - infer_start
        self.metrics.record_latency(infer_time)
        self.metrics.record_score(score)
        
        # 6. Adjust score based on peer risk (optional enhancement)
        # adjusted_score = min(1.0, score + peer_risk * 0.1)
        adjusted_score = score  # For now, don't adjust
        
        # 7. Safe action policy
        context = {"peer_risk": peer_risk, "peer_addr": peer_addr}
        decision = self.policy.evaluate(adjusted_score, context)
        
        # 8. Record decision
        self.metrics.record_decision(decision.action.value, "ml")
        if decision.action == Action.ACCEPT:
            self._stats["accept_count"] += 1
        elif decision.action == Action.FLAG:
            self._stats["flag_count"] += 1
        else:
            self._stats["reject_count"] += 1
        
        # 9. Generate explanation for FLAG/REJECT
        explanation_dict = None
        if decision.action != Action.ACCEPT and self.explainer:
            try:
                explanation = self.explainer.explain(
                    features, txid, score, decision.action.value
                )
                self.explainer.save_explanation(explanation)
                explanation_dict = explanation.to_dict()
            except Exception as e:
                logger.error(f"SHAP explanation error: {e}")
        
        # 10. Publish to Redis for CP2
        if self.redis and decision.action != Action.ACCEPT:
            self.redis.publish_tx_decision(
                txid=txid,
                score=score,
                decision=decision.action.value,
                announcing_peer=peer_addr,
                explanation=explanation_dict
            )
        
        # Log
        total_time = time.perf_counter() - start_time
        log_level = logging.INFO if decision.action == Action.ACCEPT else logging.WARNING
        logger.log(
            log_level,
            f"{decision.action.value} | txid={txid[:16]}... | "
            f"score={score:.4f} | latency={total_time*1000:.1f}ms"
        )
        
        return decision
    
    def run(self):
        """Main run loop."""
        self.running = True
        logger.info("CP1 Runtime started. Listening for transactions...")
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while self.running:
            try:
                # Receive multipart message: [topic, body, sequence]
                # Topic is 'rawtx' (already filtered by subscription)
                topic = self.zmq_socket.recv_string()
                body = self.zmq_socket.recv()
                
                # Optional sequence number
                try:
                    seq = self.zmq_socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    seq = None
                
                # Only process rawtx (should already be filtered)
                if topic != "rawtx":
                    continue
                
                # Check minimum size (valid tx is at least 60 bytes)
                if len(body) < 60:
                    logger.debug(f"Skipping small frame: {len(body)} bytes")
                    continue
                
                # Convert binary to hex and process
                raw_hex = body.hex()
                self.process_transaction(raw_hex, peer_addr=None)
                
            except zmq.Again:
                # Timeout - no message, continue
                continue
            
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt")
                break
            
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Error processing transaction: {e}")
                continue
        
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down CP1 Runtime...")
        
        # Print stats
        logger.info("-" * 60)
        logger.info("Session Statistics:")
        for key, value in self._stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("-" * 60)
        
        # Close connections
        if self.redis:
            self.redis.close()
        
        self.zmq_socket.close()
        self.zmq_context.term()
        
        logger.info("CP1 Runtime stopped.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    config = RuntimeConfig.from_env()
    runtime = CP1ProductionRuntime(config)
    runtime.run()


if __name__ == "__main__":
    main()
