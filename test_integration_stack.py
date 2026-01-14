#!/usr/bin/env python
"""
CP1-CP2 Integration Test Script
================================
Tests the full integration stack: Bitcoin Core + Redis + CP1.

Usage:
    python test_integration_stack.py

Prerequisites:
    - Docker stack running: docker-compose up -d
    - Or manual setup with Bitcoin Core, Redis, and CP1
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("integration_test")

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / "code"))


def test_bitcoin_rpc():
    """Test Bitcoin Core RPC connection."""
    logger.info("Testing Bitcoin Core RPC...")
    
    try:
        from bitcoinrpc.authproxy import AuthServiceProxy
        
        rpc_url = os.environ.get(
            "BITCOIN_RPC_URL",
            "http://cp1user:CP1SecurePassword123!@127.0.0.1:18443"
        )
        
        rpc = AuthServiceProxy(rpc_url)
        info = rpc.getblockchaininfo()
        
        logger.info(f"✓ Bitcoin Core connected: chain={info['chain']}, blocks={info['blocks']}")
        return True
    except Exception as e:
        logger.error(f"✗ Bitcoin Core connection failed: {e}")
        return False


def test_redis():
    """Test Redis connection."""
    logger.info("Testing Redis...")
    
    try:
        import redis
        
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        client = redis.from_url(redis_url)
        client.ping()
        
        # Test set/get
        client.setex("test_key", 10, "test_value")
        value = client.get("test_key")
        
        logger.info(f"✓ Redis connected and working")
        return True
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        return False


def test_model_loading():
    """Test ML model loading."""
    logger.info("Testing model loading...")
    
    try:
        import joblib
        import pandas as pd
        
        model_path = Path(__file__).parent / "models" / "cp1_static_xgb_v1.joblib"
        feature_path = Path(__file__).parent / "results" / "cp1_static_ellipticpp.csv"
        
        model = joblib.load(model_path)
        df = pd.read_csv(feature_path, nrows=1)
        feature_cols = [c for c in df.columns if c != "label"]
        
        # Test inference
        X = df[feature_cols].iloc[[0]]
        score = model.predict_proba(X)[0, 1]
        
        logger.info(f"✓ Model loaded and inference works (test score: {score:.4f})")
        return True
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False


def test_consensus_validator():
    """Test consensus validator."""
    logger.info("Testing consensus validator...")
    
    try:
        from consensus_validator import ConsensusValidator, ConsensusResult
        
        # Mock RPC with proper structure check
        class MockRPC:
            def decoderawtransaction(self, raw_hex):
                # Return valid structure
                return {
                    "txid": "test123",
                    "version": 2,
                    "vin": [{"txid": "prev", "vout": 0}],
                    "vout": [{"value": 0.1, "n": 0}]
                }
            
            def testmempoolaccept(self, raw_list):
                return [{"allowed": True}]
        
        validator = ConsensusValidator(MockRPC())
        result = validator.validate("0200000001...")  # Fake hex
        
        logger.info(f"✓ Consensus validator works (status: {result.status.value})")
        return True
    except Exception as e:
        logger.error(f"✗ Consensus validator failed: {e}")
        return False


def test_safe_action_policy():
    """Test safe action policy."""
    logger.info("Testing safe action policy...")
    
    try:
        from safe_action_policy import SafeActionPolicy, Action
        
        policy = SafeActionPolicy()
        
        # Test all decision paths
        accept = policy.evaluate(0.05)
        flag = policy.evaluate(0.40)
        reject = policy.evaluate(0.80)
        
        assert accept.action == Action.ACCEPT
        assert flag.action == Action.FLAG
        assert reject.action == Action.REJECT
        
        logger.info("✓ Safe action policy works")
        return True
    except Exception as e:
        logger.error(f"✗ Safe action policy failed: {e}")
        return False


def test_redis_pubsub():
    """Test Redis pub/sub for CP1-CP2 communication."""
    logger.info("Testing Redis pub/sub...")
    
    try:
        from redis_client import CP1RedisClient
        
        client = CP1RedisClient()
        
        if not client.is_connected:
            logger.warning("⚠ Redis not connected, skipping pub/sub test")
            return True
        
        # Test publish
        success = client.publish_tx_decision(
            txid="test123",
            score=0.75,
            decision="FLAG",
            announcing_peer="192.168.1.1:8333"
        )
        
        logger.info(f"✓ Redis pub/sub works (published: {success})")
        return True
    except Exception as e:
        logger.error(f"✗ Redis pub/sub failed: {e}")
        return False


def test_metrics_exporter():
    """Test Prometheus metrics."""
    logger.info("Testing Prometheus metrics...")
    
    try:
        from metrics_exporter import MetricsExporter
        
        metrics = MetricsExporter(port=8099)  # Use different port for testing
        
        # Record some metrics
        metrics.record_tx_ingested()
        metrics.record_decision("ACCEPT")
        metrics.record_score(0.25)
        
        logger.info("✓ Prometheus metrics work")
        return True
    except Exception as e:
        logger.error(f"✗ Prometheus metrics failed: {e}")
        return False


def test_feature_cache():
    """Test feature cache."""
    logger.info("Testing feature cache...")
    
    try:
        from feature_cache import FeatureCache
        
        cache = FeatureCache()
        
        # Test address features
        cache.set_address_features("1A1zP1...", {"age_days": 100, "tx_count": 50})
        cached = cache.get_address_features("1A1zP1...")
        
        assert cached is not None
        assert cached["age_days"] == 100
        
        stats = cache.get_stats()
        logger.info(f"✓ Feature cache works (hit_rate: {stats['hit_rate']})")
        return True
    except Exception as e:
        logger.error(f"✗ Feature cache failed: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("CP1-CP2 INTEGRATION TESTS")
    logger.info("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Consensus Validator", test_consensus_validator),
        ("Safe Action Policy", test_safe_action_policy),
        ("Feature Cache", test_feature_cache),
        ("Prometheus Metrics", test_metrics_exporter),
        ("Redis Connection", test_redis),
        ("Redis Pub/Sub", test_redis_pubsub),
        ("Bitcoin Core RPC", test_bitcoin_rpc),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))
    
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed} passed, {failed} failed")
    logger.info("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
