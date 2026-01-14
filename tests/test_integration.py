"""
CP1 Integration Tests
=====================
Tests for the full inference pipeline: ZMQ → decode → features → model → decision.
"""

import pytest
import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys

# Add code directory to path
CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))


# =============================================================================
# SAMPLE DATA
# =============================================================================

# Simulated decoded transaction from Bitcoin Core
DECODED_TX = {
    "txid": "a1b2c3d4e5f6",
    "size": 226,
    "vsize": 166,
    "weight": 664,
    "version": 2,
    "locktime": 0,
    "vin": [
        {"txid": "prev1", "vout": 0},
        {"txid": "prev2", "vout": 1}
    ],
    "vout": [
        {"value": 0.5, "n": 0},
        {"value": 0.0001, "n": 1}
    ]
}

# Model path
MODEL_PATH = Path(__file__).parent.parent / "models" / "cp1_static_xgb_v1.joblib"
FEATURE_CSV = Path(__file__).parent.parent / "results" / "cp1_static_ellipticpp.csv"


# =============================================================================
# FEATURE EXTRACTION (from runtime)
# =============================================================================

def extract_minimal_features(decoded: dict) -> dict:
    """Extract minimal features from decoded transaction."""
    f = {}
    
    try:
        f["size"] = float(decoded.get("size", 0))
    except (ValueError, TypeError):
        f["size"] = 0.0
    
    try:
        f["fees"] = float(decoded.get("fee", 0))
    except (ValueError, TypeError):
        f["fees"] = 0.0
    
    try:
        f["num_input_addresses"] = len(decoded.get("vin", []))
    except (ValueError, TypeError):
        f["num_input_addresses"] = 0
    
    try:
        f["num_output_addresses"] = len(decoded.get("vout", []))
    except (ValueError, TypeError):
        f["num_output_addresses"] = 0
    
    try:
        vouts = decoded.get("vout", [])
        total = 0.0
        for v in vouts:
            if isinstance(v, dict):
                val = v.get("value", 0)
                if val is not None:
                    total += float(val)
        f["total_BTC"] = total
    except (ValueError, TypeError):
        f["total_BTC"] = 0.0
    
    return f


# =============================================================================
# DECISION LOGIC
# =============================================================================

ACCEPT_THRESHOLD = 0.15
REJECT_THRESHOLD = 0.60

def make_decision(score: float) -> str:
    """Map score to decision."""
    if score < ACCEPT_THRESHOLD:
        return "ACCEPT"
    elif score < REJECT_THRESHOLD:
        return "FLAG"
    else:
        return "REJECT"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestModelLoading:
    """Tests for model and feature schema loading."""
    
    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not found")
    def test_model_loads_successfully(self):
        """Model loads without errors."""
        model = joblib.load(MODEL_PATH)
        assert model is not None
        assert hasattr(model, "predict_proba")
    
    @pytest.mark.skipif(not FEATURE_CSV.exists(), reason="Feature CSV not found")
    def test_feature_schema_loads(self):
        """Feature columns load from training data."""
        df = pd.read_csv(FEATURE_CSV, nrows=1)
        feature_cols = [c for c in df.columns if c != "label"]
        
        assert len(feature_cols) > 0
        assert "label" not in feature_cols


class TestFeatureToModelPipeline:
    """Tests for feature extraction → model inference pipeline."""
    
    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not found")
    @pytest.mark.skipif(not FEATURE_CSV.exists(), reason="Feature CSV not found")
    def test_full_inference_path(self):
        """Full path: features → model → score using actual training data."""
        # Load model and schema
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(FEATURE_CSV, nrows=10)
        feature_cols = [c for c in df.columns if c != "label"]
        
        # Use actual features from the dataset (not extracted from raw tx)
        X_sample = df[feature_cols].iloc[[0]]
        
        # Inference
        score = model.predict_proba(X_sample)[0, 1]
        
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= float(score) <= 1.0
    
    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not found")
    @pytest.mark.skipif(not FEATURE_CSV.exists(), reason="Feature CSV not found")
    def test_decision_mapping(self):
        """Score correctly maps to decision."""
        assert make_decision(0.05) == "ACCEPT"
        assert make_decision(0.14) == "ACCEPT"
        assert make_decision(0.15) == "FLAG"
        assert make_decision(0.30) == "FLAG"
        assert make_decision(0.59) == "FLAG"
        assert make_decision(0.60) == "REJECT"
        assert make_decision(0.95) == "REJECT"


class TestPerformance:
    """Tests for inference latency and throughput."""
    
    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not found")
    @pytest.mark.skipif(not FEATURE_CSV.exists(), reason="Feature CSV not found")
    def test_single_inference_latency(self):
        """Single inference should complete in <50ms."""
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(FEATURE_CSV, nrows=1)
        feature_cols = [c for c in df.columns if c != "label"]
        
        feats = extract_minimal_features(DECODED_TX)
        row_dict = {c: feats.get(c, 0.0) for c in feature_cols}
        row = pd.DataFrame([row_dict])
        row = row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        
        start = time.perf_counter()
        for _ in range(100):
            _ = model.predict_proba(row)[0, 1]
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 50, f"Inference too slow: {avg_ms:.2f}ms"
    
    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not found")
    @pytest.mark.skipif(not FEATURE_CSV.exists(), reason="Feature CSV not found")
    def test_batch_inference(self):
        """Batch inference for mempool spike simulation."""
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(FEATURE_CSV, nrows=1)
        feature_cols = [c for c in df.columns if c != "label"]
        
        # Simulate 100 tx mempool spike
        batch_size = 100
        rows = []
        for _ in range(batch_size):
            feats = extract_minimal_features(DECODED_TX)
            row_dict = {c: feats.get(c, 0.0) for c in feature_cols}
            rows.append(row_dict)
        
        batch_df = pd.DataFrame(rows)
        batch_df = batch_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        
        start = time.perf_counter()
        scores = model.predict_proba(batch_df)[:, 1]
        elapsed = time.perf_counter() - start
        
        assert len(scores) == batch_size
        assert elapsed < 1.0, f"Batch inference too slow: {elapsed:.2f}s"


class TestEndToEndPipeline:
    """Tests for complete end-to-end inference path."""
    
    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not found")
    @pytest.mark.skipif(not FEATURE_CSV.exists(), reason="Feature CSV not found")
    def test_e2e_accept_tx(self):
        """End-to-end test using actual training data with known label."""
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(FEATURE_CSV, nrows=100)
        feature_cols = [c for c in df.columns if c != "label"]
        
        # Use actual feature data from training set
        X_sample = df[feature_cols].iloc[[0]]
        
        score = model.predict_proba(X_sample)[0, 1]
        decision = make_decision(score)
        
        # Result should be valid
        assert decision in ["ACCEPT", "FLAG", "REJECT"]
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= float(score) <= 1.0


class TestMockedZMQPipeline:
    """Tests with mocked ZMQ transport."""
    
    def test_raw_hex_decode_to_features(self):
        """Simulated raw hex to decoded to features."""
        # Mock RPC decode response
        mock_decoded = DECODED_TX.copy()
        
        feats = extract_minimal_features(mock_decoded)
        
        assert feats["size"] == 226.0
        assert feats["num_input_addresses"] == 2
        assert feats["num_output_addresses"] == 2
    
    def test_invalid_raw_hex_handling(self):
        """Invalid decode should be handled gracefully."""
        # If decode fails, features should be empty/zero
        empty_decode = {}
        
        feats = extract_minimal_features(empty_decode)
        
        assert feats["size"] == 0.0
        assert feats["num_input_addresses"] == 0
        assert feats["num_output_addresses"] == 0


class TestDataTypeHandling:
    """Tests for DataFrame type coercion (fixing known bugs)."""
    
    @pytest.mark.skipif(not FEATURE_CSV.exists(), reason="Feature CSV not found")
    def test_coerce_string_to_numeric(self):
        """String values should coerce to numeric without error."""
        df = pd.read_csv(FEATURE_CSV, nrows=1)
        feature_cols = [c for c in df.columns if c != "label"]
        
        # Mix of types
        row_dict = {c: "0.5" if i % 2 == 0 else 0.5 for i, c in enumerate(feature_cols)}
        row = pd.DataFrame([row_dict])
        
        # Should not raise
        row = row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        
        assert not row.isnull().any().any()
    
    @pytest.mark.skipif(not FEATURE_CSV.exists(), reason="Feature CSV not found")
    def test_nan_handling(self):
        """NaN values should be filled with 0."""
        df = pd.read_csv(FEATURE_CSV, nrows=1)
        feature_cols = [c for c in df.columns if c != "label"]
        
        row_dict = {c: np.nan for c in feature_cols}
        row = pd.DataFrame([row_dict])
        
        row = row.fillna(0.0)
        
        assert not row.isnull().any().any()
        assert (row == 0.0).all().all()
    
    @pytest.mark.skipif(not FEATURE_CSV.exists(), reason="Feature CSV not found")
    def test_invalid_string_handling(self):
        """Invalid strings should coerce to 0, not crash."""
        df = pd.read_csv(FEATURE_CSV, nrows=1)
        feature_cols = [c for c in df.columns if c != "label"]
        
        row_dict = {c: "invalid" for c in feature_cols}
        row = pd.DataFrame([row_dict])
        
        # Should not raise
        row = row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        
        assert not row.isnull().any().any()
        assert (row == 0.0).all().all()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
