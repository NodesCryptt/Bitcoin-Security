"""
CP1 Feature Extraction Unit Tests
==================================
Tests for deterministic feature extraction from raw transactions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add tests directory to path for imports
TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))

# Import sample data - pytest auto-loads conftest, but we need explicit import for constants
import conftest
SAMPLE_P2PKH_TX = conftest.SAMPLE_P2PKH_TX
SAMPLE_COINBASE_TX = conftest.SAMPLE_COINBASE_TX
SAMPLE_SEGWIT_TX = conftest.SAMPLE_SEGWIT_TX
SAMPLE_OP_RETURN_TX = conftest.SAMPLE_OP_RETURN_TX
SAMPLE_LARGE_TX = conftest.SAMPLE_LARGE_TX
SAMPLE_MISSING_VIN_TX = conftest.SAMPLE_MISSING_VIN_TX
SAMPLE_MEMPOOL_TX = conftest.SAMPLE_MEMPOOL_TX


# =============================================================================
# FEATURE EXTRACTION FUNCTION (Minimal - matches cp1_live_infer_runtime.py)
# =============================================================================

def extract_minimal_features(decoded: dict) -> dict:
    """
    Extract minimal features from decoded transaction.
    Must handle all edge cases gracefully.
    """
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
# BASIC TRANSACTION TESTS
# =============================================================================

class TestBasicFeatureExtraction:
    """Tests for basic feature extraction functionality."""
    
    def test_p2pkh_tx_features(self, sample_p2pkh_tx):
        """Standard P2PKH transaction produces expected features."""
        features = extract_minimal_features(sample_p2pkh_tx)
        
        assert features["size"] == 226.0
        assert features["num_input_addresses"] == 2
        assert features["num_output_addresses"] == 2
        assert abs(features["total_BTC"] - 0.5001) < 0.0001
        assert features["fees"] == 0.0  # No fee in raw decode
    
    def test_all_features_numeric(self, sample_p2pkh_tx):
        """All extracted features must be numeric (float/int)."""
        features = extract_minimal_features(sample_p2pkh_tx)
        
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric: {type(value)}"
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is infinite"
    
    def test_features_non_negative(self, sample_p2pkh_tx):
        """All features should be non-negative."""
        features = extract_minimal_features(sample_p2pkh_tx)
        
        for key, value in features.items():
            assert value >= 0, f"{key} is negative: {value}"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestCoinbaseTransaction:
    """Tests for coinbase (mining reward) transactions."""
    
    def test_coinbase_features(self, sample_coinbase_tx):
        """Coinbase transaction has 1 input (coinbase marker)."""
        features = extract_minimal_features(sample_coinbase_tx)
        
        assert features["size"] == 164.0
        assert features["num_input_addresses"] == 1  # Coinbase entry
        assert features["num_output_addresses"] == 1
        assert features["total_BTC"] == 6.25  # Block reward
    
    def test_coinbase_no_crash(self, sample_coinbase_tx):
        """Coinbase tx doesn't crash despite missing prevout."""
        try:
            features = extract_minimal_features(sample_coinbase_tx)
            assert True
        except Exception as e:
            pytest.fail(f"Coinbase extraction crashed: {e}")


class TestSegwitTransaction:
    """Tests for SegWit transactions."""
    
    def test_segwit_features(self, sample_segwit_tx):
        """SegWit transaction features extracted correctly."""
        features = extract_minimal_features(sample_segwit_tx)
        
        assert features["size"] == 222.0
        assert features["num_input_addresses"] == 1
        assert features["num_output_addresses"] == 1
        assert abs(features["total_BTC"] - 0.1) < 0.0001
    
    def test_segwit_witness_ignored(self, sample_segwit_tx):
        """Witness data should not cause extraction issues."""
        # Ensure txinwitness field doesn't break extraction
        features = extract_minimal_features(sample_segwit_tx)
        assert features is not None
        assert len(features) == 5


class TestOpReturnTransaction:
    """Tests for OP_RETURN data embedding transactions."""
    
    def test_op_return_features(self, sample_op_return_tx):
        """OP_RETURN transaction with zero-value output."""
        features = extract_minimal_features(sample_op_return_tx)
        
        assert features["size"] == 245.0
        assert features["num_output_addresses"] == 2  # OP_RETURN + change
        assert abs(features["total_BTC"] - 0.001) < 0.0001  # Only change output
    
    def test_op_return_zero_value(self, sample_op_return_tx):
        """Zero-value OP_RETURN output handled correctly."""
        # OP_RETURN outputs have value=0
        features = extract_minimal_features(sample_op_return_tx)
        # Should not count OP_RETURN value as negative
        assert features["total_BTC"] >= 0


class TestLargeTransaction:
    """Tests for large transactions with many inputs/outputs."""
    
    def test_large_tx_features(self, sample_large_tx):
        """Large transaction with 50 inputs, 100 outputs."""
        features = extract_minimal_features(sample_large_tx)
        
        assert features["size"] == 15000.0
        assert features["num_input_addresses"] == 50
        assert features["num_output_addresses"] == 100
        assert abs(features["total_BTC"] - 1.0) < 0.0001  # 100 * 0.01
    
    def test_large_tx_performance(self, sample_large_tx):
        """Large transaction extraction should be fast (<100ms)."""
        import time
        
        start = time.perf_counter()
        for _ in range(100):
            extract_minimal_features(sample_large_tx)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms < 100, f"Extraction too slow: {avg_ms:.2f}ms per tx"


class TestMissingVinData:
    """Tests for transactions with missing or empty vin."""
    
    def test_missing_vin_features(self, sample_missing_vin_tx):
        """Empty vin list handled gracefully."""
        features = extract_minimal_features(sample_missing_vin_tx)
        
        assert features["num_input_addresses"] == 0
        assert features["num_output_addresses"] == 1
        assert features["total_BTC"] == 0.1
    
    def test_missing_vin_no_crash(self, sample_missing_vin_tx):
        """No crash on empty vin list."""
        try:
            features = extract_minimal_features(sample_missing_vin_tx)
            assert True
        except Exception as e:
            pytest.fail(f"Missing vin extraction crashed: {e}")


class TestMempoolTransaction:
    """Tests for transactions with mempool context (fee info)."""
    
    def test_mempool_fee_extraction(self, sample_mempool_tx):
        """Fee extracted from mempool context."""
        features = extract_minimal_features(sample_mempool_tx)
        
        assert features["fees"] == 0.0001
        assert features["size"] == 250.0


# =============================================================================
# ROBUSTNESS TESTS
# =============================================================================

class TestRobustness:
    """Tests for handling malformed or unexpected input."""
    
    def test_empty_dict(self):
        """Empty dict should return zero features."""
        features = extract_minimal_features({})
        
        assert features["size"] == 0.0
        assert features["fees"] == 0.0
        assert features["num_input_addresses"] == 0
        assert features["num_output_addresses"] == 0
        assert features["total_BTC"] == 0.0
    
    def test_none_values(self):
        """None values in fields handled gracefully."""
        tx = {
            "size": None,
            "fee": None,
            "vin": None,
            "vout": None
        }
        
        features = extract_minimal_features(tx)
        
        assert features["size"] == 0.0
        assert features["fees"] == 0.0
        assert features["num_input_addresses"] == 0
        assert features["num_output_addresses"] == 0
    
    def test_string_size(self):
        """String size value coerced to float."""
        tx = {"size": "256", "vin": [], "vout": []}
        features = extract_minimal_features(tx)
        
        assert features["size"] == 256.0
    
    def test_invalid_string_size(self):
        """Non-numeric string size defaults to 0."""
        tx = {"size": "invalid", "vin": [], "vout": []}
        features = extract_minimal_features(tx)
        
        assert features["size"] == 0.0
    
    def test_negative_value(self):
        """Negative values in vout handled (shouldn't happen in real data)."""
        tx = {
            "size": 100,
            "vin": [],
            "vout": [{"value": -0.5}]  # Invalid but test robustness
        }
        features = extract_minimal_features(tx)
        
        # Should still extract without crash
        assert isinstance(features["total_BTC"], float)
    
    def test_mixed_vout_types(self):
        """Mixed types in vout list handled."""
        tx = {
            "size": 100,
            "vin": [],
            "vout": [
                {"value": 0.1},
                None,
                "invalid",
                {"value": 0.2}
            ]
        }
        features = extract_minimal_features(tx)
        
        assert abs(features["total_BTC"] - 0.3) < 0.0001


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests to ensure feature extraction is deterministic."""
    
    def test_same_input_same_output(self, sample_p2pkh_tx):
        """Same input always produces same output."""
        features1 = extract_minimal_features(sample_p2pkh_tx)
        features2 = extract_minimal_features(sample_p2pkh_tx)
        
        assert features1 == features2
    
    def test_determinism_across_runs(self, sample_p2pkh_tx):
        """Multiple runs produce identical results."""
        results = [extract_minimal_features(sample_p2pkh_tx) for _ in range(100)]
        
        reference = results[0]
        for result in results[1:]:
            assert result == reference


# =============================================================================
# DATAFRAME CONVERSION TESTS
# =============================================================================

class TestDataFrameConversion:
    """Tests for converting features to DataFrame for model inference."""
    
    def test_features_to_dataframe(self, sample_p2pkh_tx):
        """Features can be converted to DataFrame."""
        features = extract_minimal_features(sample_p2pkh_tx)
        df = pd.DataFrame([features])
        
        assert len(df) == 1
        assert set(df.columns) == set(features.keys())
    
    def test_dataframe_numeric_types(self, sample_p2pkh_tx):
        """DataFrame columns are numeric types."""
        features = extract_minimal_features(sample_p2pkh_tx)
        df = pd.DataFrame([features])
        
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"
    
    def test_dataframe_no_nulls(self, sample_missing_vin_tx):
        """DataFrame has no null values after conversion."""
        features = extract_minimal_features(sample_missing_vin_tx)
        df = pd.DataFrame([features])
        
        assert not df.isnull().any().any()
    
    def test_batch_conversion(self):
        """Multiple transactions converted to batch DataFrame."""
        txs = [SAMPLE_P2PKH_TX, SAMPLE_SEGWIT_TX, SAMPLE_OP_RETURN_TX]
        features_list = [extract_minimal_features(tx) for tx in txs]
        df = pd.DataFrame(features_list)
        
        assert len(df) == 3
        assert not df.isnull().any().any()


# =============================================================================
# FEATURE SCHEMA TESTS
# =============================================================================

class TestFeatureSchema:
    """Tests for feature schema consistency."""
    
    EXPECTED_FEATURES = {"size", "fees", "num_input_addresses", "num_output_addresses", "total_BTC"}
    
    def test_all_expected_features_present(self, sample_p2pkh_tx):
        """All expected features are present in output."""
        features = extract_minimal_features(sample_p2pkh_tx)
        
        assert set(features.keys()) == self.EXPECTED_FEATURES
    
    def test_no_extra_features(self, sample_p2pkh_tx):
        """No unexpected features in output."""
        features = extract_minimal_features(sample_p2pkh_tx)
        
        extra = set(features.keys()) - self.EXPECTED_FEATURES
        assert len(extra) == 0, f"Unexpected features: {extra}"
    
    def test_consistent_schema_across_tx_types(self):
        """All transaction types produce same feature schema."""
        txs = [
            SAMPLE_P2PKH_TX,
            SAMPLE_COINBASE_TX,
            SAMPLE_SEGWIT_TX,
            SAMPLE_OP_RETURN_TX,
            SAMPLE_LARGE_TX,
            SAMPLE_MISSING_VIN_TX,
        ]
        
        for tx in txs:
            features = extract_minimal_features(tx)
            assert set(features.keys()) == self.EXPECTED_FEATURES


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
