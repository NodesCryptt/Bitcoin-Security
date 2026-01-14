#!/usr/bin/env python
"""
CP1 SHAP Logging Wrapper Tests
==============================
Tests for the SHAP logging wrapper component.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pandas as pd

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from shap_logging_wrapper import (
    SHAPLoggingWrapper,
    ExplanationLog,
    SOCTriagePacket
)


# =============================================================================
# EXPLANATION LOG TESTS
# =============================================================================

class TestExplanationLog:
    """Tests for ExplanationLog dataclass."""
    
    def test_to_dict(self):
        """Converts to dict correctly."""
        log = ExplanationLog(
            txid="abc123",
            score=0.75,
            decision="FLAG",
            timestamp="2024-01-01T00:00:00",
            model_version="v1",
            top_features=[{"feature_name": "size", "shap_value": 0.3}],
            human_reason="Large transaction size",
            raw_features={"size": 1000},
            processing_latency_ms=50.0,
            explanation_latency_ms=100.0
        )
        
        d = log.to_dict()
        assert d["txid"] == "abc123"
        assert d["score"] == 0.75
        assert d["decision"] == "FLAG"
        assert len(d["top_features"]) == 1
    
    def test_to_json(self):
        """Converts to JSON correctly."""
        log = ExplanationLog(
            txid="test",
            score=0.5,
            decision="FLAG",
            timestamp="2024-01-01",
            model_version="v1",
            top_features=[],
            human_reason="Test",
            raw_features={},
            processing_latency_ms=0,
            explanation_latency_ms=0
        )
        
        json_str = log.to_json()
        parsed = json.loads(json_str)
        assert parsed["txid"] == "test"


# =============================================================================
# SOC TRIAGE PACKET TESTS
# =============================================================================

class TestSOCTriagePacket:
    """Tests for SOC triage packet generation."""
    
    def test_severity_levels(self):
        """Severity is assigned correctly based on score."""
        # CRITICAL (>= 0.8)
        log = ExplanationLog(
            txid="t1", score=0.9, decision="REJECT",
            timestamp="2024-01-01", model_version="v1",
            top_features=[{"feature_name": "f1"}],
            human_reason="High risk",
            raw_features={"size": 100},
            processing_latency_ms=0, explanation_latency_ms=0
        )
        packet = SOCTriagePacket.from_explanation(log)
        assert packet.severity == "CRITICAL"
        
        # HIGH (>= 0.6)
        log.score = 0.7
        packet = SOCTriagePacket.from_explanation(log)
        assert packet.severity == "HIGH"
        
        # MEDIUM (>= 0.4)
        log.score = 0.5
        packet = SOCTriagePacket.from_explanation(log)
        assert packet.severity == "MEDIUM"
        
        # LOW
        log.score = 0.3
        packet = SOCTriagePacket.from_explanation(log)
        assert packet.severity == "LOW"
    
    def test_risk_indicators(self):
        """Risk indicators are detected correctly."""
        log = ExplanationLog(
            txid="t1", score=0.7, decision="FLAG",
            timestamp="2024-01-01", model_version="v1",
            top_features=[],
            human_reason="Test",
            raw_features={
                "is_double_spend": 1.0,
                "any_illicit_addr": 1.0,
                "fees": 0,
                "num_input_addresses": 15,
                "num_output_addresses": 25
            },
            processing_latency_ms=0, explanation_latency_ms=0
        )
        
        packet = SOCTriagePacket.from_explanation(log)
        
        assert "DOUBLE_SPEND_DETECTED" in packet.risk_indicators
        assert "ILLICIT_ADDRESS_INPUT" in packet.risk_indicators
        assert "ZERO_FEE" in packet.risk_indicators
        assert "HIGH_INPUT_COUNT" in packet.risk_indicators
        assert "HIGH_OUTPUT_COUNT" in packet.risk_indicators


# =============================================================================
# SHAP LOGGING WRAPPER TESTS
# =============================================================================

class TestSHAPLoggingWrapper:
    """Tests for SHAP logging wrapper."""
    
    @pytest.fixture
    def mock_explainer(self):
        """Create mock SHAP explainer."""
        explainer = Mock()
        
        # Mock explanation result
        mock_explanation = Mock()
        mock_explanation.top_features = [
            Mock(to_dict=lambda: {"feature_name": "size", "shap_value": 0.3})
        ]
        mock_explanation.human_reason = "Large transaction"
        mock_explanation.raw_features = {"size": 1000}
        
        explainer.explain.return_value = mock_explanation
        return explainer
    
    @pytest.fixture
    def wrapper(self, mock_explainer, tmp_path):
        """Create wrapper with temp directories."""
        return SHAPLoggingWrapper(
            explainer=mock_explainer,
            output_dir=tmp_path / "explanations",
            soc_output_dir=tmp_path / "soc",
            async_saving=False  # Sync for testing
        )
    
    def test_explain_and_log(self, wrapper, mock_explainer):
        """Explanation is generated and logged."""
        features = pd.DataFrame([{"size": 1000}])
        
        log = wrapper.explain_and_log(
            features=features,
            txid="test_tx",
            score=0.6,
            decision="FLAG"
        )
        
        assert log.txid == "test_tx"
        assert log.score == 0.6
        assert log.decision == "FLAG"
        mock_explainer.explain.assert_called_once()
    
    def test_files_created(self, wrapper, tmp_path):
        """Log files are created."""
        features = pd.DataFrame([{"size": 100}])
        
        wrapper.explain_and_log(
            features=features,
            txid="file_test",
            score=0.7,
            decision="REJECT"
        )
        
        # Check explanation dir
        exp_dir = tmp_path / "explanations"
        assert exp_dir.exists()
        assert len(list(exp_dir.glob("*.json*"))) > 0
        
        # Check SOC dir
        soc_dir = tmp_path / "soc"
        assert soc_dir.exists()
    
    def test_stats(self, wrapper):
        """Stats are tracked."""
        features = pd.DataFrame([{"size": 100}])
        
        wrapper.explain_and_log(features, "t1", 0.5, "FLAG")
        wrapper.explain_and_log(features, "t2", 0.6, "FLAG")
        
        stats = wrapper.get_stats()
        assert stats["explanations_generated"] == 2
        assert stats["saves_completed"] == 2


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
