"""
CP1 SHAP Explainer
==================
Provides SHAP-based explainability for CP1 model predictions.

For every FLAG/REJECT decision, this module generates:
- Top 5 contributing features with SHAP values
- Human-readable reason string
- Full explanation JSON for audit logging

The explanations are stored in results/explanations/ for review.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import SHAP, with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Using feature importance fallback.")

# Explanation storage directory
EXPLANATIONS_DIR = Path(__file__).parent.parent / "results" / "explanations"


@dataclass
class FeatureContribution:
    """A single feature's contribution to the prediction."""
    feature_name: str
    shap_value: float
    feature_value: float
    contribution_direction: str  # "increases_risk" or "decreases_risk"
    
    def to_dict(self) -> dict:
        return {
            "feature": self.feature_name,
            "shap_value": round(self.shap_value, 6),
            "feature_value": round(self.feature_value, 6),
            "direction": self.contribution_direction
        }


@dataclass
class Explanation:
    """Complete explanation for a prediction."""
    txid: str
    score: float
    decision: str
    top_features: List[FeatureContribution]
    base_value: float
    human_reason: str
    raw_features: Dict[str, float]
    timestamp: str
    model_version: str = "v1"
    
    def to_dict(self) -> dict:
        return {
            "txid": self.txid,
            "timestamp": self.timestamp,
            "score": round(self.score, 4),
            "decision": self.decision,
            "model_version": self.model_version,
            "base_value": round(self.base_value, 4),
            "top_features": [f.to_dict() for f in self.top_features],
            "human_reason": self.human_reason,
            "raw_features": {k: round(v, 6) for k, v in self.raw_features.items()}
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SHAPExplainer:
    """
    Generates SHAP explanations for XGBoost model predictions.
    
    Usage:
        explainer = SHAPExplainer(model, feature_names)
        explanation = explainer.explain(
            features=feature_df,
            txid="abc123",
            score=0.72,
            decision="REJECT"
        )
        explainer.save_explanation(explanation)
    """
    
    def __init__(
        self, 
        model, 
        feature_names: List[str],
        model_version: str = "v1"
    ):
        """
        Initialize explainer.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature column names
            model_version: Version string for model tracking
        """
        self.model = model
        self.feature_names = feature_names
        self.model_version = model_version
        self._shap_explainer = None
        
        if SHAP_AVAILABLE:
            try:
                self._shap_explainer = shap.TreeExplainer(model)
                logger.info("SHAP TreeExplainer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP: {e}")
    
    def explain(
        self,
        features: pd.DataFrame,
        txid: str,
        score: float,
        decision: str,
        top_k: int = 5
    ) -> Explanation:
        """
        Generate explanation for a prediction.
        
        Args:
            features: DataFrame with feature values (single row)
            txid: Transaction ID
            score: Model prediction score
            decision: Decision string (ACCEPT/FLAG/REJECT)
            top_k: Number of top features to include
        
        Returns:
            Explanation object
        """
        if len(features) != 1:
            raise ValueError("Features must be a single-row DataFrame")
        
        # Get raw feature values
        raw_features = features.iloc[0].to_dict()
        
        # Calculate SHAP values or use feature importance fallback
        if self._shap_explainer is not None:
            top_contributions, base_value = self._get_shap_contributions(
                features, top_k
            )
        else:
            top_contributions, base_value = self._get_importance_contributions(
                features, top_k
            )
        
        # Generate human-readable reason
        human_reason = self._generate_reason_string(top_contributions)
        
        return Explanation(
            txid=txid,
            score=score,
            decision=decision,
            top_features=top_contributions,
            base_value=base_value,
            human_reason=human_reason,
            raw_features=raw_features,
            timestamp=datetime.utcnow().isoformat(),
            model_version=self.model_version
        )
    
    def _get_shap_contributions(
        self, 
        features: pd.DataFrame, 
        top_k: int
    ) -> Tuple[List[FeatureContribution], float]:
        """Get top feature contributions using SHAP."""
        shap_values = self._shap_explainer.shap_values(features)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification: use class 1 (illicit)
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif hasattr(shap_values, 'values'):
            values = shap_values.values[0]
        else:
            values = shap_values[0]
        
        base_value = self._shap_explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        # Pair features with SHAP values
        feature_contributions = list(zip(self.feature_names, values, features.iloc[0].values))
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top k
        top_contributions = []
        for name, shap_val, feat_val in feature_contributions[:top_k]:
            direction = "increases_risk" if shap_val > 0 else "decreases_risk"
            top_contributions.append(FeatureContribution(
                feature_name=name,
                shap_value=float(shap_val),
                feature_value=float(feat_val),
                contribution_direction=direction
            ))
        
        return top_contributions, float(base_value)
    
    def _get_importance_contributions(
        self, 
        features: pd.DataFrame, 
        top_k: int
    ) -> Tuple[List[FeatureContribution], float]:
        """
        Fallback: Use feature importance when SHAP not available.
        
        This provides less accurate per-instance explanations but
        still gives insight into which features matter.
        """
        importances = self.model.feature_importances_
        
        # Pair features with importances and values
        feature_data = list(zip(self.feature_names, importances, features.iloc[0].values))
        
        # Sort by importance
        feature_data.sort(key=lambda x: x[1], reverse=True)
        
        top_contributions = []
        for name, importance, feat_val in feature_data[:top_k]:
            # Use importance as pseudo-SHAP (not directional)
            top_contributions.append(FeatureContribution(
                feature_name=name,
                shap_value=float(importance),
                feature_value=float(feat_val),
                contribution_direction="importance"  # Not directional
            ))
        
        # Base value is mean prediction
        base_value = 0.5  # Approximate
        
        return top_contributions, base_value
    
    def _generate_reason_string(
        self, 
        contributions: List[FeatureContribution]
    ) -> str:
        """Generate a human-readable explanation string."""
        if not contributions:
            return "No significant contributing features identified."
        
        # Map feature names to readable descriptions
        feature_descriptions = {
            "size": "transaction size",
            "vsize": "virtual size",
            "fees": "fee amount",
            "num_input_addresses": "number of inputs",
            "num_output_addresses": "number of outputs",
            "total_BTC": "total BTC transferred",
            "fee_rate": "fee rate",
        }
        
        reasons = []
        for i, contrib in enumerate(contributions[:3]):  # Top 3 for readability
            name = feature_descriptions.get(
                contrib.feature_name, 
                contrib.feature_name.replace("_", " ")
            )
            direction = "high" if contrib.shap_value > 0 else "low"
            reasons.append(f"{direction} {name}")
        
        if len(reasons) == 1:
            return f"Flagged due to {reasons[0]}."
        elif len(reasons) == 2:
            return f"Flagged due to {reasons[0]} and {reasons[1]}."
        else:
            return f"Flagged due to {', '.join(reasons[:-1])}, and {reasons[-1]}."
    
    def save_explanation(
        self, 
        explanation: Explanation,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save explanation to JSON file.
        
        Args:
            explanation: Explanation to save
            output_dir: Directory to save to (default: results/explanations/)
        
        Returns:
            Path to saved file
        """
        output_dir = output_dir or EXPLANATIONS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp and txid
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        txid_short = explanation.txid[:8] if len(explanation.txid) >= 8 else explanation.txid
        filename = f"{timestamp}_{txid_short}_{explanation.decision.lower()}.json"
        
        filepath = output_dir / filename
        
        with open(filepath, "w") as f:
            f.write(explanation.to_json())
        
        logger.info(f"Saved explanation to {filepath}")
        return filepath


def create_explainer(model, feature_names: List[str]) -> SHAPExplainer:
    """
    Factory function to create an explainer.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        SHAPExplainer instance
    """
    return SHAPExplainer(model, feature_names)
