"""
CP1 Model Regression Tests
==========================
Tests to prevent model performance regression.
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import sys

# Paths
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_FILE = Path(__file__).parent.parent / "results" / "cp1_static_ellipticpp.csv"


# =============================================================================
# BASELINE METRICS (from initial training)
# =============================================================================

# These are the baseline metrics from the initial model training
# Update these when a new baseline is established
BASELINE_METRICS = {
    "auc": 0.95,            # Minimum acceptable AUC
    "max_fpr": 0.02,        # Maximum acceptable FPR at threshold
    "min_recall": 0.45,     # Minimum acceptable recall at threshold
    "threshold": 0.65,      # Threshold for regression testing
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_model_and_data():
    """Load current model and test data."""
    model_path = MODEL_DIR / "cp1_static_xgb_v1.joblib"
    
    if not model_path.exists():
        pytest.skip("Model not found")
    if not DATA_FILE.exists():
        pytest.skip("Data not found")
    
    model = joblib.load(model_path)
    df = pd.read_csv(DATA_FILE)
    
    X = df.drop(columns=["label"])
    y = df["label"]
    
    # Time-aware split (same as training)
    split_idx = int(0.8 * len(df))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    return model, X_test, y_test


def compute_metrics(model, X_test, y_test, threshold=0.5):
    """Compute model performance metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    auc = roc_auc_score(y_test, y_prob)
    
    # Confusion matrix components
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return {
        "auc": auc,
        "fpr": fpr,
        "recall": recall,
        "precision": precision,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestModelAUC:
    """Tests for model AUC performance."""
    
    def test_auc_above_baseline(self):
        """Model AUC should not decrease more than 10% from baseline."""
        model, X_test, y_test = load_model_and_data()
        metrics = compute_metrics(model, X_test, y_test)
        
        min_auc = BASELINE_METRICS["auc"] * 0.9  # 10% tolerance
        
        assert metrics["auc"] >= min_auc, (
            f"AUC regression detected: {metrics['auc']:.4f} < {min_auc:.4f} "
            f"(baseline: {BASELINE_METRICS['auc']:.4f})"
        )
    
    def test_auc_minimum_threshold(self):
        """Model AUC should be above minimum acceptable threshold."""
        model, X_test, y_test = load_model_and_data()
        metrics = compute_metrics(model, X_test, y_test)
        
        # Absolute minimum AUC for production
        min_auc = 0.80
        
        assert metrics["auc"] >= min_auc, (
            f"AUC below minimum: {metrics['auc']:.4f} < {min_auc:.4f}"
        )


class TestFalsePositiveRate:
    """Tests for false positive rate at operational thresholds."""
    
    def test_fpr_at_reject_threshold(self):
        """FPR should be within acceptable limits at REJECT threshold."""
        model, X_test, y_test = load_model_and_data()
        metrics = compute_metrics(
            model, X_test, y_test, 
            threshold=BASELINE_METRICS["threshold"]
        )
        
        max_fpr = BASELINE_METRICS["max_fpr"]
        
        assert metrics["fpr"] <= max_fpr, (
            f"FPR too high at threshold {BASELINE_METRICS['threshold']}: "
            f"{metrics['fpr']:.4f} > {max_fpr:.4f}"
        )
    
    def test_fpr_increase_regression(self):
        """FPR should not increase more than 10% from baseline."""
        model, X_test, y_test = load_model_and_data()
        metrics = compute_metrics(
            model, X_test, y_test,
            threshold=BASELINE_METRICS["threshold"]
        )
        
        max_fpr = BASELINE_METRICS["max_fpr"] * 1.1  # 10% tolerance
        
        assert metrics["fpr"] <= max_fpr, (
            f"FPR regression detected: {metrics['fpr']:.4f} > {max_fpr:.4f}"
        )


class TestRecall:
    """Tests for recall (detection rate) at operational thresholds."""
    
    def test_recall_at_reject_threshold(self):
        """Recall should be above minimum at REJECT threshold."""
        model, X_test, y_test = load_model_and_data()
        metrics = compute_metrics(
            model, X_test, y_test,
            threshold=BASELINE_METRICS["threshold"]
        )
        
        min_recall = BASELINE_METRICS["min_recall"]
        
        assert metrics["recall"] >= min_recall, (
            f"Recall too low at threshold {BASELINE_METRICS['threshold']}: "
            f"{metrics['recall']:.4f} < {min_recall:.4f}"
        )
    
    def test_recall_decrease_regression(self):
        """Recall should not decrease more than 10% from baseline."""
        model, X_test, y_test = load_model_and_data()
        metrics = compute_metrics(
            model, X_test, y_test,
            threshold=BASELINE_METRICS["threshold"]
        )
        
        min_recall = BASELINE_METRICS["min_recall"] * 0.9  # 10% tolerance
        
        assert metrics["recall"] >= min_recall, (
            f"Recall regression detected: {metrics['recall']:.4f} < {min_recall:.4f}"
        )


class TestPredictionDistribution:
    """Tests for prediction score distribution."""
    
    def test_score_range(self):
        """All scores should be in [0, 1] range."""
        model, X_test, y_test = load_model_and_data()
        y_prob = model.predict_proba(X_test)[:, 1]
        
        assert y_prob.min() >= 0.0, f"Score below 0: {y_prob.min()}"
        assert y_prob.max() <= 1.0, f"Score above 1: {y_prob.max()}"
    
    def test_score_variance(self):
        """Scores should have reasonable variance (not degenerate)."""
        model, X_test, y_test = load_model_and_data()
        y_prob = model.predict_proba(X_test)[:, 1]
        
        variance = np.var(y_prob)
        
        # Score variance should not be too low (degenerate model)
        assert variance > 0.01, f"Score variance too low: {variance:.4f}"
    
    def test_no_nan_predictions(self):
        """Predictions should not contain NaN values."""
        model, X_test, y_test = load_model_and_data()
        y_prob = model.predict_proba(X_test)[:, 1]
        
        assert not np.isnan(y_prob).any(), "NaN values in predictions"
    
    def test_no_inf_predictions(self):
        """Predictions should not contain infinite values."""
        model, X_test, y_test = load_model_and_data()
        y_prob = model.predict_proba(X_test)[:, 1]
        
        assert not np.isinf(y_prob).any(), "Infinite values in predictions"


class TestThresholdSweep:
    """Tests for threshold sweep analysis."""
    
    def test_optimal_threshold_exists(self):
        """There should be a threshold with acceptable FPR and recall."""
        model, X_test, y_test = load_model_and_data()
        y_prob = model.predict_proba(X_test)[:, 1]
        
        thresholds = np.linspace(0.1, 0.9, 17)
        found_acceptable = False
        
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            
            tn = ((y_pred == 0) & (y_test == 0)).sum()
            fp = ((y_pred == 1) & (y_test == 0)).sum()
            fn = ((y_pred == 0) & (y_test == 1)).sum()
            tp = ((y_pred == 1) & (y_test == 1)).sum()
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Looking for threshold with FPR < 2% and recall > 40%
            if fpr < 0.02 and recall > 0.40:
                found_acceptable = True
                break
        
        assert found_acceptable, (
            "No threshold found with acceptable FPR (<2%) and recall (>40%)"
        )


class TestModelConsistency:
    """Tests for model prediction consistency."""
    
    def test_deterministic_predictions(self):
        """Same input should always produce same output."""
        model, X_test, y_test = load_model_and_data()
        
        # Take first 10 samples
        X_sample = X_test.head(10)
        
        # Run predictions multiple times
        results = [model.predict_proba(X_sample)[:, 1] for _ in range(5)]
        
        for i, result in enumerate(results[1:], 1):
            np.testing.assert_array_almost_equal(
                results[0], result,
                decimal=10,
                err_msg=f"Prediction mismatch between run 0 and {i}"
            )
    
    def test_feature_importance_exists(self):
        """Model should have feature importances."""
        model, X_test, y_test = load_model_and_data()
        
        # XGBoost models have feature_importances_
        assert hasattr(model, "feature_importances_")
        
        importances = model.feature_importances_
        assert len(importances) > 0
        assert importances.sum() > 0


class TestCompareWithBaseline:
    """Tests comparing current model with baseline metrics."""
    
    def test_overall_performance_summary(self):
        """Print overall performance for review."""
        model, X_test, y_test = load_model_and_data()
        
        thresholds = [0.12, 0.15, 0.35, 0.60, 0.65]
        
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for t in thresholds:
            metrics = compute_metrics(model, X_test, y_test, threshold=t)
            print(f"Threshold {t:.2f}: "
                  f"FPR={metrics['fpr']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, "
                  f"Precision={metrics['precision']:.4f}")
        
        # Overall AUC
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nOverall AUC: {auc:.4f}")
        print("=" * 60)
        
        # This test always passes - just for reporting
        assert True


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
