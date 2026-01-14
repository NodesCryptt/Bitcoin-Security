import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from pathlib import Path
import joblib

# -----------------------------
# PATHS
# -----------------------------
DATA_FILE = Path(r"C:\Users\asus\cp1_projects\results\cp1_static_ellipticpp.csv")
MODEL_DIR = Path(r"C:\Users\asus\cp1_projects\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "cp1_static_xgb_v1.joblib"

print("=" * 60)
print("TRAINING CP1 STATIC XGBOOST BASELINE")
print("=" * 60)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["label"])
y = df["label"]

print(f"Total samples: {len(df)}")
print(f"Features: {X.shape[1]}")

# -----------------------------
# TIME-AWARE SPLIT
# -----------------------------
split_idx = int(0.8 * len(df))

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]

X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# -----------------------------
# MODEL (XGBOOST 3.x SAFE)
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=40
)

print("\nTraining model...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# -----------------------------
# EVALUATION
# -----------------------------
print("\nEvaluating on test set...")
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC: {auc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, MODEL_FILE)
print("\nModel saved to:")
print(MODEL_FILE)

print("\nCP1 STATIC XGBOOST TRAINING COMPLETE.")
