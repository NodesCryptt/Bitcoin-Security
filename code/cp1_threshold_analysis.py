import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

# -----------------------------
# PATHS
# -----------------------------
DATA_FILE = Path(r"C:\Users\asus\cp1_projects\results\cp1_static_ellipticpp.csv")
MODEL_FILE = Path(r"C:\Users\asus\cp1_projects\models\cp1_static_xgb_v1.joblib")

print("=" * 60)
print("CP1 THRESHOLD ANALYSIS")
print("=" * 60)

# -----------------------------
# LOAD DATA & MODEL
# -----------------------------
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["label"])
y = df["label"]

# same time-aware split as training
split_idx = int(0.8 * len(df))
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

model = joblib.load(MODEL_FILE)

y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# THRESHOLD SWEEP
# -----------------------------
thresholds = np.linspace(0.05, 0.95, 19)

results = []

print("\nThreshold | Recall(illicit) | FPR(licit) | Illicit caught | Licit flagged")
print("-" * 75)

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    results.append((t, recall, fpr, tp, fp))

    print(
        f"{t:8.2f} | {recall:14.3f} | {fpr:10.4f} |"
        f"{tp:15d} | {fp:13d}"
    )

# -----------------------------
# SAVE RESULTS
# -----------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "threshold",
        "recall_illicit",
        "fpr_licit",
        "illicit_caught",
        "licit_flagged"
    ]
)

out_file = Path(r"C:\Users\asus\cp1_projects\results\cp1_threshold_analysis.csv")
results_df.to_csv(out_file, index=False)

print("\nSaved threshold analysis to:")
print(out_file)
