import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\asus\cp1_projects")
DATASET_DIR = BASE_DIR / "datasets" / "elliptic" / "Elliptic++ Dataset"
OUTPUT_DIR = BASE_DIR / "results"

TX_FEATURES_FILE = DATASET_DIR / "txs_features.csv"
TX_CLASSES_FILE = DATASET_DIR / "txs_classes.csv"

OUTPUT_FILE = OUTPUT_DIR / "elliptic_tx_master.csv"

# ============================================================
# HELPER FUNCTION
# ============================================================

def detect_tx_id_column(df, file_name):
    """
    Detect transaction ID column in Elliptic datasets.
    """
    if "tx_id" in df.columns:
        return "tx_id"
    if "txId" in df.columns:
        return "txId"
    if df.columns[0].lower() in ["txid", "tx_id"]:
        return df.columns[0]

    raise ValueError(
        f"Could not detect transaction ID column in {file_name}. "
        f"Columns found: {list(df.columns[:5])} ..."
    )

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ELLIPTIC++ MASTER DATASET BUILDER")
print("=" * 60)

print(f"Dataset directory: {DATASET_DIR}")

print("Loading transaction features...")
tx_features = pd.read_csv(TX_FEATURES_FILE)

print("Loading transaction classes...")
tx_classes = pd.read_csv(TX_CLASSES_FILE)

# ============================================================
# DETECT & NORMALIZE TX ID COLUMN
# ============================================================

tx_feat_id_col = detect_tx_id_column(tx_features, "txs_features.csv")
tx_cls_id_col = detect_tx_id_column(tx_classes, "txs_classes.csv")

print(f"Detected tx_id column in features: {tx_feat_id_col}")
print(f"Detected tx_id column in classes:  {tx_cls_id_col}")

tx_features = tx_features.rename(columns={tx_feat_id_col: "tx_id"})
tx_classes = tx_classes.rename(columns={tx_cls_id_col: "tx_id"})

# ============================================================
# BASIC SANITY CHECKS
# ============================================================

if "class" not in tx_classes.columns:
    raise ValueError("txs_classes.csv must contain a 'class' column")

print(f"Transactions with features: {len(tx_features)}")
print(f"Transactions with labels:   {len(tx_classes)}")

# ============================================================
# MERGE FEATURES + LABELS
# ============================================================

print("Merging on tx_id (INNER JOIN)...")
tx_master = tx_features.merge(tx_classes, on="tx_id", how="inner")

print(f"Transactions after merge: {len(tx_master)}")

# ============================================================
# HANDLE LABELS
# ============================================================

print("Inspecting class labels...")
print("Unique class values:", tx_master["class"].unique())

# Normalize class column to string (safe for mixed types)
tx_master["class"] = tx_master["class"].astype(str).str.lower().str.strip()

# Drop unknown / unlabeled if present
unknown_mask = tx_master["class"].isin(["0", "unknown", "unlabelled", "unlabeled"])
before_drop = len(tx_master)
tx_master = tx_master[~unknown_mask]
after_drop = len(tx_master)

print(f"Dropped {before_drop - after_drop} unknown-label transactions")

# Map labels semantically
label_mapping = {
    "licit": 0,
    "2": 0,
    "illicit": 1,
    "1": 1
}

tx_master["label"] = tx_master["class"].map(label_mapping)

# Final validation
if tx_master["label"].isna().any():
    bad_vals = tx_master.loc[tx_master["label"].isna(), "class"].unique()
    raise ValueError(f"Unmapped class labels found: {bad_vals}")

tx_master.drop(columns=["class"], inplace=True)

print("Label mapping successful: licit=0, illicit=1")


# ============================================================
# FINAL SANITY CHECKS
# ============================================================

if not tx_master["tx_id"].is_unique:
    raise ValueError("Duplicate tx_id found after merge")

if not tx_master["label"].isin([0, 1]).all():
    raise ValueError("Invalid labels found after mapping")

print("Final label distribution:")
print(tx_master["label"].value_counts(normalize=True))

print(f"Total usable transactions: {len(tx_master)}")
print(f"Total feature columns (excluding tx_id & label): {tx_master.shape[1] - 2}")

# ============================================================
# SAVE MASTER DATASET
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tx_master.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("SUCCESS ✅")
print("Elliptic transaction master dataset created:")
print(OUTPUT_FILE)
print("=" * 60)
