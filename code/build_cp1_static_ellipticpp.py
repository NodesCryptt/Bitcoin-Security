import pandas as pd
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
BASE_DATASET_PATH = Path(
    r"C:\Users\asus\cp1_projects\datasets\elliptic\Elliptic++ Dataset"
)

FEATURES_FILE = BASE_DATASET_PATH / "txs_features.csv"
CLASSES_FILE  = BASE_DATASET_PATH / "txs_classes.csv"

OUTPUT_DIR = Path(r"C:\Users\asus\cp1_projects\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cp1_static_ellipticpp.csv"

print("=" * 60)
print("BUILDING CP1 STATIC DATASET (ELLIPTIC++)")
print("=" * 60)

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading transaction features...")
features = pd.read_csv(FEATURES_FILE)

print("Loading transaction classes...")
classes = pd.read_csv(CLASSES_FILE)

# -----------------------------
# MERGE
# -----------------------------
print("Merging features and classes...")
df = features.merge(classes, on="txId", how="inner")

print(f"Rows after merge: {len(df)}")

# -----------------------------
# DROP BACKGROUND CLASS
# -----------------------------
print("Dropping background transactions (class = 3)...")
before = len(df)
df = df[df["class"].isin([1, 2])]
after = len(df)

print(f"Dropped {before - after} background transactions")
print(f"Remaining rows: {after}")

# -----------------------------
# MAP LABELS
# -----------------------------
print("Mapping labels: illicit=1, licit=0")
df["label"] = df["class"].map({1: 1, 2: 0})

# -----------------------------
# DROP LEAKAGE / UNUSED COLUMNS
# -----------------------------
DROP_COLUMNS = [
    "txId",
    "class",
    "Time step"
]

print("Dropping non-CP1 columns:", DROP_COLUMNS)
df = df.drop(columns=DROP_COLUMNS)

# -----------------------------
# REORDER COLUMNS (FEATURES → LABEL)
# -----------------------------
feature_cols = [c for c in df.columns if c != "label"]
df = df[feature_cols + ["label"]]

# -----------------------------
# FINAL CHECKS
# -----------------------------
print("\nFINAL DATASET SUMMARY")
print("-" * 40)
print(f"Final rows: {df.shape[0]}")
print(f"Final features: {len(feature_cols)}")
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nLabel ratios:")
print(df["label"].value_counts(normalize=True))

# -----------------------------
# SAVE
# -----------------------------
df.to_csv(OUTPUT_FILE, index=False)
print("\nCP1 static dataset saved to:")
print(OUTPUT_FILE)

print("\nCP1 STATIC DATASET BUILD COMPLETE.")
