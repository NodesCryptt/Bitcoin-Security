import pandas as pd


BASE_PATH = r"C:\Users\asus\cp1_projects\datasets\elliptic\Elliptic++ Dataset"

FEATURES_FILE = BASE_PATH + r"\txs_features.csv"
CLASSES_FILE  = BASE_PATH + r"\txs_classes.csv"

print("Loading transaction features...")
features = pd.read_csv(FEATURES_FILE)

print("Loading transaction classes...")
classes = pd.read_csv(CLASSES_FILE)


print("\n================ BASIC INFO ================")
print(f"Number of transactions (features): {features.shape[0]}")
print(f"Number of feature columns: {features.shape[1]}")

print(f"Number of transactions (classes): {classes.shape[0]}")
print(f"Class columns: {list(classes.columns)}")

# -----------------------------
# MERGE (TEMPORARY, FOR EDA ONLY)
# -----------------------------
df = features.merge(classes, on="txId", how="inner")

print("\n================ AFTER MERGE ================")
print(f"Total rows after merge: {df.shape[0]}")
print(f"Total columns after merge: {df.shape[1]}")

# -----------------------------
# LABEL DISTRIBUTION
# -----------------------------
print("\n================ LABEL DISTRIBUTION ================")
print(df["class"].value_counts())
print("\nLabel ratios:")
print(df["class"].value_counts(normalize=True))

# -----------------------------
# FEATURE NAME INSPECTION
# -----------------------------
print("\n================ FEATURE NAMES (SAMPLE) ================")
feature_cols = [c for c in df.columns if c not in ["txId", "class"]]

print(f"Total usable feature columns (raw): {len(feature_cols)}")
print("First 20 features:")
print(feature_cols[:20])

print("\nLast 20 features:")
print(feature_cols[-20:])

# -----------------------------
# MISSING & CONSTANT CHECK
# -----------------------------
print("\n================ DATA QUALITY CHECK ================")
missing_total = df.isnull().sum().sum()
print(f"Total missing values in dataset: {missing_total}")

constant_features = [c for c in feature_cols if df[c].nunique() <= 1]
print(f"Number of constant features: {len(constant_features)}")

if constant_features:
    print("Constant feature names:")
    print(constant_features)

print("\nEDA COMPLETE — NO DATA MODIFIED.")
