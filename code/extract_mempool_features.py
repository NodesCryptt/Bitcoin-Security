import json
from pathlib import Path
import numpy as np
import pandas as pd

print("=" * 60)
print("EXTRACTING MAINNET MEMPOOL FEATURES (CP1)")
print("=" * 60)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
MEMPOOL_DIR = Path(
    r"C:\Users\asus\cp1_projects\datasets\mempool_snapshots\mainnet"
)

OUTPUT_FILE = Path(
    r"C:\Users\asus\cp1_projects\results\cp1_mempool_features.csv"
)

snapshots = sorted(MEMPOOL_DIR.glob("*.json"))

if not snapshots:
    raise RuntimeError("No mempool snapshot JSON files found!")

print(f"Found {len(snapshots)} mempool snapshots")

rows = []

# ------------------------------------------------------------------
# PROCESS SNAPSHOTS
# ------------------------------------------------------------------
for snap in snapshots:
    with open(snap, "r") as f:
        snap_data = json.load(f)

    timestamp = snap_data.get("timestamp", snap.stem)
    mempool_info = snap_data.get("mempool_info", {})
    tx_map = snap_data.get("transactions", {})

    if not isinstance(tx_map, dict) or not tx_map:
        continue

    fee_rates = []

    for txid, tx in tx_map.items():
        if not isinstance(tx, dict):
            continue

        vsize = tx.get("vsize", 0)
        fees = tx.get("fees", {})

        if vsize <= 0:
            continue

        fee = fees.get("base", None)
        if fee is None:
            continue

        fee_rates.append(fee / vsize)

    if not fee_rates:
        continue

    row = {
        "timestamp": timestamp,
        "mempool_tx_count": len(fee_rates),
        "mempool_size": mempool_info.get("size", None),
        "mempool_bytes": mempool_info.get("bytes", None),
        "fee_rate_min": float(np.min(fee_rates)),
        "fee_rate_median": float(np.median(fee_rates)),
        "fee_rate_p90": float(np.percentile(fee_rates, 90)),
        "fee_rate_max": float(np.max(fee_rates)),
    }

    rows.append(row)

# ------------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved mempool features to:")
print(OUTPUT_FILE)

print("\nSample:")
print(df.head())

print("\nExtraction complete.")
