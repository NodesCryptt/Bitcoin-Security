import json
import time
from datetime import datetime
from pathlib import Path
import subprocess

OUTPUT_DIR = Path(r"C:\Users\asus\cp1_projects\mempool\mainnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_SECONDS = 300  # 5 minutes

print("🚀 Starting MAINNET mempool collection (Ctrl+C to stop)")

while True:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    out_file = OUTPUT_DIR / f"{timestamp}.json"

    try:
        mempool_info = subprocess.check_output(
            ["bitcoin-cli", "getmempoolinfo"], text=True
        )
        mempool_txs = subprocess.check_output(
            ["bitcoin-cli", "getrawmempool", "true"], text=True
        )

        snapshot = {
            "timestamp": timestamp,
            "mempool_info": json.loads(mempool_info),
            "transactions": json.loads(mempool_txs)
        }

        with open(out_file, "w") as f:
            json.dump(snapshot, f)

        print(f"[{timestamp}] Saved {len(snapshot['transactions'])} transactions")

    except Exception as e:
        print(f"[{timestamp}] ERROR: {e}")

    time.sleep(INTERVAL_SECONDS)
