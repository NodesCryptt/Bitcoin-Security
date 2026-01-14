import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\asus\cp1_projects\datasets\mempool_snapshots\testnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_SECONDS = 300  # 5 minutes

def get_mempool():
    result = subprocess.run(
        ["bitcoin-cli", "-testnet", "getrawmempool", "true"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)

print("🚀 Starting TESTNET mempool collection (Ctrl+C to stop)")

try:
    while True:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        mempool = get_mempool()

        outfile = OUTPUT_DIR / f"mempool_{timestamp}.json"
        with open(outfile, "w") as f:
            json.dump(mempool, f)

        print(f"[{timestamp}] Saved {len(mempool)} transactions")
        time.sleep(INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("🛑 Mempool collection stopped.")
