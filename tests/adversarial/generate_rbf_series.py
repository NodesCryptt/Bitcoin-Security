#!/usr/bin/env python
"""
CP1 Adversarial Test Harness - RBF Series Generator
===================================================
Generates RBF (Replace-By-Fee) transaction series for testing CP1 detection.

RBF transactions are legitimate but can be used maliciously to:
1. Cancel payments after receiving goods
2. Double-spend by increasing fees to prioritize a different version

Usage:
    python generate_rbf_series.py --count 50 --output rbf_series.json
"""

import argparse
import json
import hashlib
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "adversarial"


def create_rbf_series(series_length: int = 3) -> List[Dict]:
    """
    Create an RBF series - multiple versions of the same transaction
    with increasing fees.
    """
    # Shared UTXO for all versions
    shared_utxo_txid = hashlib.sha256(
        f"utxo_{datetime.utcnow().isoformat()}_{random.randint(0, 999999)}".encode()
    ).hexdigest()
    shared_utxo_value = round(random.uniform(0.1, 1.0), 8)
    
    series_id = hashlib.sha256(f"series_{time.time()}".encode()).hexdigest()[:16]
    series = []
    
    base_fee = 0.00001  # 1 sat/vB baseline
    
    for i in range(series_length):
        fee_multiplier = (i + 1) ** 2  # Quadratic fee increase
        current_fee = base_fee * fee_multiplier
        
        # RBF signaling: sequence < 0xfffffffe
        sequence = 0xfffffffd - i
        
        tx_id = hashlib.sha256(
            f"rbf_{series_id}_{i}_{random.randint(0, 999999)}".encode()
        ).hexdigest()
        
        tx = {
            "txid": tx_id,
            "type": f"rbf_version_{i + 1}",
            "size": 226 + random.randint(-10, 20),
            "vsize": 166 + random.randint(-5, 15),
            "weight": 664 + random.randint(-20, 60),
            "version": 2,
            "locktime": 0,
            "fee": round(current_fee, 8),
            "fee_rate_sat_vb": round(current_fee * 100_000_000 / 166, 2),
            "vin": [
                {
                    "txid": shared_utxo_txid,
                    "vout": 0,
                    "value": shared_utxo_value,
                    "sequence": sequence,  # RBF signaling
                    "scriptSig": {"asm": f"sig_v{i+1}", "hex": "483045..."}
                }
            ],
            "vout": [
                {
                    "value": round(shared_utxo_value - current_fee - 0.001, 8),
                    "n": 0,
                    "scriptPubKey": {
                        "type": "pubkeyhash",
                        "address": f"1Recipient{tx_id[:18]}"
                    }
                },
                {
                    "value": 0.001,  # Fixed change
                    "n": 1,
                    "scriptPubKey": {
                        "type": "pubkeyhash",
                        "address": f"1Change{tx_id[:21]}"
                    }
                }
            ],
            # RBF itself is not illicit, but rapid re-spending can be suspicious
            "label": 0 if i == 0 else (1 if i >= 2 else 0),  # Flag if 3+ versions
            "attack_type": "rbf_series" if i >= 2 else "rbf_normal",
            "series_id": series_id,
            "version_number": i + 1,
            "is_rbf_enabled": True,
            "rbf_sequence": sequence,
        }
        
        series.append(tx)
    
    return series


def generate_rbf_dataset(series_count: int = 50) -> List[Dict]:
    """Generate a dataset of RBF transaction series."""
    dataset = []
    
    for i in range(series_count):
        # Random series length between 2-5 versions
        series_length = random.randint(2, 5)
        series = create_rbf_series(series_length)
        dataset.extend(series)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{series_count} RBF series...")
    
    return dataset


def save_dataset(dataset: List[Dict], output_file: Path):
    """Save dataset to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved {len(dataset)} transactions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate RBF series test data")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of RBF series to generate")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "rbf_series.json",
                        help="Output file path")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CP1 ADVERSARIAL TEST - RBF SERIES GENERATOR")
    print("=" * 60)
    print(f"Generating {args.count} RBF series...")
    
    dataset = generate_rbf_dataset(args.count)
    
    save_dataset(dataset, args.output)
    
    # Summary statistics
    normal_count = sum(1 for tx in dataset if tx.get("label") == 0)
    suspicious_count = sum(1 for tx in dataset if tx.get("label") == 1)
    
    print("\nSummary:")
    print(f"  Total transactions: {len(dataset)}")
    print(f"  Normal RBF: {normal_count}")
    print(f"  Suspicious RBF (3+ versions): {suspicious_count}")
    print(f"  Output file: {args.output}")
    print("\nDone!")


if __name__ == "__main__":
    main()
