#!/usr/bin/env python
"""
CP1 Adversarial Test Harness - Low Fee Spam Generator
=====================================================
Generates low-fee spam bursts for testing CP1 detection.

Low-fee spam attacks flood the mempool with cheap transactions to:
1. Cause mempool bloat and node memory issues
2. Delay legitimate transactions
3. Exploit free relay policies

Usage:
    python generate_low_fee_spam.py --count 200 --output low_fee_spam.json
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


def create_low_fee_spam_burst(burst_size: int = 20) -> List[Dict]:
    """
    Create a burst of low-fee spam transactions.
    
    Characteristics:
    - Very low fee rate (< 1 sat/vB)
    - Often small transactions
    - Sent in rapid succession
    - May create long chains
    """
    burst_id = hashlib.sha256(f"spam_{time.time()}".encode()).hexdigest()[:16]
    burst = []
    
    timestamp_base = time.time()
    
    for i in range(burst_size):
        tx_id = hashlib.sha256(f"spam_{burst_id}_{i}".encode()).hexdigest()
        
        # Very low fee rate: 0.1 - 0.9 sat/vB
        fee_rate_sat_vb = round(random.uniform(0.1, 0.9), 2)
        vsize = random.randint(100, 300)
        fee = fee_rate_sat_vb * vsize / 100_000_000
        
        tx = {
            "txid": tx_id,
            "type": "low_fee_spam",
            "size": vsize + random.randint(0, 50),
            "vsize": vsize,
            "weight": vsize * 4,
            "version": 2,
            "locktime": 0,
            "fee": round(fee, 8),
            "fee_rate_sat_vb": fee_rate_sat_vb,
            "vin": [
                {
                    "txid": hashlib.sha256(f"input_{tx_id}".encode()).hexdigest(),
                    "vout": 0,
                    "value": round(random.uniform(0.0001, 0.001), 8),
                    "sequence": 4294967295
                }
            ],
            "vout": [
                {
                    "value": round(random.uniform(0.00005, 0.0009), 8),
                    "n": 0,
                    "scriptPubKey": {
                        "type": "pubkeyhash",
                        "address": f"1Spam{tx_id[:23]}"
                    }
                }
            ],
            "label": 1,  # Low-fee spam is considered illicit
            "attack_type": "low_fee_spam",
            "burst_id": burst_id,
            "burst_position": i,
            "arrival_time": timestamp_base + (i * 0.01),  # 10ms apart
        }
        
        burst.append(tx)
    
    return burst


def generate_low_fee_spam_dataset(burst_count: int = 10) -> List[Dict]:
    """Generate a dataset of low-fee spam bursts."""
    dataset = []
    
    for i in range(burst_count):
        burst_size = random.randint(10, 30)
        burst = create_low_fee_spam_burst(burst_size)
        dataset.extend(burst)
        
        if (i + 1) % 5 == 0:
            print(f"Generated {i + 1}/{burst_count} spam bursts...")
    
    return dataset


def save_dataset(dataset: List[Dict], output_file: Path):
    """Save dataset to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved {len(dataset)} transactions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate low-fee spam test data")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of spam bursts to generate")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "low_fee_spam.json",
                        help="Output file path")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CP1 ADVERSARIAL TEST - LOW FEE SPAM GENERATOR")
    print("=" * 60)
    print(f"Generating {args.count} spam bursts...")
    
    dataset = generate_low_fee_spam_dataset(args.count)
    
    save_dataset(dataset, args.output)
    
    # Summary statistics
    avg_fee_rate = sum(tx.get("fee_rate_sat_vb", 0) for tx in dataset) / len(dataset)
    
    print("\nSummary:")
    print(f"  Total transactions: {len(dataset)}")
    print(f"  Spam bursts: {args.count}")
    print(f"  Average fee rate: {avg_fee_rate:.2f} sat/vB")
    print(f"  Output file: {args.output}")
    print("\nDone!")


if __name__ == "__main__":
    main()
