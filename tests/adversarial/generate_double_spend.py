#!/usr/bin/env python
"""
CP1 Adversarial Test Harness - Double Spend Generator
=====================================================
Generates double-spend transaction pairs for testing CP1 detection.

This script creates pairs of conflicting transactions that spend the same UTXO.
Use with Bitcoin Core in regtest mode.

Usage:
    python generate_double_spend.py --regtest --rpc-url=http://user:pass@localhost:18443
"""

import argparse
import json
import hashlib
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# Default output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "adversarial"


def create_mock_double_spend_pair() -> Tuple[Dict, Dict]:
    """
    Create a mock double-spend pair for testing without actual Bitcoin Core.
    
    In production, this would use Bitcoin Core RPC to create real transactions.
    For testing, we generate synthetic data.
    """
    # Shared UTXO (same input for both transactions)
    shared_utxo_txid = hashlib.sha256(
        f"utxo_{datetime.utcnow().isoformat()}_{random.randint(0, 999999)}".encode()
    ).hexdigest()
    shared_utxo_vout = 0
    shared_utxo_value = round(random.uniform(0.01, 1.0), 8)
    
    # Transaction 1: Original transaction
    tx1_id = hashlib.sha256(f"tx1_{time.time()}_{random.randint(0, 999999)}".encode()).hexdigest()
    tx1 = {
        "txid": tx1_id,
        "type": "double_spend_original",
        "size": 226 + random.randint(-20, 50),
        "vsize": 166 + random.randint(-10, 30),
        "weight": 664 + random.randint(-40, 120),
        "version": 2,
        "locktime": 0,
        "vin": [
            {
                "txid": shared_utxo_txid,
                "vout": shared_utxo_vout,
                "value": shared_utxo_value,
                "scriptSig": {"asm": "sig1", "hex": "483045..."},
                "sequence": 4294967295
            }
        ],
        "vout": [
            {
                "value": round(shared_utxo_value * 0.95, 8),  # Output to recipient
                "n": 0,
                "scriptPubKey": {
                    "type": "pubkeyhash",
                    "address": f"1Original{tx1_id[:20]}"
                }
            },
            {
                "value": round(shared_utxo_value * 0.04, 8),  # Change
                "n": 1,
                "scriptPubKey": {
                    "type": "pubkeyhash",
                    "address": f"1Change{tx1_id[:22]}"
                }
            }
        ],
        "label": 1,  # Illicit (double-spend attempt)
        "attack_type": "double_spend",
        "pair_id": tx1_id[:16]
    }
    
    # Transaction 2: Double-spend (same input, different output)
    tx2_id = hashlib.sha256(f"tx2_{time.time()}_{random.randint(0, 999999)}".encode()).hexdigest()
    tx2 = {
        "txid": tx2_id,
        "type": "double_spend_conflict",
        "size": 226 + random.randint(-20, 50),
        "vsize": 166 + random.randint(-10, 30),
        "weight": 664 + random.randint(-40, 120),
        "version": 2,
        "locktime": 0,
        "vin": [
            {
                "txid": shared_utxo_txid,  # SAME UTXO as tx1
                "vout": shared_utxo_vout,
                "value": shared_utxo_value,
                "scriptSig": {"asm": "sig2", "hex": "483045..."},
                "sequence": 4294967295
            }
        ],
        "vout": [
            {
                "value": round(shared_utxo_value * 0.90, 8),  # Different recipient
                "n": 0,
                "scriptPubKey": {
                    "type": "pubkeyhash",
                    "address": f"1Attacker{tx2_id[:19]}"
                }
            },
            {
                "value": round(shared_utxo_value * 0.09, 8),  # Different change
                "n": 1,
                "scriptPubKey": {
                    "type": "pubkeyhash",
                    "address": f"1Fee{tx2_id[:24]}"
                }
            }
        ],
        "label": 1,  # Illicit (double-spend attempt)
        "attack_type": "double_spend",
        "pair_id": tx1_id[:16]
    }
    
    return tx1, tx2


def generate_double_spend_dataset(count: int = 100) -> List[Dict]:
    """Generate a dataset of double-spend transaction pairs."""
    dataset = []
    
    for i in range(count):
        tx1, tx2 = create_mock_double_spend_pair()
        dataset.append(tx1)
        dataset.append(tx2)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{count} double-spend pairs...")
    
    return dataset


def extract_features_for_training(tx: Dict) -> Dict:
    """Extract features from a transaction for training."""
    features = {
        "size": float(tx.get("size", 0)),
        "vsize": float(tx.get("vsize", 0)),
        "weight": float(tx.get("weight", 0)),
        "num_input_addresses": len(tx.get("vin", [])),
        "num_output_addresses": len(tx.get("vout", [])),
        "total_BTC": sum(v.get("value", 0) for v in tx.get("vout", [])),
        "fees": sum(v.get("value", 0) for v in tx.get("vin", [])) - 
                sum(v.get("value", 0) for v in tx.get("vout", [])),
        
        # Attack indicators
        "is_double_spend": 1 if tx.get("attack_type") == "double_spend" else 0,
        
        # Label
        "label": tx.get("label", 0),
        "attack_type": tx.get("attack_type", "none"),
        "txid": tx.get("txid", ""),
    }
    return features


def save_dataset(dataset: List[Dict], output_file: Path):
    """Save dataset to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved {len(dataset)} transactions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate double-spend test data")
    parser.add_argument("--count", type=int, default=100, 
                        help="Number of double-spend pairs to generate")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "double_spend_pairs.json",
                        help="Output file path")
    parser.add_argument("--regtest", action="store_true",
                        help="Generate for regtest network")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CP1 ADVERSARIAL TEST - DOUBLE SPEND GENERATOR")
    print("=" * 60)
    print(f"Generating {args.count} double-spend pairs...")
    
    dataset = generate_double_spend_dataset(args.count)
    
    save_dataset(dataset, args.output)
    
    print("\nSummary:")
    print(f"  Total transactions: {len(dataset)}")
    print(f"  Double-spend pairs: {len(dataset) // 2}")
    print(f"  Output file: {args.output}")
    print("\nDone!")


if __name__ == "__main__":
    main()
