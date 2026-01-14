#!/usr/bin/env python
"""
CP1 Adversarial Test Harness - Run All Generators
==================================================
Runs all adversarial test generators and creates a combined dataset.

Usage:
    python run_all.py --network=regtest
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_double_spend import generate_double_spend_dataset
from generate_rbf_series import generate_rbf_dataset
from generate_low_fee_spam import generate_low_fee_spam_dataset

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "adversarial"


def generate_all_adversarial_data(config: dict) -> dict:
    """Generate all types of adversarial test data."""
    
    print("=" * 60)
    print("CP1 ADVERSARIAL TEST HARNESS")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print(f"Network: {config.get('network', 'regtest')}")
    print()
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "network": config.get("network", "regtest"),
        "datasets": {}
    }
    
    # Generate double-spend pairs
    print("[1/3] Generating double-spend pairs...")
    double_spend_data = generate_double_spend_dataset(config.get("double_spend_count", 50))
    results["datasets"]["double_spend"] = {
        "count": len(double_spend_data),
        "pairs": len(double_spend_data) // 2
    }
    
    # Generate RBF series
    print("\n[2/3] Generating RBF series...")
    rbf_data = generate_rbf_dataset(config.get("rbf_count", 30))
    results["datasets"]["rbf_series"] = {
        "count": len(rbf_data),
        "series": config.get("rbf_count", 30)
    }
    
    # Generate low-fee spam
    print("\n[3/3] Generating low-fee spam...")
    spam_data = generate_low_fee_spam_dataset(config.get("spam_count", 10))
    results["datasets"]["low_fee_spam"] = {
        "count": len(spam_data)
    }
    
    # Combine all data
    all_data = {
        "metadata": results,
        "transactions": double_spend_data + rbf_data + spam_data
    }
    
    # Statistics
    total_tx = len(all_data["transactions"])
    illicit_count = sum(1 for tx in all_data["transactions"] if tx.get("label") == 1)
    licit_count = total_tx - illicit_count
    
    results["summary"] = {
        "total_transactions": total_tx,
        "illicit": illicit_count,
        "licit": licit_count,
        "illicit_ratio": round(illicit_count / total_tx, 4)
    }
    
    return all_data


def save_combined_dataset(data: dict, output_file: Path):
    """Save combined dataset."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved combined dataset to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run all adversarial test generators")
    parser.add_argument("--network", choices=["regtest", "testnet", "mainnet"],
                        default="regtest", help="Bitcoin network to target")
    parser.add_argument("--double-spend-count", type=int, default=50,
                        help="Number of double-spend pairs")
    parser.add_argument("--rbf-count", type=int, default=30,
                        help="Number of RBF series")
    parser.add_argument("--spam-count", type=int, default=10,
                        help="Number of spam bursts")
    parser.add_argument("--output", type=Path, 
                        default=OUTPUT_DIR / "adversarial_combined.json",
                        help="Output file path")
    
    args = parser.parse_args()
    
    config = {
        "network": args.network,
        "double_spend_count": args.double_spend_count,
        "rbf_count": args.rbf_count,
        "spam_count": args.spam_count
    }
    
    data = generate_all_adversarial_data(config)
    
    save_combined_dataset(data, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total transactions: {data['metadata']['summary']['total_transactions']}")
    print(f"Illicit: {data['metadata']['summary']['illicit']} "
          f"({data['metadata']['summary']['illicit_ratio']*100:.1f}%)")
    print(f"Licit: {data['metadata']['summary']['licit']}")
    print(f"\nOutput: {args.output}")
    print("\nTo include in training, run:")
    print(f"  python merge_adversarial_data.py --input {args.output}")


if __name__ == "__main__":
    main()
