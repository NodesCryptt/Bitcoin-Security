"""
CP1 Test Configuration and Fixtures
====================================
Provides sample transaction data and shared fixtures for testing.
"""

import pytest
import sys
from pathlib import Path

# Add code directory to path for imports
CODE_DIR = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))


# =============================================================================
# SAMPLE TRANSACTION DATA
# =============================================================================

# Standard P2PKH transaction (2 inputs, 2 outputs)
SAMPLE_P2PKH_TX = {
    "txid": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
    "size": 226,
    "vsize": 226,
    "weight": 904,
    "version": 2,
    "locktime": 0,
    "vin": [
        {
            "txid": "prev1111111111111111111111111111111111111111111111111111111111111111",
            "vout": 0,
            "scriptSig": {"asm": "3045...", "hex": "483045..."},
            "sequence": 4294967295
        },
        {
            "txid": "prev2222222222222222222222222222222222222222222222222222222222222222",
            "vout": 1,
            "scriptSig": {"asm": "3045...", "hex": "483045..."},
            "sequence": 4294967295
        }
    ],
    "vout": [
        {
            "value": 0.5,
            "n": 0,
            "scriptPubKey": {
                "asm": "OP_DUP OP_HASH160 ... OP_EQUALVERIFY OP_CHECKSIG",
                "hex": "76a914...",
                "type": "pubkeyhash",
                "address": "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
            }
        },
        {
            "value": 0.0001,
            "n": 1,
            "scriptPubKey": {
                "asm": "OP_DUP OP_HASH160 ... OP_EQUALVERIFY OP_CHECKSIG",
                "hex": "76a914...",
                "type": "pubkeyhash",
                "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
            }
        }
    ]
}


# Coinbase transaction (mining reward)
SAMPLE_COINBASE_TX = {
    "txid": "coinbase1111111111111111111111111111111111111111111111111111111111",
    "size": 164,
    "vsize": 137,
    "weight": 548,
    "version": 1,
    "locktime": 0,
    "vin": [
        {
            "coinbase": "03a4090104...",
            "sequence": 4294967295
        }
    ],
    "vout": [
        {
            "value": 6.25,
            "n": 0,
            "scriptPubKey": {
                "asm": "OP_DUP OP_HASH160 ... OP_EQUALVERIFY OP_CHECKSIG",
                "hex": "76a914...",
                "type": "pubkeyhash",
                "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
            }
        }
    ]
}


# SegWit transaction (native P2WPKH)
SAMPLE_SEGWIT_TX = {
    "txid": "segwit111111111111111111111111111111111111111111111111111111111111",
    "size": 222,
    "vsize": 141,
    "weight": 561,
    "version": 2,
    "locktime": 0,
    "vin": [
        {
            "txid": "prev3333333333333333333333333333333333333333333333333333333333333333",
            "vout": 0,
            "scriptSig": {"asm": "", "hex": ""},
            "txinwitness": [
                "3045...",
                "02..."
            ],
            "sequence": 4294967294
        }
    ],
    "vout": [
        {
            "value": 0.1,
            "n": 0,
            "scriptPubKey": {
                "asm": "0 ...",
                "hex": "0014...",
                "type": "witness_v0_keyhash",
                "address": "bc1q..."
            }
        }
    ]
}


# OP_RETURN transaction (data embedding)
SAMPLE_OP_RETURN_TX = {
    "txid": "opreturn1111111111111111111111111111111111111111111111111111111111",
    "size": 245,
    "vsize": 245,
    "weight": 980,
    "version": 2,
    "locktime": 0,
    "vin": [
        {
            "txid": "prev4444444444444444444444444444444444444444444444444444444444444444",
            "vout": 0,
            "scriptSig": {"asm": "3045...", "hex": "483045..."},
            "sequence": 4294967295
        }
    ],
    "vout": [
        {
            "value": 0.0,
            "n": 0,
            "scriptPubKey": {
                "asm": "OP_RETURN 48656c6c6f",
                "hex": "6a0548656c6c6f",
                "type": "nulldata"
            }
        },
        {
            "value": 0.001,
            "n": 1,
            "scriptPubKey": {
                "asm": "OP_DUP OP_HASH160 ... OP_EQUALVERIFY OP_CHECKSIG",
                "hex": "76a914...",
                "type": "pubkeyhash",
                "address": "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
            }
        }
    ]
}


# Large transaction (many inputs/outputs)
SAMPLE_LARGE_TX = {
    "txid": "largetx11111111111111111111111111111111111111111111111111111111111",
    "size": 15000,
    "vsize": 10000,
    "weight": 40000,
    "version": 2,
    "locktime": 0,
    "vin": [{"txid": f"input{i:060d}", "vout": 0, "scriptSig": {"asm": "", "hex": ""}, "sequence": 4294967295} for i in range(50)],
    "vout": [{"value": 0.01, "n": i, "scriptPubKey": {"type": "pubkeyhash", "address": f"addr{i}"}} for i in range(100)]
}


# Transaction with missing vin data
SAMPLE_MISSING_VIN_TX = {
    "txid": "missingvin11111111111111111111111111111111111111111111111111111111",
    "size": 100,
    "vsize": 100,
    "weight": 400,
    "version": 2,
    "locktime": 0,
    "vin": [],
    "vout": [
        {
            "value": 0.1,
            "n": 0,
            "scriptPubKey": {
                "type": "pubkeyhash",
                "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
            }
        }
    ]
}


# Transaction with fee info (mempool context)
SAMPLE_MEMPOOL_TX = {
    "txid": "mempool11111111111111111111111111111111111111111111111111111111111",
    "size": 250,
    "vsize": 166,
    "weight": 664,
    "version": 2,
    "locktime": 0,
    "fee": 0.0001,
    "vin": [
        {
            "txid": "prev5555555555555555555555555555555555555555555555555555555555555555",
            "vout": 0,
            "scriptSig": {"asm": "", "hex": ""},
            "sequence": 4294967295
        }
    ],
    "vout": [
        {
            "value": 0.5,
            "n": 0,
            "scriptPubKey": {"type": "pubkeyhash", "address": "1addr"}
        }
    ]
}


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_p2pkh_tx():
    """Standard P2PKH transaction."""
    return SAMPLE_P2PKH_TX.copy()


@pytest.fixture
def sample_coinbase_tx():
    """Coinbase (mining reward) transaction."""
    return SAMPLE_COINBASE_TX.copy()


@pytest.fixture
def sample_segwit_tx():
    """Native SegWit transaction."""
    return SAMPLE_SEGWIT_TX.copy()


@pytest.fixture
def sample_op_return_tx():
    """OP_RETURN data embedding transaction."""
    return SAMPLE_OP_RETURN_TX.copy()


@pytest.fixture
def sample_large_tx():
    """Large transaction with many inputs/outputs."""
    return SAMPLE_LARGE_TX.copy()


@pytest.fixture
def sample_missing_vin_tx():
    """Transaction with empty vin list."""
    return SAMPLE_MISSING_VIN_TX.copy()


@pytest.fixture
def sample_mempool_tx():
    """Transaction with mempool context (fee info)."""
    return SAMPLE_MEMPOOL_TX.copy()
