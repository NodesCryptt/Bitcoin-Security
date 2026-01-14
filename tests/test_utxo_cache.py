#!/usr/bin/env python
"""
CP1 UTXO Cache Unit Tests
=========================
Tests for the UTXOAddressCache component.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime


# Import module under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from utxo_address_cache import (
    UTXOAddressCache, 
    UTXOData, 
    AddressHistory, 
    TxInputRisk,
    LRUCache,
    enrich_features_with_cache
)


# =============================================================================
# LRU CACHE TESTS
# =============================================================================

class TestLRUCache:
    """Tests for LRU cache fallback."""
    
    def test_basic_get_set(self):
        """Basic get/set operations."""
        cache = LRUCache(maxsize=100)
        cache.set("key1", {"value": 123})
        result = cache.get("key1")
        assert result == {"value": 123}
    
    def test_missing_key_returns_none(self):
        """Missing key returns None."""
        cache = LRUCache()
        assert cache.get("nonexistent") is None
    
    def test_ttl_expiration(self):
        """Values expire after TTL."""
        cache = LRUCache()
        cache.set("key", {"data": "test"}, ttl=1)
        assert cache.get("key") is not None
        time.sleep(1.1)
        assert cache.get("key") is None
    
    def test_lru_eviction(self):
        """Oldest entries evicted when full."""
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a"
        
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4
    
    def test_access_updates_lru_order(self):
        """Accessing a key moves it to end."""
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        # Access "a" to make it recent
        cache.get("a")
        
        # Add new entry - should evict "b" not "a"
        cache.set("d", 4)
        
        assert cache.get("a") == 1
        assert cache.get("b") is None


# =============================================================================
# UTXO CACHE TESTS
# =============================================================================

class TestUTXOAddressCache:
    """Tests for UTXO cache without Redis."""
    
    @pytest.fixture
    def cache(self):
        """Create cache with LRU fallback only."""
        return UTXOAddressCache(redis_url=None, lru_maxsize=1000)
    
    def test_utxo_set_get(self, cache):
        """UTXO caching works."""
        utxo = UTXOData(
            txid="abc123",
            vout=0,
            value=1.5,
            script_type="pubkeyhash",
            address="1ABC...",
            confirmations=6
        )
        cache.set_utxo(utxo)
        
        result = cache.get_utxo("abc123", 0)
        assert result is not None
        assert result.txid == "abc123"
        assert result.value == 1.5
        assert result.address == "1ABC..."
    
    def test_utxo_miss(self, cache):
        """Missing UTXO returns None."""
        assert cache.get_utxo("nonexistent", 0) is None
    
    def test_address_set_get(self, cache):
        """Address history caching works."""
        history = AddressHistory(
            address="1ABC...",
            first_seen="2024-01-01T00:00:00",
            tx_count=100,
            total_received=50.0,
            total_sent=48.0,
            risk_score=0.2,
            risk_reason="Low activity",
            is_exchange=False,
            is_known_illicit=False,
            cluster_id=None
        )
        cache.set_address(history)
        
        result = cache.get_address("1ABC...")
        assert result is not None
        assert result.tx_count == 100
        assert result.risk_score == 0.2
    
    def test_address_risk_default(self, cache):
        """Unknown address returns default risk."""
        risk = cache.get_address_risk("unknown_addr")
        assert risk == 0.5  # Default moderate risk
    
    def test_double_spend_detection(self, cache):
        """Double-spend detection works."""
        # Mark UTXO as spent
        cache.mark_utxo_spent("tx1", 0, "spending_tx1")
        
        # Check if spent
        is_spent, spending_txid = cache.is_utxo_spent("tx1", 0)
        assert is_spent is True
        assert spending_txid == "spending_tx1"
        
        # Unspent UTXO
        is_spent, spending_txid = cache.is_utxo_spent("tx2", 0)
        assert is_spent is False
        assert spending_txid is None
    
    def test_stats(self, cache):
        """Stats tracking works."""
        cache.get_utxo("miss1", 0)
        cache.get_utxo("miss2", 0)
        
        utxo = UTXOData("hit", 0, 1.0, "p2pkh", "addr", 1)
        cache.set_utxo(utxo)
        cache.get_utxo("hit", 0)
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 1/3


# =============================================================================
# TX INPUT RISK TESTS
# =============================================================================

class TestTxInputRisk:
    """Tests for transaction input risk assessment."""
    
    @pytest.fixture
    def cache(self):
        return UTXOAddressCache(redis_url=None)
    
    def test_empty_inputs(self, cache):
        """Empty vin returns safe defaults."""
        decoded = {"txid": "test", "vin": []}
        result = cache.get_tx_input_risk(decoded)
        
        assert result.num_inputs == 0
        assert result.avg_input_risk == 0.0
        assert result.cache_hit_rate == 1.0
    
    def test_coinbase_ignored(self, cache):
        """Coinbase inputs are skipped."""
        decoded = {
            "txid": "coinbase_tx",
            "vin": [{"coinbase": "abc123"}]
        }
        result = cache.get_tx_input_risk(decoded)
        assert result.num_inputs == 1
    
    def test_cached_utxo_used(self, cache):
        """Cached UTXOs are used for risk calculation."""
        # Pre-cache a UTXO
        utxo = UTXOData("prev_tx", 0, 1.0, "p2pkh", "1ABC...", 10)
        cache.set_utxo(utxo)
        
        # Pre-cache address with known risk
        history = AddressHistory(
            "1ABC...", None, 10, 5.0, 4.0, 0.8, "High risk",
            False, False, None
        )
        cache.set_address(history)
        
        decoded = {
            "txid": "test_tx",
            "vin": [{"txid": "prev_tx", "vout": 0}]
        }
        result = cache.get_tx_input_risk(decoded, compute_missing=False)
        
        assert result.avg_input_risk == 0.8
        assert result.max_input_risk == 0.8
        assert result.cache_hit_rate == 1.0


# =============================================================================
# FEATURE ENRICHMENT TESTS
# =============================================================================

class TestFeatureEnrichment:
    """Tests for feature enrichment function."""
    
    def test_enrich_adds_fields(self):
        """Enrichment adds expected fields."""
        cache = UTXOAddressCache(redis_url=None)
        features = {"size": 250, "fees": 0.0001}
        decoded = {"txid": "test", "vin": []}
        
        enriched = enrich_features_with_cache(features, decoded, cache)
        
        assert "input_avg_risk" in enriched
        assert "input_max_risk" in enriched
        assert "is_double_spend" in enriched
        assert "cache_hit_rate" in enriched
    
    def test_double_spend_flagged(self):
        """Double-spend is flagged in features."""
        cache = UTXOAddressCache(redis_url=None)
        
        # Mark UTXO as spent
        cache.mark_utxo_spent("prev", 0, "other_tx")
        
        features = {}
        decoded = {
            "txid": "double_spend_tx",
            "vin": [{"txid": "prev", "vout": 0}]
        }
        
        enriched = enrich_features_with_cache(features, decoded, cache)
        
        assert enriched["is_double_spend"] == 1.0
        assert enriched["num_double_spend_inputs"] == 1.0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
