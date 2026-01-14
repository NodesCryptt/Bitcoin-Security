#!/usr/bin/env python
"""
CP1 UTXO & Address Cache
========================
Production-ready Redis-backed cache for UTXO lookups and address history.

Provides:
- UTXO caching: Decoded UTXOs by txid:vout (TTL: 1 hour)
- Address history: Risk scores, tx counts, first-seen times (TTL: 24 hours)
- Batch operations: Bulk lookups for all inputs in a transaction
- Cache metrics: Hit rate, latency exposed to Prometheus
- Double-spend detection: Track spent UTXOs

Usage:
    cache = UTXOAddressCache(redis_url="redis://localhost:6379")
    
    # Get UTXO data
    utxo = cache.get_utxo("txid...", 0)
    
    # Get address risk
    risk = cache.get_address_risk("1BvBMSE...")
    
    # Batch lookup for tx inputs
    input_risk = cache.get_tx_input_risk(decoded_tx, rpc_client)
"""

import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Using in-memory fallback.")


# =============================================================================
# TTL CONSTANTS
# =============================================================================

TTL_UTXO = 3600           # 1 hour for UTXOs
TTL_ADDRESS = 86400       # 24 hours for address history
TTL_SPENT_UTXO = 600      # 10 minutes for spent UTXO tracking (double-spend detection)
TTL_DECODED_TX = 120      # 2 minutes for decoded tx cache


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UTXOData:
    """Cached UTXO data."""
    txid: str
    vout: int
    value: float
    script_type: str
    address: Optional[str]
    confirmations: int
    is_spent: bool = False
    cached_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UTXOData":
        return cls(**data)


@dataclass
class AddressHistory:
    """Cached address history and risk features."""
    address: str
    first_seen: Optional[str]  # ISO timestamp
    tx_count: int
    total_received: float
    total_sent: float
    risk_score: float  # 0.0 - 1.0
    risk_reason: str
    is_exchange: bool
    is_known_illicit: bool
    cluster_id: Optional[str]
    cached_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AddressHistory":
        return cls(**data)


@dataclass
class TxInputRisk:
    """Risk assessment for all inputs in a transaction."""
    txid: str
    num_inputs: int
    total_input_value: float
    avg_input_risk: float
    max_input_risk: float
    any_unknown_address: bool
    any_illicit_address: bool
    any_exchange_address: bool
    input_addresses: List[str]
    cache_hit_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# LRU CACHE (Fallback when Redis unavailable)
# =============================================================================

class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check expiry
            if key in self._expiry and time.time() > self._expiry[key]:
                del self._cache[key]
                del self._expiry[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: int = None):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    # Remove oldest
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                    if oldest in self._expiry:
                        del self._expiry[oldest]
            
            self._cache[key] = value
            if ttl:
                self._expiry[key] = time.time() + ttl
    
    def delete(self, key: str):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._expiry:
                del self._expiry[key]
    
    def size(self) -> int:
        return len(self._cache)


# =============================================================================
# UTXO ADDRESS CACHE
# =============================================================================

class UTXOAddressCache:
    """
    Production Redis-backed cache for UTXO and address data.
    
    Key patterns:
    - utxo:{txid}:{vout} -> UTXOData
    - addr:{address} -> AddressHistory
    - spent:{txid}:{vout} -> timestamp (for double-spend detection)
    - decoded:{txid} -> decoded transaction JSON
    """
    
    # Cache key prefixes
    PREFIX_UTXO = "utxo"
    PREFIX_ADDR = "addr"
    PREFIX_SPENT = "spent"
    PREFIX_DECODED = "decoded"
    
    def __init__(
        self,
        redis_url: str = None,
        redis_client = None,
        lru_maxsize: int = 50000
    ):
        """
        Initialize cache.
        
        Args:
            redis_url: Redis URL (redis://host:port/db)
            redis_client: Existing Redis client (optional)
            lru_maxsize: Max size for LRU fallback
        """
        self._redis = None
        self._lru = LRUCache(maxsize=lru_maxsize)
        
        # Stats
        self._stats = {
            "hits": 0,
            "misses": 0,
            "utxo_lookups": 0,
            "addr_lookups": 0,
            "batch_lookups": 0,
        }
        self._lock = threading.Lock()
        
        # Initialize Redis
        if redis_client:
            self._redis = redis_client
            logger.info("Using provided Redis client")
        elif redis_url and REDIS_AVAILABLE:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info(f"Connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using LRU fallback.")
                self._redis = None
        else:
            logger.info("Redis not available, using LRU fallback cache")
    
    @property
    def is_redis_available(self) -> bool:
        """Check if Redis is connected."""
        if not self._redis:
            return False
        try:
            self._redis.ping()
            return True
        except:
            return False
    
    # =========================================================================
    # UTXO CACHE METHODS
    # =========================================================================
    
    def get_utxo(self, txid: str, vout: int) -> Optional[UTXOData]:
        """
        Get UTXO data from cache.
        
        Args:
            txid: Transaction ID
            vout: Output index
            
        Returns:
            UTXOData or None if not cached
        """
        key = f"{self.PREFIX_UTXO}:{txid}:{vout}"
        with self._lock:
            self._stats["utxo_lookups"] += 1
        
        data = self._get(key)
        if data:
            with self._lock:
                self._stats["hits"] += 1
            return UTXOData.from_dict(data)
        
        with self._lock:
            self._stats["misses"] += 1
        return None
    
    def set_utxo(self, utxo: UTXOData):
        """Cache UTXO data."""
        key = f"{self.PREFIX_UTXO}:{utxo.txid}:{utxo.vout}"
        utxo.cached_at = datetime.utcnow().isoformat()
        self._set(key, utxo.to_dict(), TTL_UTXO)
    
    def mark_utxo_spent(self, txid: str, vout: int, spending_txid: str):
        """
        Mark a UTXO as spent (for double-spend detection).
        
        Args:
            txid: UTXO transaction ID
            vout: UTXO output index
            spending_txid: Transaction spending this UTXO
        """
        key = f"{self.PREFIX_SPENT}:{txid}:{vout}"
        data = {
            "spending_txid": spending_txid,
            "spent_at": datetime.utcnow().isoformat()
        }
        self._set(key, data, TTL_SPENT_UTXO)
    
    def is_utxo_spent(self, txid: str, vout: int) -> Tuple[bool, Optional[str]]:
        """
        Check if UTXO is already marked as spent.
        
        Returns:
            (is_spent, spending_txid)
        """
        key = f"{self.PREFIX_SPENT}:{txid}:{vout}"
        data = self._get(key)
        if data:
            return True, data.get("spending_txid")
        return False, None
    
    # =========================================================================
    # ADDRESS CACHE METHODS
    # =========================================================================
    
    def get_address(self, address: str) -> Optional[AddressHistory]:
        """
        Get address history from cache.
        
        Args:
            address: Bitcoin address
            
        Returns:
            AddressHistory or None
        """
        key = f"{self.PREFIX_ADDR}:{address}"
        with self._lock:
            self._stats["addr_lookups"] += 1
        
        data = self._get(key)
        if data:
            with self._lock:
                self._stats["hits"] += 1
            return AddressHistory.from_dict(data)
        
        with self._lock:
            self._stats["misses"] += 1
        return None
    
    def set_address(self, history: AddressHistory):
        """Cache address history."""
        key = f"{self.PREFIX_ADDR}:{history.address}"
        history.cached_at = datetime.utcnow().isoformat()
        self._set(key, history.to_dict(), TTL_ADDRESS)
    
    def get_address_risk(self, address: str) -> float:
        """
        Get risk score for an address.
        
        Returns:
            Risk score 0.0-1.0, or 0.5 if unknown
        """
        history = self.get_address(address)
        if history:
            return history.risk_score
        return 0.5  # Default moderate risk for unknown addresses
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    def get_tx_input_risk(
        self, 
        decoded_tx: Dict[str, Any],
        rpc = None,
        compute_missing: bool = True
    ) -> TxInputRisk:
        """
        Compute aggregate risk for all inputs in a transaction.
        
        This is the main integration point for CP1 inference.
        
        Args:
            decoded_tx: Decoded transaction from Bitcoin RPC
            rpc: Bitcoin RPC client for fetching missing UTXOs
            compute_missing: If True, fetch and cache missing UTXOs
            
        Returns:
            TxInputRisk with aggregate metrics
        """
        with self._lock:
            self._stats["batch_lookups"] += 1
        
        vin = decoded_tx.get("vin", [])
        txid = decoded_tx.get("txid", "unknown")
        
        if not vin:
            return TxInputRisk(
                txid=txid,
                num_inputs=0,
                total_input_value=0.0,
                avg_input_risk=0.0,
                max_input_risk=0.0,
                any_unknown_address=False,
                any_illicit_address=False,
                any_exchange_address=False,
                input_addresses=[],
                cache_hit_rate=1.0
            )
        
        # Collect input data
        input_values = []
        input_risks = []
        input_addresses = []
        cache_hits = 0
        unknown_addresses = False
        illicit_addresses = False
        exchange_addresses = False
        
        for inp in vin:
            # Skip coinbase inputs
            if "coinbase" in inp:
                continue
            
            prev_txid = inp.get("txid")
            prev_vout = inp.get("vout", 0)
            
            if not prev_txid:
                continue
            
            # Try to get UTXO from cache
            utxo = self.get_utxo(prev_txid, prev_vout)
            
            if not utxo and compute_missing and rpc:
                # Fetch from node
                try:
                    raw = rpc.getrawtransaction(prev_txid, True)
                    vout_data = raw.get("vout", [])[prev_vout] if len(raw.get("vout", [])) > prev_vout else None
                    
                    if vout_data:
                        script_pubkey = vout_data.get("scriptPubKey", {})
                        address = None
                        if "address" in script_pubkey:
                            address = script_pubkey["address"]
                        elif "addresses" in script_pubkey and script_pubkey["addresses"]:
                            address = script_pubkey["addresses"][0]
                        
                        utxo = UTXOData(
                            txid=prev_txid,
                            vout=prev_vout,
                            value=float(vout_data.get("value", 0)),
                            script_type=script_pubkey.get("type", "unknown"),
                            address=address,
                            confirmations=raw.get("confirmations", 0),
                            is_spent=True
                        )
                        self.set_utxo(utxo)
                except Exception as e:
                    logger.debug(f"Failed to fetch UTXO {prev_txid}:{prev_vout}: {e}")
            
            if utxo:
                cache_hits += 1
                input_values.append(utxo.value)
                
                if utxo.address:
                    input_addresses.append(utxo.address)
                    addr_history = self.get_address(utxo.address)
                    
                    if addr_history:
                        input_risks.append(addr_history.risk_score)
                        if addr_history.is_known_illicit:
                            illicit_addresses = True
                        if addr_history.is_exchange:
                            exchange_addresses = True
                    else:
                        # Unknown address - moderate risk
                        input_risks.append(0.5)
                        unknown_addresses = True
                else:
                    input_risks.append(0.5)
                    unknown_addresses = True
            else:
                # UTXO not found
                input_risks.append(0.5)
                unknown_addresses = True
        
        # Compute aggregates
        num_inputs = len(vin)
        cache_hit_rate = cache_hits / num_inputs if num_inputs > 0 else 1.0
        total_value = sum(input_values) if input_values else 0.0
        avg_risk = sum(input_risks) / len(input_risks) if input_risks else 0.5
        max_risk = max(input_risks) if input_risks else 0.5
        
        return TxInputRisk(
            txid=txid,
            num_inputs=num_inputs,
            total_input_value=total_value,
            avg_input_risk=avg_risk,
            max_input_risk=max_risk,
            any_unknown_address=unknown_addresses,
            any_illicit_address=illicit_addresses,
            any_exchange_address=exchange_addresses,
            input_addresses=input_addresses,
            cache_hit_rate=cache_hit_rate
        )
    
    # =========================================================================
    # DOUBLE-SPEND DETECTION
    # =========================================================================
    
    def check_double_spend(self, decoded_tx: Dict[str, Any]) -> List[Tuple[str, int, str]]:
        """
        Check if any inputs are already spent by another transaction.
        
        Args:
            decoded_tx: Decoded transaction
            
        Returns:
            List of (txid, vout, spending_txid) for double-spent inputs
        """
        double_spends = []
        
        for inp in decoded_tx.get("vin", []):
            if "coinbase" in inp:
                continue
            
            prev_txid = inp.get("txid")
            prev_vout = inp.get("vout", 0)
            
            if prev_txid:
                is_spent, spending_txid = self.is_utxo_spent(prev_txid, prev_vout)
                if is_spent:
                    double_spends.append((prev_txid, prev_vout, spending_txid))
        
        return double_spends
    
    def record_tx_inputs(self, decoded_tx: Dict[str, Any]):
        """
        Record all inputs of a transaction as spent.
        Call this after accepting a transaction.
        """
        txid = decoded_tx.get("txid")
        if not txid:
            return
        
        for inp in decoded_tx.get("vin", []):
            if "coinbase" in inp:
                continue
            
            prev_txid = inp.get("txid")
            prev_vout = inp.get("vout", 0)
            
            if prev_txid:
                self.mark_utxo_spent(prev_txid, prev_vout, txid)
    
    # =========================================================================
    # DECODED TX CACHE
    # =========================================================================
    
    def get_decoded_tx(self, txid: str) -> Optional[Dict[str, Any]]:
        """Get cached decoded transaction."""
        key = f"{self.PREFIX_DECODED}:{txid}"
        return self._get(key)
    
    def set_decoded_tx(self, txid: str, decoded: Dict[str, Any]):
        """Cache decoded transaction."""
        key = f"{self.PREFIX_DECODED}:{txid}"
        self._set(key, decoded, TTL_DECODED_TX)
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from Redis or LRU cache."""
        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.debug(f"Redis get error: {e}")
        
        # Fallback to LRU
        return self._lru.get(key)
    
    def _set(self, key: str, value: Dict[str, Any], ttl: int):
        """Set value in Redis or LRU cache."""
        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(value))
                return
            except Exception as e:
                logger.debug(f"Redis set error: {e}")
        
        # Fallback to LRU
        self._lru.set(key, value, ttl)
    
    def _delete(self, key: str):
        """Delete from cache."""
        if self._redis:
            try:
                self._redis.delete(key)
            except:
                pass
        self._lru.delete(key)
    
    # =========================================================================
    # STATS & MANAGEMENT
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = dict(self._stats)
        
        total = stats["hits"] + stats["misses"]
        stats["hit_rate"] = stats["hits"] / total if total > 0 else 0.0
        stats["is_redis"] = self.is_redis_available
        stats["lru_size"] = self._lru.size()
        
        return stats
    
    def clear(self):
        """Clear all caches (use with caution)."""
        self._lru = LRUCache(maxsize=self._lru.maxsize)
        with self._lock:
            self._stats = {k: 0 for k in self._stats}
        logger.info("Cache cleared")


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

_default_cache: Optional[UTXOAddressCache] = None


def get_utxo_cache(redis_url: str = None) -> UTXOAddressCache:
    """Get or create default UTXO cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = UTXOAddressCache(redis_url=redis_url)
    return _default_cache


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def enrich_features_with_cache(
    features: Dict[str, Any],
    decoded_tx: Dict[str, Any],
    cache: UTXOAddressCache,
    rpc = None
) -> Dict[str, Any]:
    """
    Enrich feature dict with cached UTXO/address data.
    
    This is the main integration point for cp1_live_infer_runtime.py.
    
    Args:
        features: Current feature dict
        decoded_tx: Decoded transaction
        cache: UTXO cache instance
        rpc: Bitcoin RPC client
        
    Returns:
        Enhanced feature dict
    """
    # Get input risk assessment
    input_risk = cache.get_tx_input_risk(decoded_tx, rpc)
    
    # Add new features
    features["input_avg_risk"] = input_risk.avg_input_risk
    features["input_max_risk"] = input_risk.max_input_risk
    features["total_input_value"] = input_risk.total_input_value
    features["any_unknown_addr"] = 1.0 if input_risk.any_unknown_address else 0.0
    features["any_illicit_addr"] = 1.0 if input_risk.any_illicit_address else 0.0
    features["any_exchange_addr"] = 1.0 if input_risk.any_exchange_address else 0.0
    features["cache_hit_rate"] = input_risk.cache_hit_rate
    
    # Check double-spend
    double_spends = cache.check_double_spend(decoded_tx)
    features["is_double_spend"] = 1.0 if double_spends else 0.0
    features["num_double_spend_inputs"] = float(len(double_spends))
    
    return features
