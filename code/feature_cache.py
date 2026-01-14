"""
CP1 Feature Cache
=================
Redis-based feature caching layer for expensive lookups.

Caches:
- Address history features (TTL: 24h)
- Decoded transactions (TTL: 60s for mempool churn)
- UTXO data (TTL: 1h)
- Graph cluster membership (TTL: 6h)

Falls back to LRU in-memory cache if Redis unavailable.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import OrderedDict
import threading
import time

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
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
        """Set value in cache with optional TTL."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            
            if ttl:
                self._expiry[key] = time.time() + ttl
            
            # Evict if over capacity
            while len(self._cache) > self.maxsize:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                self._expiry.pop(oldest, None)
    
    def delete(self, key: str):
        """Delete key from cache."""
        with self._lock:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()


class FeatureCache:
    """
    Feature caching layer for CP1.
    
    Provides fast access to:
    - Address features (age, tx counts, risk scores)
    - Decoded transactions
    - UTXO data
    - Graph clusters
    
    Usage:
        cache = FeatureCache()
        
        # Cache address features
        cache.set_address_features("1A1zP1...", {"age_days": 100, ...})
        features = cache.get_address_features("1A1zP1...")
        
        # Cache decoded transaction
        cache.set_decoded_tx("abc123...", decoded_dict)
        decoded = cache.get_decoded_tx("abc123...")
    """
    
    # TTLs in seconds
    TTL_ADDRESS = 86400      # 24 hours
    TTL_TX = 60              # 60 seconds (mempool churn)
    TTL_UTXO = 3600          # 1 hour
    TTL_CLUSTER = 21600      # 6 hours
    
    # Key prefixes
    PREFIX_ADDR = "cp1:addr:"
    PREFIX_TX = "cp1:tx:"
    PREFIX_UTXO = "cp1:utxo:"
    PREFIX_CLUSTER = "cp1:cluster:"
    
    def __init__(
        self,
        redis_client=None,
        redis_url: str = None,
        lru_maxsize: int = 10000
    ):
        """
        Initialize feature cache.
        
        Args:
            redis_client: Existing Redis client (optional)
            redis_url: Redis URL (optional)
            lru_maxsize: Max size for LRU fallback cache
        """
        self._redis = redis_client
        self._lru = LRUCache(maxsize=lru_maxsize)
        
        # Stats
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
        }
        
        if redis_url and REDIS_AVAILABLE and not self._redis:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info(f"Feature cache connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed, using LRU: {e}")
                self._redis = None
    
    @property
    def is_redis_available(self) -> bool:
        """Check if Redis is available."""
        if not self._redis:
            return False
        try:
            self._redis.ping()
            return True
        except:
            return False
    
    # =========================================================================
    # ADDRESS FEATURES
    # =========================================================================
    
    def get_address_features(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Get cached features for an address.
        
        Returns:
            Feature dict or None if not cached
        """
        key = f"{self.PREFIX_ADDR}{address}"
        return self._get(key)
    
    def set_address_features(self, address: str, features: Dict[str, Any]) -> bool:
        """Cache features for an address."""
        key = f"{self.PREFIX_ADDR}{address}"
        return self._set(key, features, self.TTL_ADDRESS)
    
    def get_or_compute_address_features(
        self,
        address: str,
        compute_fn,
        *args, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached features or compute if not cached.
        
        Args:
            address: Bitcoin address
            compute_fn: Function to compute features if not cached
            *args, **kwargs: Arguments for compute_fn
        
        Returns:
            Feature dict
        """
        cached = self.get_address_features(address)
        if cached is not None:
            return cached
        
        features = compute_fn(*args, **kwargs)
        if features:
            self.set_address_features(address, features)
        return features
    
    # =========================================================================
    # DECODED TRANSACTIONS
    # =========================================================================
    
    def get_decoded_tx(self, txid: str) -> Optional[Dict[str, Any]]:
        """Get cached decoded transaction."""
        key = f"{self.PREFIX_TX}{txid}"
        return self._get(key)
    
    def set_decoded_tx(self, txid: str, decoded: Dict[str, Any]) -> bool:
        """Cache decoded transaction."""
        key = f"{self.PREFIX_TX}{txid}"
        return self._set(key, decoded, self.TTL_TX)
    
    # =========================================================================
    # UTXO DATA
    # =========================================================================
    
    def get_utxo(self, txid: str, vout: int) -> Optional[Dict[str, Any]]:
        """Get cached UTXO data."""
        key = f"{self.PREFIX_UTXO}{txid}:{vout}"
        return self._get(key)
    
    def set_utxo(self, txid: str, vout: int, data: Dict[str, Any]) -> bool:
        """Cache UTXO data."""
        key = f"{self.PREFIX_UTXO}{txid}:{vout}"
        return self._set(key, data, self.TTL_UTXO)
    
    # =========================================================================
    # GRAPH CLUSTERS
    # =========================================================================
    
    def get_address_cluster(self, address: str) -> Optional[str]:
        """Get cluster ID for an address."""
        key = f"{self.PREFIX_CLUSTER}{address}"
        result = self._get(key)
        return result.get("cluster_id") if result else None
    
    def set_address_cluster(self, address: str, cluster_id: str, is_illicit: bool = False) -> bool:
        """Cache cluster membership for an address."""
        key = f"{self.PREFIX_CLUSTER}{address}"
        return self._set(key, {"cluster_id": cluster_id, "is_illicit": is_illicit}, self.TTL_CLUSTER)
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _get(self, key: str) -> Optional[Any]:
        """Get value from cache (Redis or LRU)."""
        if self.is_redis_available:
            try:
                import json
                value = self._redis.get(key)
                if value:
                    self._stats["hits"] += 1
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Redis GET error: {e}")
        
        # Fallback to LRU
        value = self._lru.get(key)
        if value:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1
        return value
    
    def _set(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in cache (Redis or LRU)."""
        self._stats["sets"] += 1
        
        if self.is_redis_available:
            try:
                import json
                self._redis.setex(key, ttl, json.dumps(value))
                return True
            except Exception as e:
                logger.error(f"Redis SET error: {e}")
        
        # Fallback to LRU
        self._lru.set(key, value, ttl)
        return True
    
    def _delete(self, key: str):
        """Delete from cache."""
        if self.is_redis_available:
            try:
                self._redis.delete(key)
            except Exception as e:
                logger.error(f"Redis DELETE error: {e}")
        
        self._lru.delete(key)
    
    # =========================================================================
    # STATS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        total = self._stats["hits"] + self._stats["misses"]
        if total > 0:
            hit_rate = self._stats["hits"] / total
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "hit_rate": round(hit_rate, 4),
            "lru_size": self._lru.size(),
            "redis_available": self.is_redis_available,
        }
    
    def clear(self):
        """Clear all caches."""
        if self.is_redis_available:
            try:
                # Clear only our prefixes
                for prefix in [self.PREFIX_ADDR, self.PREFIX_TX, self.PREFIX_UTXO, self.PREFIX_CLUSTER]:
                    keys = self._redis.keys(f"{prefix}*")
                    if keys:
                        self._redis.delete(*keys)
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
        
        self._lru.clear()
        self._stats = {"hits": 0, "misses": 0, "sets": 0}


# Default cache instance
_default_cache: Optional[FeatureCache] = None


def get_feature_cache(redis_url: str = None) -> FeatureCache:
    """Get or create default feature cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = FeatureCache(redis_url=redis_url)
    return _default_cache
