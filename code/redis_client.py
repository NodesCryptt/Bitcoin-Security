"""
CP1 Redis Client
================
Redis client for CP1-CP2 communication and feature caching.

Provides:
- Feature caching (address history, decoded transactions)
- Message bus for CP1→CP2 alerts
- CP2→CP1 peer risk lookup
"""

import json
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
import os

logger = logging.getLogger(__name__)

# Try to import redis, with graceful fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Using in-memory fallback.")


@dataclass
class TxAlert:
    """Alert published when a transaction is flagged/rejected."""
    event: str  # "tx_flagged" or "tx_rejected"
    txid: str
    score: float
    decision: str
    announcing_peer: Optional[str]
    timestamp: str
    explanation: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class PeerAlert:
    """Alert received from CP2 about suspicious peers."""
    event: str  # "suspicious_peer"
    peer_addr: str
    score: float
    reason: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_json(cls, data: str) -> "PeerAlert":
        d = json.loads(data)
        # Handle extra fields gracefully
        return cls(
            event=d.get("event", "unknown"),
            peer_addr=d.get("peer_addr", "unknown"),
            score=d.get("score", 0.0),
            reason=d.get("reason", ""),
            timestamp=d.get("timestamp", ""),
            details=d.get("details")
        )



class CP1RedisClient:
    """
    Redis client for CP1.
    
    Handles:
    - Feature caching with TTL
    - Publishing transaction alerts
    - Subscribing to peer alerts from CP2
    - Peer risk lookup
    """
    
    # Channel names
    CHANNEL_CP1_ALERTS = "cp1:alerts"
    CHANNEL_CP2_ALERTS = "cp2:alerts"
    
    # Key prefixes
    PREFIX_ADDR = "addr:"
    PREFIX_TX = "tx:"
    PREFIX_PEER_RISK = "cp2:peer_risk:"
    PREFIX_CLUSTER = "cluster:"
    
    # TTLs (seconds)
    TTL_ADDR = 86400      # 24 hours for address history
    TTL_TX = 60           # 60 seconds for mempool tx
    TTL_PEER = 3600       # 1 hour for peer risk
    TTL_CLUSTER = 21600   # 6 hours for cluster info
    
    def __init__(
        self,
        redis_url: str = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str = None
    ):
        """
        Initialize Redis client.
        
        Args:
            redis_url: Redis URL (redis://host:port/db)
            host: Redis host (if not using URL)
            port: Redis port (if not using URL)
            db: Redis database number
            password: Redis password (optional)
        """
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._subscriber_thread: Optional[threading.Thread] = None
        self._peer_alert_callbacks: List[Callable[[PeerAlert], None]] = []
        
        self._fallback_cache: Dict[str, Any] = {}  # In-memory fallback
        
        if REDIS_AVAILABLE:
            self._connect()
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            if self.redis_url:
                self._client = redis.from_url(self.redis_url)
            else:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True
                )
            
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = None
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not REDIS_AVAILABLE or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except:
            return False
    
    # =========================================================================
    # FEATURE CACHING
    # =========================================================================
    
    def cache_address_features(self, address: str, features: Dict[str, Any]) -> bool:
        """
        Cache features for an address.
        
        Args:
            address: Bitcoin address
            features: Feature dictionary
        
        Returns:
            True if cached successfully
        """
        key = f"{self.PREFIX_ADDR}{address}"
        return self._set_json(key, features, self.TTL_ADDR)
    
    def get_address_features(self, address: str) -> Optional[Dict[str, Any]]:
        """Get cached features for an address."""
        key = f"{self.PREFIX_ADDR}{address}"
        return self._get_json(key)
    
    def cache_decoded_tx(self, txid: str, decoded: Dict[str, Any]) -> bool:
        """Cache a decoded transaction."""
        key = f"{self.PREFIX_TX}{txid}"
        return self._set_json(key, decoded, self.TTL_TX)
    
    def get_decoded_tx(self, txid: str) -> Optional[Dict[str, Any]]:
        """Get cached decoded transaction."""
        key = f"{self.PREFIX_TX}{txid}"
        return self._get_json(key)
    
    # =========================================================================
    # CP1 → CP2 ALERTS (PUBLISHING)
    # =========================================================================
    
    def publish_tx_decision(
        self,
        txid: str,
        score: float,
        decision: str,
        announcing_peer: str = None,
        explanation: Dict[str, Any] = None
    ) -> bool:
        """
        Publish a transaction decision to CP2.
        
        Args:
            txid: Transaction ID
            score: ML score
            decision: Decision (ACCEPT/FLAG/REJECT)
            announcing_peer: Peer that announced this tx
            explanation: SHAP explanation (optional)
        
        Returns:
            True if published successfully
        """
        if decision == "ACCEPT":
            return True  # Don't publish ACCEPT decisions
        
        event = "tx_flagged" if decision == "FLAG" else "tx_rejected"
        
        alert = TxAlert(
            event=event,
            txid=txid,
            score=score,
            decision=decision,
            announcing_peer=announcing_peer,
            timestamp=datetime.utcnow().isoformat(),
            explanation=explanation
        )
        
        return self._publish(self.CHANNEL_CP1_ALERTS, alert.to_json())
    
    # =========================================================================
    # CP2 → CP1 ALERTS (SUBSCRIBING)
    # =========================================================================
    
    def subscribe_peer_alerts(self, callback: Callable[[PeerAlert], None]):
        """
        Subscribe to peer alerts from CP2.
        
        Args:
            callback: Function to call when peer alert received
        """
        self._peer_alert_callbacks.append(callback)
        
        if self._subscriber_thread is None:
            self._start_subscriber()
    
    def _start_subscriber(self):
        """Start background subscriber thread."""
        if not self.is_connected:
            logger.warning("Cannot start subscriber - Redis not connected")
            return
        
        self._pubsub = self._client.pubsub()
        self._pubsub.subscribe(self.CHANNEL_CP2_ALERTS)
        
        def listener():
            for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        alert = PeerAlert.from_json(message["data"])
                        for callback in self._peer_alert_callbacks:
                            callback(alert)
                    except Exception as e:
                        logger.error(f"Error processing peer alert: {e}")
        
        self._subscriber_thread = threading.Thread(target=listener, daemon=True)
        self._subscriber_thread.start()
        logger.info(f"Subscribed to {self.CHANNEL_CP2_ALERTS}")
    
    # =========================================================================
    # PEER RISK LOOKUP
    # =========================================================================
    
    def get_peer_risk(self, peer_addr: str) -> float:
        """
        Get risk score for a peer (from CP2).
        
        Args:
            peer_addr: Peer address (IP:port)
        
        Returns:
            Risk score (0-1), 0 if not found
        """
        if not peer_addr:
            return 0.0
        
        key = f"{self.PREFIX_PEER_RISK}{peer_addr}"
        
        if self.is_connected:
            try:
                value = self._client.get(key)
                return float(value) if value else 0.0
            except Exception as e:
                logger.error(f"Error getting peer risk: {e}")
        
        return self._fallback_cache.get(key, 0.0)
    
    def set_peer_risk(self, peer_addr: str, risk: float) -> bool:
        """Set risk score for a peer (for testing)."""
        key = f"{self.PREFIX_PEER_RISK}{peer_addr}"
        
        if self.is_connected:
            try:
                self._client.setex(key, self.TTL_PEER, str(risk))
                return True
            except Exception as e:
                logger.error(f"Error setting peer risk: {e}")
        
        self._fallback_cache[key] = risk
        return True
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _set_json(self, key: str, value: Dict[str, Any], ttl: int) -> bool:
        """Set a JSON value with TTL."""
        if self.is_connected:
            try:
                self._client.setex(key, ttl, json.dumps(value))
                return True
            except Exception as e:
                logger.error(f"Redis SET error: {e}")
        
        self._fallback_cache[key] = value
        return True
    
    def _get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a JSON value."""
        if self.is_connected:
            try:
                value = self._client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                logger.error(f"Redis GET error: {e}")
        
        return self._fallback_cache.get(key)
    
    def _publish(self, channel: str, message: str) -> bool:
        """Publish a message to a channel."""
        if self.is_connected:
            try:
                self._client.publish(channel, message)
                logger.debug(f"Published to {channel}: {message[:100]}...")
                return True
            except Exception as e:
                logger.error(f"Redis PUBLISH error: {e}")
        
        logger.warning(f"Cannot publish - Redis not connected")
        return False
    
    def close(self):
        """Close Redis connection."""
        if self._pubsub:
            self._pubsub.close()
        if self._client:
            self._client.close()
        logger.info("Redis client closed")


# Default client instance
_default_client: Optional[CP1RedisClient] = None


def get_redis_client() -> CP1RedisClient:
    """Get or create default Redis client."""
    global _default_client
    if _default_client is None:
        _default_client = CP1RedisClient()
    return _default_client
