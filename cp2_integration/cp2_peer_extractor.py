#!/usr/bin/env python
"""
CP2 Peer Security Extractor
===========================
Monitors Bitcoin Core peer connections and detects suspicious behavior.

Detects:
- Sybil attacks (many peers from same subnet)
- Eclipse attempts (peer isolation patterns)
- Suspicious connection patterns

Publishes alerts to Redis for CP1 correlation.

Usage:
    python cp2_peer_extractor.py

Environment Variables:
    BITCOIN_RPC_URL: Bitcoin Core RPC URL
    REDIS_URL: Redis URL
    PROMETHEUS_PORT: Metrics port (default: 8001)
"""

import os
import sys
import time
import signal
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cp2.peer_extractor")

# Try imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available")

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available")

try:
    from bitcoinrpc.authproxy import AuthServiceProxy
except ImportError:
    logger.error("python-bitcoinrpc not installed")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CP2Config:
    """CP2 configuration."""
    rpc_url: str = "http://cp1user:CP1SecurePassword123!@127.0.0.1:18443"
    redis_url: str = "redis://localhost:6379"
    metrics_port: int = 8001
    poll_interval: float = 10.0  # seconds
    
    # Detection thresholds
    sybil_subnet_threshold: int = 5  # Max peers from same /24 subnet
    eclipse_min_peers: int = 3  # Minimum healthy peer count
    suspicious_version_age_days: int = 365  # Flag very old versions
    
    @classmethod
    def from_env(cls) -> "CP2Config":
        return cls(
            rpc_url=os.environ.get("BITCOIN_RPC_URL", cls.rpc_url),
            redis_url=os.environ.get("REDIS_URL", cls.redis_url),
            metrics_port=int(os.environ.get("PROMETHEUS_PORT", 8001)),
            poll_interval=float(os.environ.get("POLL_INTERVAL", 10.0)),
        )


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PeerInfo:
    """Information about a connected peer."""
    addr: str
    version: int
    subver: str
    inbound: bool
    connection_time: float
    last_send: float
    last_recv: float
    bytessent: int
    bytesrecv: int
    synced_headers: int
    synced_blocks: int
    services: str
    subnet: str = ""
    
    @classmethod
    def from_rpc(cls, data: Dict) -> "PeerInfo":
        addr = data.get("addr", "unknown")
        # Extract subnet (first 3 octets for /24)
        try:
            ip = addr.split(":")[0]
            subnet = ".".join(ip.split(".")[:3])
        except:
            subnet = "unknown"
        
        return cls(
            addr=addr,
            version=data.get("version", 0),
            subver=data.get("subver", ""),
            inbound=data.get("inbound", False),
            connection_time=data.get("conntime", 0),
            last_send=data.get("lastsend", 0),
            last_recv=data.get("lastrecv", 0),
            bytessent=data.get("bytessent", 0),
            bytesrecv=data.get("bytesrecv", 0),
            synced_headers=data.get("synced_headers", -1),
            synced_blocks=data.get("synced_blocks", -1),
            services=data.get("servicesnames", ""),
            subnet=subnet
        )


@dataclass
class PeerAlert:
    """Alert for suspicious peer behavior."""
    event: str
    peer_addr: str
    score: float
    reason: str
    details: Dict[str, Any]
    timestamp: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


# =============================================================================
# METRICS
# =============================================================================

class CP2Metrics:
    """Prometheus metrics for CP2."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        
        if PROMETHEUS_AVAILABLE:
            self.peer_count = Gauge("cp2_peer_count", "Number of connected peers", ["type"])
            self.suspicious_peer_count = Counter("cp2_suspicious_peer_count", "Suspicious peers detected", ["reason"])
            self.sybil_score = Gauge("cp2_sybil_score", "Current Sybil attack score")
            self.eclipse_score = Gauge("cp2_eclipse_score", "Current Eclipse attack score")
            self.poll_duration = Histogram("cp2_poll_duration_seconds", "Time to poll peer info")
            self.alerts_published = Counter("cp2_alerts_published_total", "Alerts published to Redis")
    
    def start_server(self):
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(self.port)
                logger.info(f"CP2 Prometheus metrics on port {self.port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")


# =============================================================================
# DETECTOR
# =============================================================================

class PeerSecurityDetector:
    """Detects Sybil, Eclipse, and other peer-based attacks."""
    
    def __init__(self, config: CP2Config):
        self.config = config
        self._peer_history: Dict[str, List[float]] = defaultdict(list)  # addr -> connection times
        self._known_peers: Set[str] = set()
    
    def analyze_peers(self, peers: List[PeerInfo]) -> List[PeerAlert]:
        """Analyze peer list for suspicious patterns."""
        alerts = []
        
        # Update known peers
        current_addrs = {p.addr for p in peers}
        new_peers = current_addrs - self._known_peers
        disconnected = self._known_peers - current_addrs
        self._known_peers = current_addrs
        
        # Sybil detection: too many peers from same subnet
        sybil_alerts = self._detect_sybil(peers)
        alerts.extend(sybil_alerts)
        
        # Eclipse detection: too few peers
        eclipse_alerts = self._detect_eclipse(peers)
        alerts.extend(eclipse_alerts)
        
        # Suspicious version detection
        version_alerts = self._detect_suspicious_versions(peers)
        alerts.extend(version_alerts)
        
        # Log peer churn
        if new_peers:
            logger.info(f"New peers connected: {len(new_peers)}")
        if disconnected:
            logger.info(f"Peers disconnected: {len(disconnected)}")
        
        return alerts
    
    def _detect_sybil(self, peers: List[PeerInfo]) -> List[PeerAlert]:
        """Detect potential Sybil attack from subnet clustering."""
        alerts = []
        
        subnet_counts = defaultdict(list)
        for peer in peers:
            subnet_counts[peer.subnet].append(peer)
        
        for subnet, subnet_peers in subnet_counts.items():
            if len(subnet_peers) >= self.config.sybil_subnet_threshold:
                score = min(1.0, len(subnet_peers) / 10.0)
                
                for peer in subnet_peers:
                    alerts.append(PeerAlert(
                        event="sybil_cluster",
                        peer_addr=peer.addr,
                        score=score,
                        reason=f"Part of suspicious cluster: {len(subnet_peers)} peers from {subnet}.0/24",
                        details={
                            "subnet": subnet,
                            "peer_count_in_subnet": len(subnet_peers),
                            "threshold": self.config.sybil_subnet_threshold
                        },
                        timestamp=datetime.utcnow().isoformat()
                    ))
                
                logger.warning(f"Sybil cluster detected: {len(subnet_peers)} peers from {subnet}.0/24")
        
        return alerts
    
    def _detect_eclipse(self, peers: List[PeerInfo]) -> List[PeerAlert]:
        """Detect potential Eclipse attack from low peer count."""
        alerts = []
        
        outbound = [p for p in peers if not p.inbound]
        
        if len(outbound) < self.config.eclipse_min_peers:
            score = 1.0 - (len(outbound) / self.config.eclipse_min_peers)
            
            alerts.append(PeerAlert(
                event="eclipse_warning",
                peer_addr="node",
                score=score,
                reason=f"Low outbound peer count: {len(outbound)} (min: {self.config.eclipse_min_peers})",
                details={
                    "outbound_count": len(outbound),
                    "inbound_count": len(peers) - len(outbound),
                    "total_peers": len(peers),
                    "threshold": self.config.eclipse_min_peers
                },
                timestamp=datetime.utcnow().isoformat()
            ))
            
            logger.warning(f"Eclipse warning: only {len(outbound)} outbound peers")
        
        return alerts
    
    def _detect_suspicious_versions(self, peers: List[PeerInfo]) -> List[PeerAlert]:
        """Detect peers running suspicious/outdated versions."""
        alerts = []
        
        for peer in peers:
            # Flag empty or very old subver
            if not peer.subver or peer.version < 70015:  # Before segwit
                alerts.append(PeerAlert(
                    event="suspicious_version",
                    peer_addr=peer.addr,
                    score=0.3,
                    reason=f"Outdated or missing version: {peer.subver} (v{peer.version})",
                    details={
                        "version": peer.version,
                        "subver": peer.subver
                    },
                    timestamp=datetime.utcnow().isoformat()
                ))
        
        return alerts


# =============================================================================
# REDIS PUBLISHER
# =============================================================================

class CP2RedisPublisher:
    """Publishes CP2 alerts to Redis."""
    
    CHANNEL = "cp2:alerts"
    PEER_RISK_PREFIX = "cp2:peer_risk:"
    TTL_PEER_RISK = 3600  # 1 hour
    
    def __init__(self, redis_url: str):
        self._client = None
        if REDIS_AVAILABLE:
            try:
                self._client = redis.from_url(redis_url, decode_responses=True)
                self._client.ping()
                logger.info(f"CP2 Redis connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
    
    @property
    def is_connected(self) -> bool:
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except:
            return False
    
    def publish_alert(self, alert: PeerAlert) -> bool:
        """Publish alert to Redis channel."""
        if not self.is_connected:
            return False
        
        try:
            self._client.publish(self.CHANNEL, alert.to_json())
            
            # Also store peer risk score
            if alert.peer_addr and alert.peer_addr != "node":
                key = f"{self.PEER_RISK_PREFIX}{alert.peer_addr}"
                self._client.setex(key, self.TTL_PEER_RISK, str(alert.score))
            
            return True
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")
            return False
    
    def close(self):
        if self._client:
            self._client.close()


# =============================================================================
# MAIN RUNTIME
# =============================================================================

class CP2PeerExtractor:
    """Main CP2 peer security runtime."""
    
    def __init__(self, config: CP2Config = None):
        self.config = config or CP2Config.from_env()
        self.running = False
        
        logger.info("=" * 60)
        logger.info("CP2 PEER SECURITY EXTRACTOR")
        logger.info("=" * 60)
        
        # Initialize components
        self._init_rpc()
        self._init_redis()
        self._init_metrics()
        
        self.detector = PeerSecurityDetector(self.config)
        
        # Stats
        self._stats = {
            "polls": 0,
            "alerts": 0,
            "sybil_events": 0,
            "eclipse_events": 0,
        }
        
        logger.info(f"Poll interval: {self.config.poll_interval}s")
        logger.info("=" * 60)
    
    def _init_rpc(self):
        """Initialize Bitcoin Core RPC."""
        logger.info("Connecting to Bitcoin Core...")
        try:
            self.rpc = AuthServiceProxy(self.config.rpc_url)
            info = self.rpc.getnetworkinfo()
            logger.info(f"Connected: version={info['version']}, connections={info['connections']}")
        except Exception as e:
            logger.error(f"Bitcoin Core connection failed: {e}")
            raise
    
    def _init_redis(self):
        """Initialize Redis publisher."""
        self.redis = CP2RedisPublisher(self.config.redis_url)
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        self.metrics = CP2Metrics(self.config.metrics_port)
        self.metrics.start_server()
    
    def _poll_peers(self) -> List[PeerInfo]:
        """Get current peer list from Bitcoin Core."""
        try:
            raw_peers = self.rpc.getpeerinfo()
            return [PeerInfo.from_rpc(p) for p in raw_peers]
        except Exception as e:
            logger.error(f"Failed to get peer info: {e}")
            return []
    
    def _update_metrics(self, peers: List[PeerInfo], alerts: List[PeerAlert]):
        """Update Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        inbound = sum(1 for p in peers if p.inbound)
        outbound = len(peers) - inbound
        
        self.metrics.peer_count.labels(type="inbound").set(inbound)
        self.metrics.peer_count.labels(type="outbound").set(outbound)
        self.metrics.peer_count.labels(type="total").set(len(peers))
        
        # Update attack scores
        sybil_alerts = [a for a in alerts if "sybil" in a.event]
        eclipse_alerts = [a for a in alerts if "eclipse" in a.event]
        
        if sybil_alerts:
            self.metrics.sybil_score.set(max(a.score for a in sybil_alerts))
        else:
            self.metrics.sybil_score.set(0)
        
        if eclipse_alerts:
            self.metrics.eclipse_score.set(max(a.score for a in eclipse_alerts))
        else:
            self.metrics.eclipse_score.set(0)
    
    def run(self):
        """Main run loop."""
        self.running = True
        logger.info("CP2 Peer Extractor started. Monitoring peers...")
        
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while self.running:
            try:
                start_time = time.perf_counter()
                
                # Poll peers
                peers = self._poll_peers()
                self._stats["polls"] += 1
                
                # Analyze for threats
                alerts = self.detector.analyze_peers(peers)
                
                # Publish alerts
                for alert in alerts:
                    if self.redis.publish_alert(alert):
                        self._stats["alerts"] += 1
                        if PROMETHEUS_AVAILABLE:
                            self.metrics.alerts_published.inc()
                            self.metrics.suspicious_peer_count.labels(reason=alert.event).inc()
                    
                    if "sybil" in alert.event:
                        self._stats["sybil_events"] += 1
                    elif "eclipse" in alert.event:
                        self._stats["eclipse_events"] += 1
                
                # Update metrics
                self._update_metrics(peers, alerts)
                
                poll_time = time.perf_counter() - start_time
                if PROMETHEUS_AVAILABLE:
                    self.metrics.poll_duration.observe(poll_time)
                
                # Log status
                if self._stats["polls"] % 6 == 0:  # Every minute
                    logger.info(f"Status: peers={len(peers)}, polls={self._stats['polls']}, alerts={self._stats['alerts']}")
                
                # Wait for next poll
                time.sleep(self.config.poll_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Poll error: {e}")
                time.sleep(5)
        
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down CP2 Peer Extractor...")
        logger.info(f"Stats: {self._stats}")
        self.redis.close()
        logger.info("CP2 stopped.")


def main():
    config = CP2Config.from_env()
    extractor = CP2PeerExtractor(config)
    extractor.run()


if __name__ == "__main__":
    main()
