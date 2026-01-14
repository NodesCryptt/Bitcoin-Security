#!/usr/bin/env python
"""
CP1 Consensus Guard with IBD-Safe Guards
=========================================
Provides IBD-aware transaction processing with ZMQ topic filtering
and minimum size guards. NEVER runs ML during IBD.

This module MUST be used before any ML inference to ensure:
1. Node is fully synced (not in IBD)
2. ZMQ message is a valid rawtx (not block data)
3. Transaction meets minimum size requirements
"""

import logging
import time
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("cp1.consensus_guard")

# Minimum valid Bitcoin transaction size
# A minimal P2PKH tx is ~85 bytes, segwit can be ~60 bytes
MIN_TX_SIZE_BYTES = 60

# Maximum practical transaction size (policy limit)
MAX_TX_SIZE_BYTES = 400_000


@dataclass
class GuardResult:
    """Result of consensus guard check."""
    should_process: bool
    reason: str
    raw_hex: Optional[str] = None


class IBDGuard:
    """
    IBD-aware guard that prevents ML processing during initial sync.
    
    Usage:
        guard = IBDGuard(rpc)
        
        # In ZMQ loop
        result = guard.check_transaction(raw_bytes, topic)
        if not result.should_process:
            logger.debug(f"Skipping: {result.reason}")
            continue
        
        # Safe to process
        decoded = rpc.decoderawtransaction(result.raw_hex)
    """
    
    def __init__(self, rpc, check_interval: int = 30):
        """
        Initialize IBD guard.
        
        Args:
            rpc: Bitcoin RPC client
            check_interval: Seconds between IBD status checks
        """
        self.rpc = rpc
        self.check_interval = check_interval
        self._is_ibd = True
        self._last_check = 0
        self._sync_progress = 0.0
        
        # Initial check
        self._update_ibd_status()
    
    def _update_ibd_status(self):
        """Update IBD status from Bitcoin Core."""
        try:
            info = self.rpc.getblockchaininfo()
            self._is_ibd = info.get("initialblockdownload", True)
            self._sync_progress = info.get("verificationprogress", 0.0)
            self._last_check = time.time()
            
            if self._is_ibd:
                logger.info(f"IBD active: sync progress = {self._sync_progress:.2%}")
            else:
                logger.info("IBD complete: node fully synced")
                
        except Exception as e:
            logger.warning(f"Failed to check IBD status: {e}")
            # Assume still in IBD if check fails
            self._is_ibd = True
    
    @property
    def is_ibd(self) -> bool:
        """Check if node is in Initial Block Download."""
        if time.time() - self._last_check > self.check_interval:
            self._update_ibd_status()
        return self._is_ibd
    
    @property
    def sync_progress(self) -> float:
        """Get current sync progress (0.0 to 1.0)."""
        return self._sync_progress
    
    def check_zmq_topic(self, topic: str) -> bool:
        """
        Check if ZMQ topic should be processed.
        
        Only process 'rawtx' topic. Ignore:
        - rawblock (block data)
        - hashblock (block hashes)
        - sequence (internal)
        
        Args:
            topic: ZMQ topic string
            
        Returns:
            True if topic should be processed
        """
        valid_topics = {"rawtx"}
        return topic in valid_topics
    
    def check_tx_size(self, raw_bytes: bytes) -> Tuple[bool, str]:
        """
        Check if raw bytes meet size requirements.
        
        Args:
            raw_bytes: Raw transaction bytes
            
        Returns:
            (is_valid, reason)
        """
        size = len(raw_bytes)
        
        if size < MIN_TX_SIZE_BYTES:
            return False, f"tx_too_small ({size} < {MIN_TX_SIZE_BYTES})"
        
        if size > MAX_TX_SIZE_BYTES:
            return False, f"tx_too_large ({size} > {MAX_TX_SIZE_BYTES})"
        
        return True, "size_ok"
    
    def check_transaction(
        self, 
        raw_bytes: bytes, 
        topic: str = "rawtx"
    ) -> GuardResult:
        """
        Full guard check for a transaction.
        
        Args:
            raw_bytes: Raw transaction bytes from ZMQ
            topic: ZMQ topic
            
        Returns:
            GuardResult with should_process flag and reason
        """
        # Check 1: IBD status
        if self.is_ibd:
            return GuardResult(
                should_process=False,
                reason=f"ibd_active (progress={self._sync_progress:.2%})"
            )
        
        # Check 2: ZMQ topic
        if not self.check_zmq_topic(topic):
            return GuardResult(
                should_process=False,
                reason=f"invalid_topic ({topic})"
            )
        
        # Check 3: Size requirements
        size_ok, size_reason = self.check_tx_size(raw_bytes)
        if not size_ok:
            return GuardResult(
                should_process=False,
                reason=size_reason
            )
        
        # All checks passed
        return GuardResult(
            should_process=True,
            reason="all_checks_passed",
            raw_hex=raw_bytes.hex()
        )


def create_ibd_guard(rpc_url: str = None) -> IBDGuard:
    """
    Factory function to create IBD guard.
    
    Args:
        rpc_url: Bitcoin RPC URL (uses env var if not provided)
        
    Returns:
        IBDGuard instance
    """
    import os
    from bitcoinrpc.authproxy import AuthServiceProxy
    
    url = rpc_url or os.environ.get(
        "BITCOIN_RPC_URL", 
        "http://rpcuser:rpcpass@127.0.0.1:8332"
    )
    
    rpc = AuthServiceProxy(url)
    return IBDGuard(rpc)


# =============================================================================
# ZMQ LISTENER WITH GUARDS
# =============================================================================

class SafeZMQListener:
    """
    ZMQ listener that only processes transactions when safe.
    
    Usage:
        listener = SafeZMQListener(rpc, zmq_addr)
        
        for result in listener.listen():
            if result.should_process:
                process_transaction(result.raw_hex)
    """
    
    def __init__(
        self, 
        rpc, 
        zmq_addr: str = "tcp://127.0.0.1:28332",
        timeout_ms: int = 1000
    ):
        import zmq
        
        self.guard = IBDGuard(rpc)
        self.zmq_addr = zmq_addr
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(zmq_addr)
        
        # ONLY subscribe to rawtx topic
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "rawtx")
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        
        logger.info(f"SafeZMQListener connected to {zmq_addr}")
        logger.info("Subscribed to topic: rawtx only")
    
    def listen(self):
        """
        Generator that yields GuardResult for each ZMQ message.
        Only yields when safe to process.
        """
        import zmq
        
        while True:
            try:
                # Receive multipart message: [topic, body, seq]
                topic = self.socket.recv_string()
                body = self.socket.recv()
                
                # Optional sequence number (if enabled)
                try:
                    seq = self.socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    seq = None
                
                # Run all guards
                result = self.guard.check_transaction(body, topic)
                yield result
                
            except zmq.Again:
                # Timeout - no message
                continue
            except Exception as e:
                logger.error(f"ZMQ error: {e}")
                continue
    
    def close(self):
        """Close ZMQ connections."""
        self.socket.close()
        self.context.term()


# =============================================================================
# METRICS
# =============================================================================

class GuardMetrics:
    """Prometheus metrics for guard decisions."""
    
    def __init__(self, metrics_exporter=None):
        self.exporter = metrics_exporter
        self._counts = {
            "processed": 0,
            "skipped_ibd": 0,
            "skipped_topic": 0,
            "skipped_size": 0,
        }
    
    def record(self, result: GuardResult):
        """Record guard decision to metrics."""
        if result.should_process:
            self._counts["processed"] += 1
        elif "ibd" in result.reason:
            self._counts["skipped_ibd"] += 1
        elif "topic" in result.reason:
            self._counts["skipped_topic"] += 1
        elif "size" in result.reason:
            self._counts["skipped_size"] += 1
        
        # Update Prometheus if available
        if self.exporter:
            try:
                if result.should_process:
                    self.exporter.increment("cp1_guard_passed_total")
                else:
                    self.exporter.increment(
                        "cp1_guard_skipped_total", 
                        {"reason": result.reason.split("(")[0].strip()}
                    )
            except:
                pass
    
    def get_stats(self) -> dict:
        """Get guard statistics."""
        return dict(self._counts)
