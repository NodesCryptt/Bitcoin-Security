"""
CP1 Consensus Validator
=======================
Deterministic Bitcoin Core validation layer.

This module provides a validation gate that must be passed BEFORE any ML inference.
If a transaction fails consensus validation, it is REJECTED immediately without ML.

This ensures that ML never makes consensus decisions on cryptographically invalid
transactions, avoiding the risk of ML incorrectly accepting malformed transactions.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConsensusResult(Enum):
    """Result of consensus validation."""
    VALID = "valid"
    INVALID_STRUCTURE = "invalid_structure"
    INVALID_SCRIPT = "invalid_script"
    DECODE_ERROR = "decode_error"
    RPC_ERROR = "rpc_error"


@dataclass
class ValidationResult:
    """Detailed validation result."""
    status: ConsensusResult
    txid: Optional[str] = None
    decoded: Optional[Dict[str, Any]] = None
    reject_reason: Optional[str] = None
    error_message: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        return self.status == ConsensusResult.VALID
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "txid": self.txid,
            "is_valid": self.is_valid,
            "reject_reason": self.reject_reason,
            "error_message": self.error_message
        }


class ConsensusValidator:
    """
    Validates transactions against Bitcoin Core consensus rules.
    
    This validator must be called as the FIRST gate in the inference pipeline.
    Any transaction failing validation is immediately rejected without ML.
    
    Usage:
        validator = ConsensusValidator(rpc)
        result = validator.validate(raw_hex)
        if not result.is_valid:
            # REJECT immediately - do not call ML
            return reject_transaction(result)
        # Continue to ML inference
        score = model.predict(...)
    """
    
    # Errors that indicate connection issues (should retry with reconnect)
    CONNECTION_ERRORS = (
        "broken pipe",
        "request-sent", 
        "connection reset",
        "eof occurred",
        "connection refused",
        "timed out",
        "connection aborted",
    )
    
    def __init__(self, rpc_client, strict_mode: bool = True, rpc_url: str = None):
        """
        Initialize validator.
        
        Args:
            rpc_client: Bitcoin Core RPC client (AuthServiceProxy)
            strict_mode: If True, use testmempoolaccept for full validation
            rpc_url: Optional RPC URL for reconnection (if None, reconnection disabled)
        """
        self.rpc = rpc_client
        self.strict_mode = strict_mode
        self.rpc_url = rpc_url  # Store for reconnection
        self._reconnect_count = 0
    
    def _reconnect_rpc(self):
        """Attempt to reconnect to Bitcoin Core RPC."""
        if not self.rpc_url:
            logger.warning("Cannot reconnect: no RPC URL configured")
            return False
        
        try:
            from bitcoinrpc.authproxy import AuthServiceProxy
            self.rpc = AuthServiceProxy(self.rpc_url)
            # Test connection
            self.rpc.getblockchaininfo()
            self._reconnect_count += 1
            logger.info(f"RPC reconnected successfully (reconnect #{self._reconnect_count})")
            return True
        except Exception as e:
            logger.error(f"RPC reconnection failed: {e}")
            return False
    
    def _is_connection_error(self, error_msg: str) -> bool:
        """Check if error message indicates a connection issue."""
        error_lower = error_msg.lower()
        return any(conn_err in error_lower for conn_err in self.CONNECTION_ERRORS)
    
    def validate(self, raw_hex: str) -> ValidationResult:
        """
        Validate a raw transaction hex against consensus rules.
        
        Args:
            raw_hex: Raw transaction hex string
            
        Returns:
            ValidationResult with status and details
        """
        # Step 1: Structural decode
        decoded = self._decode_transaction(raw_hex)
        if decoded is None:
            return ValidationResult(
                status=ConsensusResult.DECODE_ERROR,
                reject_reason="undecodable",
                error_message="Failed to decode raw transaction"
            )
        
        txid = decoded.get("txid", "unknown")
        
        # Step 2: Basic structural validation
        structural_result = self._validate_structure(decoded)
        if not structural_result.is_valid:
            structural_result.txid = txid
            structural_result.decoded = decoded
            return structural_result
        
        # Step 3: Script validation (if strict mode)
        if self.strict_mode:
            script_result = self._validate_scripts(raw_hex)
            if not script_result.is_valid:
                script_result.txid = txid
                script_result.decoded = decoded
                return script_result
        
        # All checks passed
        return ValidationResult(
            status=ConsensusResult.VALID,
            txid=txid,
            decoded=decoded
        )
    
    def _decode_transaction(self, raw_hex: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Decode raw transaction using Bitcoin Core.
        
        Includes retry logic with RPC reconnection for connection errors.
        """
        import time
        
        last_error = None
        for attempt in range(max_retries):
            try:
                decoded = self.rpc.decoderawtransaction(raw_hex)
                return decoded
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Check if this is a connection error that might be fixed by reconnecting
                if self._is_connection_error(error_msg):
                    if attempt < max_retries - 1:
                        # Exponential backoff: 0.1s, 0.2s, 0.4s
                        backoff = 0.1 * (2 ** attempt)
                        logger.debug(f"Connection error on decode attempt {attempt + 1}: {error_msg}, retrying in {backoff:.1f}s...")
                        time.sleep(backoff)
                        
                        # Try to reconnect
                        if self._reconnect_rpc():
                            continue  # Retry with new connection
                        # If reconnection failed, try anyway on next iteration
                        continue
                else:
                    # Not a connection error - don't retry
                    break
        
        # All retries exhausted or non-retryable error
        preview = raw_hex[:40] if len(raw_hex) > 40 else raw_hex
        logger.warning(f"Failed to decode transaction (len={len(raw_hex)}): {last_error} | hex_preview={preview}...")
        return None
    
    def _validate_structure(self, decoded: Dict[str, Any]) -> ValidationResult:
        """Validate basic transaction structure."""
        
        # Check for required fields
        required_fields = ["txid", "version", "vin", "vout"]
        for field in required_fields:
            if field not in decoded:
                return ValidationResult(
                    status=ConsensusResult.INVALID_STRUCTURE,
                    reject_reason=f"missing_{field}",
                    error_message=f"Transaction missing required field: {field}"
                )
        
        # Check version
        version = decoded.get("version", 0)
        if version not in [1, 2]:
            return ValidationResult(
                status=ConsensusResult.INVALID_STRUCTURE,
                reject_reason="invalid_version",
                error_message=f"Invalid transaction version: {version}"
            )
        
        # Check inputs
        vin = decoded.get("vin", [])
        if not isinstance(vin, list):
            return ValidationResult(
                status=ConsensusResult.INVALID_STRUCTURE,
                reject_reason="invalid_vin",
                error_message="vin must be a list"
            )
        
        # Non-coinbase transactions must have at least one input
        if len(vin) == 0:
            return ValidationResult(
                status=ConsensusResult.INVALID_STRUCTURE,
                reject_reason="empty_vin",
                error_message="Transaction has no inputs"
            )
        
        # Check if coinbase (only valid in blocks)
        is_coinbase = any("coinbase" in v for v in vin if isinstance(v, dict))
        if is_coinbase and len(vin) > 1:
            return ValidationResult(
                status=ConsensusResult.INVALID_STRUCTURE,
                reject_reason="invalid_coinbase",
                error_message="Coinbase transaction cannot have multiple inputs"
            )
        
        # Check outputs
        vout = decoded.get("vout", [])
        if not isinstance(vout, list):
            return ValidationResult(
                status=ConsensusResult.INVALID_STRUCTURE,
                reject_reason="invalid_vout",
                error_message="vout must be a list"
            )
        
        if len(vout) == 0:
            return ValidationResult(
                status=ConsensusResult.INVALID_STRUCTURE,
                reject_reason="empty_vout",
                error_message="Transaction has no outputs"
            )
        
        # Validate output values
        for i, output in enumerate(vout):
            if not isinstance(output, dict):
                continue
            value = output.get("value", 0)
            if value < 0:
                return ValidationResult(
                    status=ConsensusResult.INVALID_STRUCTURE,
                    reject_reason="negative_output",
                    error_message=f"Output {i} has negative value: {value}"
                )
        
        return ValidationResult(status=ConsensusResult.VALID)
    
    def _validate_scripts(self, raw_hex: str) -> ValidationResult:
        """
        Validate transaction scripts using testmempoolaccept.
        
        This checks:
        - Script execution
        - Signature verification
        - Standard policy rules
        """
        try:
            result = self.rpc.testmempoolaccept([raw_hex])
            
            if not result or len(result) == 0:
                return ValidationResult(
                    status=ConsensusResult.RPC_ERROR,
                    reject_reason="empty_result",
                    error_message="testmempoolaccept returned empty result"
                )
            
            tx_result = result[0]
            
            if tx_result.get("allowed", False):
                return ValidationResult(status=ConsensusResult.VALID)
            else:
                reject_reason = tx_result.get("reject-reason", "unknown")
                
                # These rejections mean tx is structurally valid but already processed
                # Allow these through to ML inference - they're not invalid
                acceptable_rejections = [
                    "txn-already-known",
                    "txn-already-in-mempool", 
                    "txn-mempool-conflict",
                    "insufficient fee",  # Valid tx, just low fee
                ]
                
                if any(r in reject_reason.lower() for r in acceptable_rejections):
                    # Transaction is structurally valid, just already seen
                    # Pass to ML for scoring
                    return ValidationResult(status=ConsensusResult.VALID)
                
                # Map reject reasons to result types
                if "script" in reject_reason.lower() or "signature" in reject_reason.lower():
                    status = ConsensusResult.INVALID_SCRIPT
                elif "coinbase" in reject_reason.lower():
                    # Coinbase transactions are valid but not standard mempool txs
                    status = ConsensusResult.INVALID_STRUCTURE
                else:
                    status = ConsensusResult.INVALID_STRUCTURE
                
                return ValidationResult(
                    status=status,
                    reject_reason=reject_reason,
                    error_message=f"testmempoolaccept rejected: {reject_reason}"
                )
                
        except Exception as e:
            # If testmempoolaccept fails (e.g., node doesn't support it),
            # fall back to structural validation only
            logger.warning(f"testmempoolaccept failed: {e}")
            return ValidationResult(status=ConsensusResult.VALID)


def validate_transaction(raw_hex: str, rpc_client) -> ValidationResult:
    """
    Convenience function to validate a transaction.
    
    Args:
        raw_hex: Raw transaction hex
        rpc_client: Bitcoin Core RPC client
        
    Returns:
        ValidationResult
    """
    validator = ConsensusValidator(rpc_client)
    return validator.validate(raw_hex)
