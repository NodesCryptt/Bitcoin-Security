#!/usr/bin/env python
"""
CP1 Live Inference Runtime with Cache Integration
==================================================
Enhanced runtime with:
- UTXO/Address cache for enriched features
- SHAP explanation logging for FLAG/REJECT events
- Prometheus metrics
- Double-spend detection

Usage:
    python cp1_live_infer_runtime.py

Environment Variables:
    REDIS_URL: Redis URL for caching (default: redis://localhost:6379)
    ZMQ_RAWTX: ZMQ endpoint (default: tcp://127.0.0.1:28332)
    BITCOIN_RPC_URL: Bitcoin Core RPC URL
"""

import os
import zmq
import json
import time
import joblib
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cp1.live")

# Bitcoin RPC
try:
    from bitcoinrpc.authproxy import AuthServiceProxy
except ImportError:
    logger.error("python-bitcoinrpc not installed. Run: pip install python-bitcoinrpc")
    raise

# CP1 modules
from utxo_address_cache import UTXOAddressCache, enrich_features_with_cache
from shap_logging_wrapper import SHAPLoggingWrapper, create_shap_wrapper
from metrics_exporter import get_metrics


# =============================================================================
# CONFIGURATION
# =============================================================================

ZMQ_RAWTX = os.environ.get("ZMQ_RAWTX", "tcp://127.0.0.1:28332")
RPC_URL = os.environ.get("BITCOIN_RPC_URL", "http://cp1user:CP1SecurePassword123!@127.0.0.1:8332")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "cp1_static_xgb_v1.joblib"))
FEATURE_PATH = os.environ.get("FEATURE_PATH", str(BASE_DIR / "results" / "cp1_static_ellipticpp.csv"))

ACCEPT_T = float(os.environ.get("ACCEPT_THRESHOLD", "0.15"))
REJECT_T = float(os.environ.get("REJECT_THRESHOLD", "0.60"))

METRICS_PORT = int(os.environ.get("PROMETHEUS_PORT", "8000"))
ENABLE_SHAP = os.environ.get("ENABLE_SHAP", "true").lower() == "true"


# =============================================================================
# INITIALIZATION
# =============================================================================

logger.info("=" * 60)
logger.info("CP1 LIVE INFERENCE RUNTIME (Enhanced)")
logger.info("=" * 60)

# Load model
logger.info(f"Loading model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Load feature schema
logger.info(f"Loading feature schema: {FEATURE_PATH}")
feature_cols = (
    pd.read_csv(FEATURE_PATH, nrows=1)
    .drop(columns=["label"])
    .columns
    .tolist()
)
logger.info(f"Feature columns: {len(feature_cols)}")

# Initialize RPC
logger.info(f"Connecting to Bitcoin Core...")
rpc = AuthServiceProxy(RPC_URL)
try:
    info = rpc.getblockchaininfo()
    logger.info(f"Connected: chain={info['chain']}, blocks={info['blocks']}")
except Exception as e:
    logger.warning(f"RPC connection test failed (will retry on first tx): {e}")

# Initialize cache
logger.info(f"Initializing UTXO cache: {REDIS_URL}")
cache = UTXOAddressCache(redis_url=REDIS_URL)
logger.info(f"Cache Redis available: {cache.is_redis_available}")

# Initialize metrics
logger.info(f"Starting Prometheus metrics on port {METRICS_PORT}")
metrics = get_metrics(METRICS_PORT)
metrics.start_server()

# Initialize SHAP wrapper
shap_wrapper = None
if ENABLE_SHAP:
    logger.info("Initializing SHAP logging wrapper")
    shap_wrapper = create_shap_wrapper(
        model=model,
        feature_names=feature_cols,
        metrics=metrics
    )

# ZMQ setup
logger.info(f"Connecting to ZMQ: {ZMQ_RAWTX}")
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(ZMQ_RAWTX)
socket.setsockopt_string(zmq.SUBSCRIBE, "")
socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout

logger.info("-" * 60)
logger.info(f"Thresholds: ACCEPT < {ACCEPT_T} | FLAG {ACCEPT_T}-{REJECT_T} | REJECT >= {REJECT_T}")
logger.info("=" * 60)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_minimal_features(decoded):
    """Extract minimal features from decoded transaction."""
    f = {}
    
    try:
        f["size"] = float(decoded.get("size", 0))
    except (ValueError, TypeError):
        f["size"] = 0.0
    
    try:
        f["fees"] = float(decoded.get("fee", 0))
    except (ValueError, TypeError):
        f["fees"] = 0.0
    
    try:
        f["num_input_addresses"] = len(decoded.get("vin", []))
    except:
        f["num_input_addresses"] = 0
    
    try:
        f["num_output_addresses"] = len(decoded.get("vout", []))
    except:
        f["num_output_addresses"] = 0
    
    try:
        f["total_BTC"] = sum(float(v.get("value", 0)) for v in decoded.get("vout", []))
    except:
        f["total_BTC"] = 0.0
    
    return f


# =============================================================================
# MAIN LOOP
# =============================================================================

logger.info("Starting transaction processing loop...")

tx_count = 0
accept_count = 0
flag_count = 0
reject_count = 0

try:
    while True:
        try:
            raw = socket.recv()
            raw_hex = raw.hex()
            start_time = time.perf_counter()
            
            # Record ingestion
            metrics.record_tx_ingested()
            tx_count += 1
            
            # Decode TX
            try:
                decoded = rpc.decoderawtransaction(raw_hex)
            except Exception as e:
                logger.debug(f"Decode failed: {e}")
                continue
            
            txid = decoded.get("txid", "unknown")
            
            # Extract basic features
            feature_start = time.perf_counter()
            feats = extract_minimal_features(decoded)
            
            # Enrich with cache data
            feats = enrich_features_with_cache(feats, decoded, cache, rpc)
            feature_time = time.perf_counter() - feature_start
            metrics.observe("cp1_feature_extraction_seconds", feature_time)
            
            # Check double-spend
            if feats.get("is_double_spend", 0) > 0:
                metrics.record_double_spend()
                logger.warning(f"DOUBLE-SPEND DETECTED: {txid[:16]}... ({feats.get('num_double_spend_inputs', 0)} inputs)")
            
            # Build inference row
            row = pd.DataFrame([{c: feats.get(c, 0.0) for c in feature_cols}])
            row = row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            
            # Inference
            infer_start = time.perf_counter()
            score = float(model.predict_proba(row)[0, 1])
            infer_time = time.perf_counter() - infer_start
            
            metrics.record_latency(infer_time)
            metrics.record_score(score)
            
            # Decision
            if score < ACCEPT_T:
                decision = "ACCEPT"
                accept_count += 1
                metrics.record_decision("ACCEPT")
            elif score < REJECT_T:
                decision = "FLAG"
                flag_count += 1
                metrics.record_decision("FLAG")
            else:
                decision = "REJECT"
                reject_count += 1
                metrics.record_decision("REJECT", "ml")
            
            # SHAP explanation for FLAG/REJECT
            if decision != "ACCEPT" and shap_wrapper:
                processing_latency_ms = (time.perf_counter() - start_time) * 1000
                try:
                    explanation = shap_wrapper.explain_and_log(
                        features=row,
                        txid=txid,
                        score=score,
                        decision=decision,
                        processing_latency_ms=processing_latency_ms
                    )
                    logger.info(f"SHAP explanation saved: {explanation.human_reason[:80]}...")
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            
            # Record inputs as spent (for double-spend tracking)
            if decision == "ACCEPT":
                cache.record_tx_inputs(decoded)
            
            # Log
            total_time = (time.perf_counter() - start_time) * 1000
            log_level = logging.INFO if decision == "ACCEPT" else logging.WARNING
            logger.log(
                log_level,
                f"{decision} | txid={txid[:16]}... | score={score:.4f} | "
                f"size={feats['size']} | latency={total_time:.1f}ms"
            )
            
        except zmq.Again:
            # Timeout - no message
            continue
        
        except KeyboardInterrupt:
            break
        
        except Exception as e:
            logger.error(f"Processing error: {e}")
            continue

except KeyboardInterrupt:
    pass

finally:
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down CP1 runtime...")
    logger.info(f"Session stats: tx={tx_count}, accept={accept_count}, flag={flag_count}, reject={reject_count}")
    logger.info(f"Cache stats: {cache.get_stats()}")
    
    if shap_wrapper:
        shap_wrapper.shutdown()
    
    socket.close()
    context.term()
    
    logger.info("CP1 runtime stopped cleanly.")
