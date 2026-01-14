# CP1-CP2 Unified Bitcoin Security System

## Master Documentation

> **A Real-Time Machine Learning Security System for Bitcoin Transaction Classification and Peer Network Monitoring**

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Technologies Used](#technologies-used)
4. [Component Deep Dive](#component-deep-dive)
5. [Data Flow](#data-flow)
6. [API Reference](#api-reference)
7. [Metrics & Monitoring](#metrics--monitoring)

---

## System Overview

### What This System Does

The CP1-CP2 Unified Security System is a **real-time Bitcoin transaction analysis and peer network monitoring platform** that:

1. **Monitors Bitcoin Transactions (CP1)**
   - Receives raw transactions via ZeroMQ from Bitcoin Core
   - Validates transactions against Bitcoin consensus rules
   - Classifies transactions using XGBoost machine learning
   - Makes ACCEPT/FLAG/REJECT decisions
   - Provides SHAP-based explainability for flagged transactions

2. **Monitors Peer Network (CP2)**
   - Tracks all connected Bitcoin peers
   - Detects Sybil attacks (many peers from same subnet)
   - Detects Eclipse attacks (peer isolation attempts)
   - Publishes risk scores for suspicious peers

3. **Provides Live Monitoring**
   - Prometheus metrics for all operations
   - Grafana dashboards for visualization
   - Public API for external integrations

### Key Features

| Feature | Description |
|---------|-------------|
| **Real-time Processing** | Sub-100ms transaction classification |
| **Consensus-First** | Bitcoin Core validation before any ML |
| **Shadow Mode** | Safe deployment without blocking transactions |
| **Explainable AI** | SHAP explanations for every decision |
| **Two-Tier Protection** | Transaction + Network layer security |
| **Public API** | REST API for external web apps |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BITCOIN NETWORK                                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BITCOIN CORE (regtest/mainnet)                       │
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐ │
│   │  RPC :18443 │    │ ZMQ :28332  │    │ P2P Network Connections         │ │
│   └──────┬──────┘    └──────┬──────┘    └────────────────┬────────────────┘ │
└──────────┼──────────────────┼────────────────────────────┼──────────────────┘
           │                  │                            │
           │                  │                            │
           ▼                  ▼                            ▼
┌────────────────────────────────────────┐   ┌────────────────────────────────┐
│          CP1 - TRANSACTION ANALYSIS    │   │    CP2 - PEER MONITORING       │
│                                        │   │                                │
│  ┌──────────────────────────────────┐  │   │  ┌──────────────────────────┐  │
│  │   1. Consensus Validator         │  │   │  │  1. RPC Peer Polling     │  │
│  │      (Bitcoin Core validation)   │  │   │  │     (getpeerinfo)        │  │
│  └───────────────┬──────────────────┘  │   │  └─────────────┬────────────┘  │
│                  ▼                     │   │                ▼               │
│  ┌──────────────────────────────────┐  │   │  ┌──────────────────────────┐  │
│  │   2. Feature Extraction          │  │   │  │  2. Sybil Detection      │  │
│  │      (tx properties → features)  │  │   │  │     (subnet clustering) │  │
│  └───────────────┬──────────────────┘  │   │  └─────────────┬────────────┘  │
│                  ▼                     │   │                ▼               │
│  ┌──────────────────────────────────┐  │   │  ┌──────────────────────────┐  │
│  │   3. XGBoost ML Inference        │  │   │  │  3. Eclipse Detection    │  │
│  │      (score: 0.0 - 1.0)          │  │   │  │     (low peer count)     │  │
│  └───────────────┬──────────────────┘  │   │  └─────────────┬────────────┘  │
│                  ▼                     │   │                ▼               │
│  ┌──────────────────────────────────┐  │   │  ┌──────────────────────────┐  │
│  │   4. Safe Action Policy          │  │   │  │  4. Alert Publishing     │  │
│  │      (ACCEPT/FLAG/REJECT)        │  │   │  │     (Redis pub/sub)      │  │
│  └───────────────┬──────────────────┘  │   │  └──────────────────────────┘  │
│                  ▼                     │   │                                │
│  ┌──────────────────────────────────┐  │   │                                │
│  │   5. SHAP Explainer              │  │   │                                │
│  │      (why flagged?)              │  │   │                                │
│  └──────────────────────────────────┘  │   │                                │
│                                        │   │                                │
│  Metrics: :8000                        │   │  Metrics: :8001                │
└─────────────────┬──────────────────────┘   └───────────────┬────────────────┘
                  │                                          │
                  └──────────────┬───────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REDIS :6379                                     │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐   │
│  │  Feature Cache          │  │  Pub/Sub Channels                       │   │
│  │  (decoded tx, address)  │  │  cp1:alerts, cp2:alerts                 │   │
│  └─────────────────────────┘  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROMETHEUS :9090                                   │
│  Scrapes metrics from CP1 (:8000), CP2 (:8001), Public API (:5000)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────┬────────────────────────────────────────────┐
│       GRAFANA :3000            │          PUBLIC API :5000                   │
│  ┌──────────────────────────┐  │  ┌──────────────────────────────────────┐  │
│  │  Unified Dashboard       │  │  │  User Registration & Auth            │  │
│  │  - TX Processing Rate    │  │  │  - POST /api/register                │  │
│  │  - ACCEPT/FLAG/REJECT    │  │  │  - POST /api/login                   │  │
│  │  - Peer Connections      │  │  │  - GET /api/apikey                   │  │
│  │  - Sybil/Eclipse Scores  │  │  ├──────────────────────────────────────┤  │
│  │  - Inference Latency     │  │  │  Metrics Endpoints                   │  │
│  └──────────────────────────┘  │  │  - GET /api/metrics                  │  │
│                                │  │  - GET /api/metrics/summary          │  │
│                                │  │  - GET /api/metrics/transactions     │  │
│                                │  │  - GET /api/metrics/security         │  │
│                                │  │  - GET /api/metrics/history          │  │
│                                │  └──────────────────────────────────────┘  │
└────────────────────────────────┴────────────────────────────────────────────┘
```

---

## Technologies Used

### Core Technologies

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **Python 3.10** | Primary language | Rich ML ecosystem, Bitcoin RPC libraries |
| **Docker** | Containerization | Consistent deployment, isolation |
| **Docker Compose** | Orchestration | Multi-container management |

### Bitcoin Infrastructure

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **Bitcoin Core** | Full node | Official reference implementation |
| **ZeroMQ (ZMQ)** | Real-time tx stream | Low-latency pub/sub from Bitcoin Core |
| **python-bitcoinrpc** | RPC client | Reliable RPC communication |

### Machine Learning

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **XGBoost** | Classification model | Fast, accurate, production-proven |
| **SHAP** | Explainability | Model-agnostic, trustworthy explanations |
| **Pandas** | Feature processing | DataFrame operations |
| **Joblib** | Model serialization | Efficient model loading |

### Data & Messaging

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **Redis** | Message bus & cache | Fast pub/sub, feature caching |
| **SQLite** | API user storage | Lightweight, no separate server |

### Monitoring & Visualization

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **Prometheus** | Metrics collection | Industry standard, time-series DB |
| **Grafana** | Dashboards | Beautiful, flexible visualization |
| **prometheus_client** | Python metrics | Official Prometheus library |

### API & Web

| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **FastAPI** | REST API | Fast, async, auto-documentation |
| **Uvicorn** | ASGI server | High-performance Python server |
| **PyJWT** | Authentication | JWT token generation/validation |
| **httpx** | HTTP client | Async HTTP requests to Prometheus |

---

## Component Deep Dive

### CP1: Transaction Classification Runtime

**File:** `code/cp1_production_runtime.py`

The brain of transaction analysis. Processes every transaction through a multi-stage pipeline:

```python
# Processing Flow
1. ZMQ Listener → Receives raw tx hex
2. Consensus Validator → Bitcoin Core validation (MUST PASS)
3. Feature Extractor → Convert tx to ML features
4. XGBoost Model → Predict score (0.0 = safe, 1.0 = illicit)
5. Safe Action Policy → Map score to ACCEPT/FLAG/REJECT
6. SHAP Explainer → Generate explanation for flagged tx
7. Metrics → Record everything to Prometheus
```

**Key Classes:**
- `RuntimeConfig` - Environment configuration
- `CP1ProductionRuntime` - Main processing loop
- `extract_features()` - Transaction → Features mapping

### Consensus Validator

**File:** `code/consensus_validator.py`

**Critical Safety Gate** - No transaction is analyzed by ML unless it passes Bitcoin consensus rules.

```python
# Validation Steps
1. decoderawtransaction → Parse tx structure
2. Validate inputs/outputs exist
3. testmempoolaccept → Full script/signature validation
```

**Why Consensus First?**
- Prevents ML from incorrectly "accepting" malformed transactions
- Ensures only valid Bitcoin transactions reach inference
- Deterministic validation (no ML uncertainty)

### Safe Action Policy

**File:** `code/safe_action_policy.py`

Conservative decision engine that prioritizes safety:

```python
# Thresholds
ACCEPT: score < 0.12  (12% illicit probability → ACCEPT)
FLAG:   0.12 <= score < 0.65  (uncertain → flag for review)
REJECT: score >= 0.65  (high confidence illicit → quarantine)

# Shadow Mode
When enabled (default), REJECT only logs + alerts, 
never actually blocks transactions.
```

**Philosophy:** False positives (blocking legitimate tx) are worse than false negatives.

### SHAP Explainer

**File:** `code/shap_explainer.py`

Provides human-readable explanations:

```json
{
  "txid": "abc123...",
  "score": 0.78,
  "decision": "REJECT",
  "top_features": [
    {"feature": "output_count", "contribution": 0.15, "direction": "increase"},
    {"feature": "total_value", "contribution": 0.12, "direction": "increase"}
  ],
  "human_reason": "Flagged due to: high output_count (+0.15), high total_value (+0.12)"
}
```

### CP2: Peer Security Extractor

**File:** `cp2_integration/cp2_peer_extractor.py`

Monitors Bitcoin Core peer connections for network-layer attacks:

**Sybil Detection:**
```python
# If too many peers from same /24 subnet
subnet_counts = count_peers_by_subnet(peers)
if any subnet has > 5 peers:
    sybil_risk = 1.0
    publish_alert("sybil", subnet, peer_list)
```

**Eclipse Detection:**
```python
# If peer count drops too low
if total_peers < 3:
    eclipse_risk = 1.0
    publish_alert("eclipse", "low peer count")
```

### Redis Client

**File:** `code/redis_client.py`

Handles all inter-component communication:

```python
# CP1 → CP2: Transaction alerts
redis.publish("cp1:alerts", {
    "txid": "...",
    "score": 0.78,
    "decision": "REJECT",
    "announcing_peer": "192.168.1.50:8333"
})

# CP2 → CP1: Peer risk lookup
risk = redis.get("cp2:peer_risk:192.168.1.50")
if risk > 0.5:
    # Treat transactions from this peer with extra scrutiny
```

### Metrics Exporter

**File:** `code/metrics_exporter.py`

Prometheus metrics for everything:

```python
# Counters
cp1_tx_ingested_total      # Total transactions processed
cp1_accept_count_total     # ACCEPT decisions
cp1_flag_count_total       # FLAG decisions  
cp1_reject_count_total     # REJECT decisions
cp1_consensus_passed_total # Passed consensus validation
cp1_consensus_failed_total # Failed consensus validation

# Histograms
cp1_inference_latency_seconds  # ML inference time
cp1_score_distribution         # Score distribution

# Gauges (from CP2)
cp2_peer_connections           # Current peer count
cp2_sybil_risk_score          # Sybil attack risk
cp2_eclipse_risk_score        # Eclipse attack risk
```

### Public API

**File:** `website/api.py`

REST API for external applications:

```
POST /api/register       → Create user account
POST /api/login          → Get JWT token
GET  /api/apikey         → Generate API key
GET  /api/metrics        → All live metrics (requires API key)
GET  /api/metrics/summary    → Summary only
GET  /api/metrics/history    → Time-series data
```

---

## Data Flow

### Transaction Processing Flow

```
1. Bitcoin Core receives tx from network
          ↓
2. ZMQ publishes to tcp://localhost:28332
          ↓
3. CP1 ZMQ Listener receives raw hex
          ↓
4. Consensus Validator checks with Bitcoin Core RPC
   ├── INVALID → Log, increment consensus_failed, STOP
   └── VALID → Continue
          ↓
5. Feature Extractor converts to ML features
   (output_count, total_value, input_count, etc.)
          ↓
6. XGBoost model.predict(features) → score
          ↓
7. Safe Action Policy evaluates score
   ├── score < 0.12 → ACCEPT (increment accept_count)
   ├── 0.12 <= score < 0.65 → FLAG (generate SHAP explanation)
   └── score >= 0.65 → REJECT (quarantine, alert)
          ↓
8. Metrics recorded to Prometheus
          ↓
9. If FLAG/REJECT: Publish to Redis cp1:alerts
```

### Peer Monitoring Flow

```
1. CP2 polls Bitcoin Core every 10 seconds
   (getpeerinfo RPC call)
          ↓
2. Extract peer info (addr, version, subnet, etc.)
          ↓
3. Sybil Detection: Count peers per /24 subnet
   ├── > 5 peers from same subnet → Sybil alert
   └── Normal → Continue
          ↓
4. Eclipse Detection: Check total peer count
   ├── < 3 peers → Eclipse alert
   └── Normal → Continue
          ↓
5. Update Prometheus gauges (peer_count, risk_scores)
          ↓
6. Publish alerts to Redis cp2:alerts
```

---

## API Reference

### Authentication Endpoints

#### Register
```http
POST /api/register
Content-Type: application/json

{"username": "myuser", "email": "user@example.com", "password": "secret123"}
```

Response:
```json
{"access_token": "eyJhbGciOiJIUzI1NiIs...", "token_type": "bearer", "expires_in": 86400}
```

#### Login
```http
POST /api/login
Content-Type: application/json

{"username": "myuser", "password": "secret123"}
```

#### Get API Key
```http
GET /api/apikey
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

Response:
```json
{"api_key": "cpsk_9mjoGQccbNavu0DReGkIHlXiupDMYgri4kMLAI_VODA", "created_at": "2026-01-12T13:49:00"}
```

### Metrics Endpoints (Require X-API-Key header)

#### All Metrics
```http
GET /api/metrics
X-API-Key: cpsk_9mjoGQccbNavu0DReGkIHlXiupDMYgri4kMLAI_VODA
```

Response:
```json
{
  "summary": {
    "total_transactions": 1250,
    "accepted": 1180,
    "flagged": 45,
    "rejected": 25,
    "accept_rate": 94.4,
    "flag_rate": 3.6,
    "reject_rate": 2.0,
    "last_updated": "2026-01-12T13:50:00"
  },
  "transactions": {...},
  "security": {...},
  "raw_metrics": {...}
}
```

#### Historical Data
```http
GET /api/metrics/history?metric=cp1_tx_ingested_total&hours=1
X-API-Key: cpsk_...
```

---

## Metrics & Monitoring

### Grafana Dashboard

Access at: **http://localhost:3000** (admin/admin)

**Dashboard Panels:**

| Panel | Description |
|-------|-------------|
| CP1: Total TX | Total transactions processed |
| CP1: ACCEPT | Transactions accepted |
| CP1: FLAG | Transactions flagged for review |
| CP1: REJECT | Transactions quarantined |
| CP2: Peers | Connected peer count |
| CP2: Sybil Score | Sybil attack risk (0-1) |
| TX Processing Rate | Transactions per second over time |
| Peer Connections | Inbound/outbound peers over time |
| Attack Detection Scores | Sybil/Eclipse scores over time |
| Inference Latency | P50/P95/P99 ML inference time |

### Prometheus Queries

```promql
# Transaction rate per minute
rate(cp1_tx_ingested_total[1m])

# Accept rate percentage
cp1_accept_count_total / cp1_tx_ingested_total * 100

# Average inference latency
histogram_quantile(0.95, rate(cp1_infer_latency_seconds_bucket[5m]))

# Sybil risk over time
cp2_sybil_score
```

---

## File Structure

```
cp1_projects/
├── code/                          # Core Python modules
│   ├── cp1_production_runtime.py  # Main CP1 runtime
│   ├── consensus_validator.py     # Bitcoin Core validation
│   ├── safe_action_policy.py      # ACCEPT/FLAG/REJECT logic
│   ├── shap_explainer.py          # ML explainability
│   ├── metrics_exporter.py        # Prometheus metrics
│   ├── redis_client.py            # Redis communication
│   ├── feature_cache.py           # Feature caching
│   └── utxo_address_cache.py      # UTXO/address lookup
│
├── cp2_integration/               # Peer monitoring
│   └── cp2_peer_extractor.py      # Sybil/Eclipse detection
│
├── docker/                        # Docker configuration
│   ├── docker-compose.unified.yml # Main orchestration
│   ├── Dockerfile.cp1             # CP1 container
│   ├── Dockerfile.cp2             # CP2 container
│   └── configs/
│       ├── prometheus.unified.yml # Prometheus config
│       ├── alert_rules.unified.yml# Alerting rules
│       └── grafana/
│           ├── dashboards/        # Dashboard JSON files
│           └── provisioning/      # Auto-provisioning
│
├── website/                       # Public API
│   ├── api.py                     # FastAPI application
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # API container
│
├── models/                        # ML models
│   └── cp1_static_xgb_v1.joblib   # Trained XGBoost model
│
├── datasets/                      # Training data
│   └── elliptic++/                # Elliptic++ dataset
│
└── tests/                         # Test suite
    └── *.py                       # Unit/integration tests
```

---

## Security Considerations

1. **Shadow Mode**: Default deployment never blocks transactions
2. **Consensus First**: ML can't override Bitcoin Core validation
3. **Conservative Thresholds**: High bar for REJECT (65%)
4. **API Key Scoping**: Each user has unique API key
5. **JWT Expiration**: Tokens expire after 24 hours
6. **Password Hashing**: SHA-256 hashed (should upgrade to bcrypt)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-12 | Initial unified system |

---

**Documentation Last Updated:** 2026-01-12
