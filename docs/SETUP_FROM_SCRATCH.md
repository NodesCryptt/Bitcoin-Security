# How to Run Everything From Scratch

This guide covers the complete setup process to deploy the entire CP1-CP2 Unified Security System from a fresh state.

## Prerequisites

Ensure you have the following installed:
1. **Docker Desktop** (running and updated)
2. **Git**
3. **Python 3.10+** (optional, for local development)
4. **Curl** (for testing API)

---

## Step 1: Clone and Prepare

1. Clone the repository (if you haven't already):
   ```bash
   git clone <repository-url>
   cd cp1_projects
   ```

2. Verify the directory structure exists:
   ```bash
   ls -R docker/ website/ code/
   ```

## Step 2: Configure Environment

We use a unified docker-compose file that orchestrates all services. No manual `.env` file creation is strictly necessary for the default setup as defaults are baked into the Dockerfile and compose file, but for production, you should set:

**Environment Variables (Optional overrides):**
- `CP1_RPC_PASSWORD`: Bitcoin Core RPC password
- `SECRET_KEY`: Secret key for API JWT tokens

## Step 3: Build and Start Services

Navigate to the docker directory and start the unified stack:

```bash
cd docker
docker-compose -f docker-compose.unified.yml up -d --build
```

**What this does:**
1. Builds the `cp1` container (Transaction Classifier)
2. Builds the `cp2` container (Peer Monitor)
3. Builds the `api` container (Public API)
4. Pulls standard images: `bitcoin-core`, `redis`, `prometheus`, `grafana`
5. Starts all services in the correct dependency order

## Step 4: Verify Services are Running

Check the status of all containers:

```bash
docker-compose -f docker-compose.unified.yml ps
```

You should see 8 containers running:
- `cp_bitcoind`: Bitcoin Core Node
- `cp_redis`: Redis Message Bus
- `cp_tx_generator`: Transaction Generator (creates traffic)
- `cp1`: Transaction Classifier Runtime
- `cp2`: Peer Security Monitor
- `prometheus`: Metrics Collector
- `grafana`: Visualization Dashboard
- `api`: Public Metrics API

## Step 5: Verify System Functionality

### 1. Check Bitcoin Core
Ensure Bitcoin Core is producing blocks (for Regtest):
```bash
docker exec cp_bitcoind bitcoin-cli -regtest -rpcuser=cp1user -rpcpassword=CP1SecurePassword123! getblockchaininfo
```
*Expected Output:* Blocks should be increasing.

### 2. Check Metrics Generation
Verify that CP1 is processing transactions:
```bash
curl http://localhost:8000/metrics | grep cp1_tx_ingested_total
```
*Expected Output:* Counter > 0

### 3. Check Public API
Verify the API is reachable:
```bash
curl http://localhost:5000/health
```
*Expected Output:* `{"status": "healthy", ...}`

## Step 6: Accessing the System

### 1. Grafana Dashboards (Visual monitoring)
- **URL:** [http://localhost:3000](http://localhost:3000)
- **Login:** `admin` / `admin` (skip password change if asked)
- **Dashboard:** Navigate to **"CP1-CP2 Production Dashboard"**
- **What to look for:**
  - "Total TX" should successfully increment.
  - "ACCEPT/FLAG/REJECT" panels should show counts.
  - "Peer Connections" should show connected peers.

### 2. Public API (Integration)
- **Documentation:** [http://localhost:5000/docs](http://localhost:5000/docs) (Swagger UI)
- **Register a User:**
  ```bash
  curl -X POST http://localhost:5000/api/register \
    -H "Content-Type: application/json" \
    -d '{
      "username": "admin",
      "email": "admin@example.com",
      "password": "securepassword"
    }'
  ```
- **Get API Key:**
  Login and fetch key as described in the [API Documentation](../website/WALKTHROUGH.md).

### 3. Prometheus (Raw Metrics)
- **URL:** [http://localhost:9090](http://localhost:9090)
- **Useful Queries:**
  - `rate(cp1_tx_ingested_total[1m])`
  - `cp2_peer_count`

## Troubleshooting Common Issues

### "Container name already in use"
If you get errors about conflicting container names:
```bash
docker-compose -f docker-compose.unified.yml down --remove-orphans
docker-compose -f docker-compose.unified.yml up -d
```

### "No Data" in Grafana
1. Check if the Transaction Generator is running: `docker logs cp_tx_generator`
2. Ensure Prometheus is scraping: Check targets at [http://localhost:9090/targets](http://localhost:9090/targets)
3. Ensure CP1 is running without errors: `docker logs cp_tx_classifier`

### "Connection Refused" to API
Ensure the container started successfully and mapped port 5000:
```bash
docker logs cp_public_api
```

## Useful Commands Cheat Sheet

| Action | Command |
|--------|---------|
| Start All | `docker-compose -f docker-compose.unified.yml up -d` |
| Stop All | `docker-compose -f docker-compose.unified.yml down` |
| View Logs (CP1) | `docker logs -f cp_tx_classifier` |
| View Logs (API) | `docker logs -f cp_public_api` |
| Restart Service | `docker-compose -f docker-compose.unified.yml restart <service_name>` |
| Rebuild Service | `docker-compose -f docker-compose.unified.yml up -d --build <service_name>` |

---

**You are now fully operational!** 🚀
