"""
CP Security Public API
=======================
A simple FastAPI service that provides authenticated access to Grafana/Prometheus metrics.

Users can:
1. Register an account
2. Login to get a JWT token
3. Generate an API key
4. Use the API key to fetch live metrics

Endpoints:
- POST /api/register - Create account
- POST /api/login - Get JWT token
- GET /api/apikey - Get/Generate API key
- GET /api/metrics - Fetch all metrics (public)
- GET /api/metrics/summary - Get summary stats (public)
- GET /api/metrics/transactions - Get transaction metrics (public)
- GET /api/metrics/security - Get security metrics (public)
- GET /api/metrics/history - Get historical data (public)
"""

import os
import secrets
import hashlib
import sqlite3
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/users.db")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

app = FastAPI(
    title="CP Security Metrics API",
    description="Public API to fetch live Grafana/Prometheus metrics for your web applications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - Allow all origins for public API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


# =============================================================================
# Pydantic Models
# =============================================================================

class UserRegister(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRY_HOURS * 3600


class APIKeyResponse(BaseModel):
    api_key: str
    created_at: str
    message: str


class MetricsSummary(BaseModel):
    total_transactions: float
    accepted: float
    flagged: float
    rejected: float
    accept_rate: float
    flag_rate: float
    reject_rate: float
    last_updated: str


class TransactionMetrics(BaseModel):
    total_ingested: float
    consensus_passed: float
    consensus_failed: float
    ml_inferences: float
    decisions: dict
    last_updated: str


class SecurityMetrics(BaseModel):
    connected_peers: float
    sybil_risk_score: float
    eclipse_risk_score: float
    peer_diversity: float
    alerts_active: int
    last_updated: str


class FullMetrics(BaseModel):
    summary: MetricsSummary
    transactions: TransactionMetrics
    security: SecurityMetrics
    raw_metrics: dict


# =============================================================================
# Database Setup
# =============================================================================

def init_db():
    """Initialize SQLite database"""
    os.makedirs(os.path.dirname(DATABASE_PATH) if os.path.dirname(DATABASE_PATH) else ".", exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_key TEXT UNIQUE,
            api_key_created_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# Authentication Utilities
# =============================================================================

def hash_password(password: str) -> str:
    """Hash password with SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"cpsk_{secrets.token_urlsafe(32)}"


def create_jwt_token(username: str) -> str:
    """Create JWT token for user"""
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[str]:
    """Verify JWT token and return username"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current user from JWT token"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    username = verify_jwt_token(credentials.credentials)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return username


def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """Verify API key and return username"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required. Pass it in X-API-Key header.")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE api_key = ?", (x_api_key,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return row["username"]


# =============================================================================
# Prometheus Query Functions
# =============================================================================

async def query_prometheus(query: str) -> float:
    """Query Prometheus for a metric value"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query},
                timeout=10.0
            )
            
            if response.status_code != 200:
                return 0.0
            
            data = response.json()
            if data.get("status") == "success" and data.get("data", {}).get("result"):
                result = data["data"]["result"]
                if result and len(result) > 0:
                    return float(result[0]["value"][1])
            
            return 0.0
    except Exception:
        return 0.0


async def query_prometheus_range(query: str, start: str, end: str, step: str = "60s") -> list:
    """Query Prometheus for time-series data"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start,
                    "end": end,
                    "step": step
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            if data.get("status") == "success" and data.get("data", {}).get("result"):
                result = data["data"]["result"]
                if result and len(result) > 0:
                    return result[0].get("values", [])
            
            return []
    except Exception:
        return []


async def fetch_all_metrics() -> dict:
    """Fetch all relevant metrics from Prometheus"""
    metrics = {}
    
    # CP1 - Transaction Metrics
    metrics["cp1_tx_ingested_total"] = await query_prometheus("cp1_tx_ingested_total")
    metrics["cp1_accept_count_total"] = await query_prometheus("cp1_accept_count_total")
    metrics["cp1_flag_count_total"] = await query_prometheus("cp1_flag_count_total")
    metrics["cp1_reject_count_total"] = await query_prometheus("cp1_reject_count_total")
    metrics["cp1_consensus_passed_total"] = await query_prometheus("cp1_consensus_passed_total")
    metrics["cp1_consensus_failed_total"] = await query_prometheus("cp1_consensus_failed_total")
    metrics["cp1_ml_inference_total"] = await query_prometheus("cp1_ml_inference_total")
    metrics["cp1_inference_latency_seconds"] = await query_prometheus("cp1_inference_latency_seconds")
    
    # CP2 - Peer Security Metrics
    metrics["cp2_peer_connections"] = await query_prometheus("cp2_peer_connections")
    metrics["cp2_sybil_risk_score"] = await query_prometheus("cp2_sybil_risk_score")
    metrics["cp2_eclipse_risk_score"] = await query_prometheus("cp2_eclipse_risk_score")
    metrics["cp2_peer_diversity_score"] = await query_prometheus("cp2_peer_diversity_score")
    
    return metrics


# =============================================================================
# API Endpoints - Authentication
# =============================================================================

@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    init_db()


@app.post("/api/register", response_model=TokenResponse, tags=["Authentication"])
async def register(user: UserRegister):
    """
    Register a new user account.
    
    After registration, you'll receive a JWT token to access protected endpoints.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            (user.username, user.email)
        )
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        # Create user
        password_hash = hash_password(user.password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (user.username, user.email, password_hash)
        )
        conn.commit()
    
    token = create_jwt_token(user.username)
    return TokenResponse(access_token=token)


@app.post("/api/login", response_model=TokenResponse, tags=["Authentication"])
async def login(user: UserLogin):
    """
    Login and get a JWT token.
    
    Use this token in the Authorization header: `Bearer <token>`
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (user.username,)
        )
        row = cursor.fetchone()
        
        if not row or row["password_hash"] != hash_password(user.password):
            raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = create_jwt_token(user.username)
    return TokenResponse(access_token=token)


@app.get("/api/apikey", response_model=APIKeyResponse, tags=["Authentication"])
async def get_api_key(username: str = Depends(get_current_user)):
    """
    Get or generate your API key.
    
    Use this key in the `X-API-Key` header to access metrics without JWT.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT api_key, api_key_created_at FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        
        if row and row["api_key"]:
            return APIKeyResponse(
                api_key=row["api_key"],
                created_at=row["api_key_created_at"],
                message="Existing API key retrieved"
            )
        
        # Generate new API key
        api_key = generate_api_key()
        created_at = datetime.utcnow().isoformat()
        
        cursor.execute(
            "UPDATE users SET api_key = ?, api_key_created_at = ? WHERE username = ?",
            (api_key, created_at, username)
        )
        conn.commit()
        
        return APIKeyResponse(
            api_key=api_key,
            created_at=created_at,
            message="New API key generated"
        )


@app.post("/api/apikey/regenerate", response_model=APIKeyResponse, tags=["Authentication"])
async def regenerate_api_key(username: str = Depends(get_current_user)):
    """
    Regenerate your API key.
    
    This will invalidate the old key.
    """
    api_key = generate_api_key()
    created_at = datetime.utcnow().isoformat()
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET api_key = ?, api_key_created_at = ? WHERE username = ?",
            (api_key, created_at, username)
        )
        conn.commit()
    
    return APIKeyResponse(
        api_key=api_key,
        created_at=created_at,
        message="API key regenerated. Old key is now invalid."
    )


# =============================================================================
# API Endpoints - Metrics (Public)
# =============================================================================

@app.get("/api/metrics", response_model=FullMetrics, tags=["Metrics"])
async def get_all_metrics():
    """
    Get all metrics in one call.
    
    Returns summary, transaction metrics, security metrics, and raw values.
    
    This endpoint is public and requires no authentication.
    """
    raw = await fetch_all_metrics()
    now = datetime.utcnow().isoformat()
    
    total = raw.get("cp1_tx_ingested_total", 0)
    accepted = raw.get("cp1_accept_count_total", 0)
    flagged = raw.get("cp1_flag_count_total", 0)
    rejected = raw.get("cp1_reject_count_total", 0)
    
    total_decisions = accepted + flagged + rejected
    
    summary = MetricsSummary(
        total_transactions=total,
        accepted=accepted,
        flagged=flagged,
        rejected=rejected,
        accept_rate=round(accepted / total_decisions * 100, 2) if total_decisions > 0 else 0,
        flag_rate=round(flagged / total_decisions * 100, 2) if total_decisions > 0 else 0,
        reject_rate=round(rejected / total_decisions * 100, 2) if total_decisions > 0 else 0,
        last_updated=now
    )
    
    transactions = TransactionMetrics(
        total_ingested=raw.get("cp1_tx_ingested_total", 0),
        consensus_passed=raw.get("cp1_consensus_passed_total", 0),
        consensus_failed=raw.get("cp1_consensus_failed_total", 0),
        ml_inferences=raw.get("cp1_ml_inference_total", 0),
        decisions={
            "accept": accepted,
            "flag": flagged,
            "reject": rejected
        },
        last_updated=now
    )
    
    security = SecurityMetrics(
        connected_peers=raw.get("cp2_peer_connections", 0),
        sybil_risk_score=raw.get("cp2_sybil_risk_score", 0),
        eclipse_risk_score=raw.get("cp2_eclipse_risk_score", 0),
        peer_diversity=raw.get("cp2_peer_diversity_score", 0),
        alerts_active=0,  # TODO: Query alert manager
        last_updated=now
    )
    
    return FullMetrics(
        summary=summary,
        transactions=transactions,
        security=security,
        raw_metrics=raw
    )


@app.get("/api/metrics/summary", response_model=MetricsSummary, tags=["Metrics"])
async def get_summary_metrics():
    """
    Get summary metrics only.
    
    Lightweight endpoint for dashboard cards. Public endpoint.
    """
    raw = await fetch_all_metrics()
    now = datetime.utcnow().isoformat()
    
    total = raw.get("cp1_tx_ingested_total", 0)
    accepted = raw.get("cp1_accept_count_total", 0)
    flagged = raw.get("cp1_flag_count_total", 0)
    rejected = raw.get("cp1_reject_count_total", 0)
    
    total_decisions = accepted + flagged + rejected
    
    return MetricsSummary(
        total_transactions=total,
        accepted=accepted,
        flagged=flagged,
        rejected=rejected,
        accept_rate=round(accepted / total_decisions * 100, 2) if total_decisions > 0 else 0,
        flag_rate=round(flagged / total_decisions * 100, 2) if total_decisions > 0 else 0,
        reject_rate=round(rejected / total_decisions * 100, 2) if total_decisions > 0 else 0,
        last_updated=now
    )


@app.get("/api/metrics/transactions", response_model=TransactionMetrics, tags=["Metrics"])
async def get_transaction_metrics():
    """
    Get transaction-specific metrics. Public endpoint.
    """
    raw = await fetch_all_metrics()
    now = datetime.utcnow().isoformat()
    
    return TransactionMetrics(
        total_ingested=raw.get("cp1_tx_ingested_total", 0),
        consensus_passed=raw.get("cp1_consensus_passed_total", 0),
        consensus_failed=raw.get("cp1_consensus_failed_total", 0),
        ml_inferences=raw.get("cp1_ml_inference_total", 0),
        decisions={
            "accept": raw.get("cp1_accept_count_total", 0),
            "flag": raw.get("cp1_flag_count_total", 0),
            "reject": raw.get("cp1_reject_count_total", 0)
        },
        last_updated=now
    )


@app.get("/api/metrics/security", response_model=SecurityMetrics, tags=["Metrics"])
async def get_security_metrics():
    """
    Get security/peer metrics. Public endpoint.
    """
    raw = await fetch_all_metrics()
    now = datetime.utcnow().isoformat()
    
    return SecurityMetrics(
        connected_peers=raw.get("cp2_peer_connections", 0),
        sybil_risk_score=raw.get("cp2_sybil_risk_score", 0),
        eclipse_risk_score=raw.get("cp2_eclipse_risk_score", 0),
        peer_diversity=raw.get("cp2_peer_diversity_score", 0),
        alerts_active=0,
        last_updated=now
    )


@app.get("/api/metrics/history", tags=["Metrics"])
async def get_metrics_history(
    metric: str = Query(..., description="Metric name, e.g., cp1_tx_ingested_total"),
    hours: int = Query(1, ge=1, le=24, description="Hours of history to fetch")
):
    """
    Get historical time-series data for a specific metric.
    
    Useful for building charts in your web app. Public endpoint.
    """
    from datetime import datetime, timedelta
    
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    
    values = await query_prometheus_range(
        metric,
        start.isoformat() + "Z",
        end.isoformat() + "Z",
        step="60s"
    )
    
    # Convert to more usable format
    data = [
        {"timestamp": int(v[0]), "value": float(v[1])}
        for v in values
    ]
    
    return {
        "metric": metric,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "data": data
    }


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/", tags=["System"])
async def root():
    """API documentation redirect"""
    return {
        "message": "CP Security Metrics API",
        "docs": "/docs",
        "version": "1.0.0",
        "endpoints": {
            "register": "POST /api/register",
            "login": "POST /api/login",
            "get_api_key": "GET /api/apikey (requires JWT)",
            "metrics": "GET /api/metrics (public)",
            "summary": "GET /api/metrics/summary (public)",
            "transactions": "GET /api/metrics/transactions (public)",
            "security": "GET /api/metrics/security (public)",
            "history": "GET /api/metrics/history?metric=<name>&hours=1 (public)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
