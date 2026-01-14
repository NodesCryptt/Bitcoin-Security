import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
MODEL_PATH = r"C:\Users\asus\cp1_projects\models\cp1_static_xgb_v1.joblib"
MEMPOOL_FEATS = r"C:\Users\asus\cp1_projects\results\cp1_mempool_features.csv"

model = joblib.load(MODEL_PATH)
mempool_df = pd.read_csv(MEMPOOL_FEATS)

# Use latest snapshot for context
latest = mempool_df.sort_values("timestamp").iloc[-1]

# Convert BTC/vB → sat/vB
SAT = 100_000_000
p50 = latest["fee_rate_median"] * SAT
p90 = latest["fee_rate_p90"] * SAT

# ---------------------------------------------------------
# CP1 HYBRID DECISION FUNCTION
# ---------------------------------------------------------
def cp1_decide(static_score, tx_fee_sat_vb):
    """
    static_score: float (0–1)
    tx_fee_sat_vb: float (sat/vB)
    """

    # Determine fee percentile bucket
    if tx_fee_sat_vb >= p90:
        mempool_level = "HIGH"
    elif tx_fee_sat_vb >= p50:
        mempool_level = "NORMAL"
    else:
        mempool_level = "LOW"

    # Decision logic
    if static_score < 0.15:
        decision = "ACCEPT"

    elif static_score < 0.35:
        decision = "REJECT" if mempool_level == "HIGH" else "FLAG"

    else:
        decision = "REJECT"

    return {
        "static_score": round(float(static_score), 4),
        "tx_fee_sat_vb": round(float(tx_fee_sat_vb), 2),
        "mempool_level": mempool_level,
        "decision": decision
    }

# ---------------------------------------------------------
# EXAMPLE TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    example_static = 0.27
    example_fee = 3.5  # sat/vB

    result = cp1_decide(example_static, example_fee)
    print(result)
