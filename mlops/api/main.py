import os
import pickle
import logging
from contextlib import asynccontextmanager
 
import numpy as np
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mysql.connector import pooling
from dotenv import load_dotenv
 
load_dotenv()
 
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
 
# ── Config from environment (never hard-code credentials) ────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_URI           = os.getenv("MODEL_URI", "models:/ChurnPredictionModel@production")
SCALER_PATH         = os.getenv("SCALER_PATH", "/app/model/scaler.pkl")
 
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "mlops-mysql"),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "database": os.getenv("DB_NAME", "mlops"),
    "port":     int(os.getenv("DB_PORT", "3306")),
}
 
# ── App state (loaded once at startup via lifespan) ───────────────────────────
app_state: dict = {}
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup, release at shutdown."""
 
    # --- scaler ---
    log.info("Loading scaler from %s", SCALER_PATH)
    with open(SCALER_PATH, "rb") as f:
        app_state["scaler"] = pickle.load(f)
 
    # --- MLflow model ---
    log.info("Connecting to MLflow at %s", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    log.info("Loading model from %s", MODEL_URI)
    app_state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
 
    # --- MySQL connection pool (5 connections, avoids reconnect on every request) ---
    log.info("Creating MySQL connection pool")
    app_state["pool"] = pooling.MySQLConnectionPool(
        pool_name="churn_pool",
        pool_size=5,
        **DB_CONFIG,
    )
    _ensure_table(app_state["pool"])
 
    log.info("Startup complete — API ready")
    yield
 
    # Cleanup (pool releases connections automatically)
    log.info("Shutting down")
 
 
def _ensure_table(pool: pooling.MySQLConnectionPool) -> None:
    """Create prediction_logs if it doesn't exist yet."""
    ddl = """
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id               INT AUTO_INCREMENT PRIMARY KEY,
        gender           TINYINT,
        tenure           FLOAT,
        monthly_charges  FLOAT,
        total_charges    FLOAT,
        prediction       TINYINT,
        probability      FLOAT,
        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(ddl)
        conn.commit()
    finally:
        conn.close()
 
 
# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)
 
 
class CustomerData(BaseModel):
    gender:           int
    SeniorCitizen:    int
    Partner:          int
    Dependents:       int
    tenure:           float
    PhoneService:     int
    MultipleLines:    int
    InternetService:  int
    OnlineSecurity:   int
    OnlineBackup:     int
    DeviceProtection: int
    TechSupport:      int
    StreamingTV:      int
    StreamingMovies:  int
    Contract:         int
    PaperlessBilling: int
    PaymentMethod:    int
    MonthlyCharges:   float
    TotalCharges:     float
 
 
# ── Routes ────────────────────────────────────────────────────────────────────
 
@app.get("/health")
def health():
    """Kubernetes liveness / readiness probe endpoint."""
    return {"status": "ok", "model": MODEL_URI}
 
 
@app.post("/predict")
def predict(data: CustomerData):
    scaler = app_state["scaler"]
    model  = app_state["model"]
    pool   = app_state["pool"]
 
    # Build feature matrix
    X = np.array([[
        data.gender, data.SeniorCitizen, data.Partner, data.Dependents,
        data.tenure, data.PhoneService, data.MultipleLines,
        data.InternetService, data.OnlineSecurity, data.OnlineBackup,
        data.DeviceProtection, data.TechSupport, data.StreamingTV,
        data.StreamingMovies, data.Contract, data.PaperlessBilling,
        data.PaymentMethod, data.MonthlyCharges, data.TotalCharges,
    ]])
 
    X_scaled   = scaler.transform(X)
    prediction = model.predict(X_scaled)
 
    probability: float | None = None
    try:
        proba       = model.predict_proba(X_scaled)   # type: ignore[attr-defined]
        probability = float(proba[0][1])
    except Exception:
        log.warning("Model does not support predict_proba — probability unavailable")
 
    # Persist to MySQL (get/release connection from pool)
    conn = pool.get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO prediction_logs
                (gender, tenure, monthly_charges, total_charges, prediction, probability)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (data.gender, data.tenure, data.MonthlyCharges, data.TotalCharges,
             int(prediction[0]), probability),
        )
        conn.commit()
    except Exception as exc:
        log.error("DB write failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database error — prediction not persisted") from exc
    finally:
        conn.close()   # returns connection to pool
 
    return {
        "churn_prediction":  int(prediction[0]),
        "churn_probability": probability,
    }