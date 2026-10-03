"""Week 3 (Day 4-7): FastAPI service - single-customer inference + batch processing."""
import io
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from sqlalchemy import select

from api.dashboard import build_router
from api.schemas import BatchIn, BatchOut, CustomerIn, PredictionOut
from src import db
from src.inference import ScoringService

state = {"svc": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["svc"] = ScoringService()
    except Exception as e:  # models not trained yet
        state["error"] = str(e)
    yield


app = FastAPI(title="Churn Prediction & LTV Engine", version="1.0.0", lifespan=lifespan,
              description="Serves churn risk, predicted lifetime value and SHAP explanations.")


def svc() -> ScoringService:
    if state["svc"] is None:
        raise HTTPException(503, f"Models not loaded - run `python -m src.pipeline` first. ({state['error']})")
    return state["svc"]


def _records(df: pd.DataFrame):
    return df.astype(object).where(df.notna(), None).to_dict("records")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok" if state["svc"] else "models_missing", "detail": state["error"]}


@app.get("/model/info", tags=["ops"])
def model_info():
    s = svc()
    return {"churn_model": s.churn_name, "decision_threshold": s.threshold,
            "ltv_model": s.ltv_name, "ltv_horizon_months": s.horizon, "ltv_segment_edges": s.edges,
            "metrics": s.metrics}


@app.post("/predict", response_model=PredictionOut, tags=["inference"])
def predict(customer: CustomerIn, explain: bool = Query(True, description="Include SHAP top drivers")):
    """Single-customer inference: churn probability, LTV, segment and reasons."""
    out = svc().score(pd.DataFrame([customer.model_dump()]), explain=explain)
    return _records(out)[0]


@app.post("/predict/batch", response_model=BatchOut, tags=["inference"])
def predict_batch(batch: BatchIn, explain: bool = False):
    out = svc().score(pd.DataFrame([c.model_dump() for c in batch.customers]), explain=explain)
    return {"count": len(out), "high_risk": int((out.risk_band == "High").sum()),
            "total_revenue_at_risk": round(float(out.revenue_at_risk.sum()), 2), "predictions": _records(out)}


@app.post("/predict/batch/csv", tags=["inference"])
async def predict_batch_csv(file: UploadFile = File(...)):
    """Upload a Telco-format CSV (original or snake_case headers); returns scored rows as JSON."""
    try:
        df = pd.read_csv(io.BytesIO(await file.read()), dtype=str)
        out = svc().score(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Could not score file: {e}")
    return {"count": len(out), "predictions": _records(out)}


@app.get("/customers/{customer_id}/risk", response_model=PredictionOut, tags=["warehouse"])
def stored_prediction(customer_id: str):
    """Look up the latest pre-computed score from the PostgreSQL warehouse."""
    t = db.churn_predictions
    with db.get_engine().connect() as conn:
        row = conn.execute(select(t).where(t.c.customer_id == customer_id)).mappings().first()
    if not row:
        raise HTTPException(404, "customer not found")
    return dict(row)


app.include_router(build_router(svc))
