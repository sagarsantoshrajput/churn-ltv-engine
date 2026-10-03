"""Shared scoring service used by the batch job and the FastAPI app."""
import json
import logging

import joblib
import numpy as np
import pandas as pd

from src import config, segments
from src.explain import ChurnExplainer
from src.features import ALL_FEATURES, engineer_features
from src.preprocessing import clean

log = logging.getLogger(__name__)


class ScoringService:
    def __init__(self):
        d = config.MODELS_DIR
        churn = joblib.load(d / "churn_model.joblib")
        ltv = joblib.load(d / "ltv_model.joblib")
        self.churn_pipe, self.threshold, self.churn_name = churn["pipeline"], churn["threshold"], churn["model_name"]
        self.ltv_pipe, self.edges, self.ltv_name = ltv["pipeline"], ltv["segment_edges"], ltv["model_name"]
        self.horizon = ltv["horizon_months"]
        self.explainer = ChurnExplainer(joblib.load(d / "churn_xgb.joblib"))
        self.metrics = {"churn": json.loads((d / "churn_metrics.json").read_text()),
                        "ltv": json.loads((d / "ltv_metrics.json").read_text())}

    def score(self, raw: pd.DataFrame, explain: bool = False, top_k: int = 5) -> pd.DataFrame:
        df = clean(raw.reset_index(drop=True))
        if "customer_id" not in df:
            df["customer_id"] = ""
        # blank / missing ids (e.g. the dashboard form) get a row label instead of None
        df["customer_id"] = [
            str(v).strip() if str(v).strip().lower() not in ("", "none", "nan", "<na>") else f"row-{i}"
            for i, v in enumerate(df["customer_id"])]
        f = engineer_features(df)
        X = f[ALL_FEATURES]
        p = self.churn_pipe.predict_proba(X)[:, 1]
        future = np.clip(self.ltv_pipe.predict(X), 0, None)
        out = pd.DataFrame({"customer_id": df["customer_id"], "churn_probability": p.round(4),
                            "predicted_churn": (p >= self.threshold).astype(int)})
        out["risk_band"] = segments.risk_band(out["churn_probability"])
        out["predicted_future_revenue"] = future.round(2)
        out["predicted_ltv"] = (df["total_charges"].to_numpy() + future).round(2)
        out["ltv_segment"] = segments.ltv_segment(out["predicted_ltv"], self.edges)
        out["revenue_at_risk"] = (future * p).round(2)
        out["retention_priority"] = segments.retention_priority(out["risk_band"], out["ltv_segment"])
        if explain:
            out["top_drivers"] = self.explainer.top_drivers(X, top_k)
        return out
