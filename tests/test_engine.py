"""Run after the pipeline:  DATABASE_URL=sqlite:///./test.db python -m src.pipeline && pytest"""
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import config
from src.features import ALL_FEATURES, engineer_features
from src.preprocessing import clean

pytestmark = pytest.mark.skipif(not (config.MODELS_DIR / "ltv_model.joblib").exists(),
                                reason="run `python -m src.pipeline` first")

RISKY = {"customer_id": "T-1", "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
         "tenure": 1, "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
         "online_security": "No", "online_backup": "No", "device_protection": "No", "tech_support": "No",
         "streaming_tv": "No", "streaming_movies": "No", "contract": "Month-to-month",
         "paperless_billing": "Yes", "payment_method": "Electronic check", "monthly_charges": 80.0}
LOYAL = {**RISKY, "customer_id": "T-2", "partner": "Yes", "dependents": "Yes", "tenure": 68,
         "contract": "Two year", "payment_method": "Bank transfer (automatic)",
         "internet_service": "DSL", "monthly_charges": 60.0, "total_charges": 4100.0,
         "tech_support": "Yes", "online_security": "Yes"}


def test_clean_handles_blank_total_charges():
    raw = pd.DataFrame([{**RISKY, "tenure": 0, "total_charges": " ", "churn": "No"}])
    out = clean(raw)
    assert out.loc[0, "total_charges"] == 0 and out.loc[0, "churn"] == 0


def test_feature_engineering_columns():
    f = engineer_features(clean(pd.DataFrame([LOYAL])))
    assert set(ALL_FEATURES) <= set(f.columns) and f.loc[0, "is_autopay"] == 1


def test_health_and_predict():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        risky = c.post("/predict", json=RISKY).json()
        loyal = c.post("/predict", json=LOYAL).json()
        assert risky["churn_probability"] > loyal["churn_probability"]
        assert risky["risk_band"] in {"Medium", "High"} and loyal["risk_band"] == "Low"
        assert len(risky["top_drivers"]) == 5
        assert loyal["predicted_ltv"] > risky["predicted_ltv"]


def test_batch_and_validation():
    with TestClient(app) as c:
        r = c.post("/predict/batch", json={"customers": [RISKY, LOYAL]}).json()
        assert r["count"] == 2 and len(r["predictions"]) == 2
        assert c.post("/predict", json={**RISKY, "contract": "Weekly"}).status_code == 422


def test_batch_csv_upload():
    df = pd.read_csv(config.DATA_RAW, dtype=str).head(25)
    buf = io.BytesIO(df.to_csv(index=False).encode())
    with TestClient(app) as c:
        r = c.post("/predict/batch/csv", files={"file": ("x.csv", buf, "text/csv")})
        assert r.status_code == 200 and r.json()["count"] == 25


def test_stored_prediction_lookup():
    with TestClient(app) as c:
        assert c.get("/customers/7590-VHVEG/risk").status_code in (200, 404)
        assert c.get("/customers/does-not-exist/risk").status_code == 404


def test_predict_without_customer_id():
    body = {k: v for k, v in RISKY.items() if k != "customer_id"}
    with TestClient(app) as c:
        r = c.post("/predict", json=body)
        assert r.status_code == 200 and r.json()["customer_id"] == "row-0"
