"""Web dashboard: everything shown as clean tables (served at http://localhost:8000/)."""
from pathlib import Path

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import FileResponse
from sqlalchemy import text

from src import db

QUERIES = {
    "kpis": "SELECT * FROM v_kpis",
    "risk": "SELECT * FROM v_risk_summary ORDER BY CASE risk_band WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END",
    "segments": "SELECT * FROM v_ltv_segment_summary ORDER BY avg_ltv DESC",
    "contracts": "SELECT * FROM v_churn_by_contract ORDER BY churn_rate_pct DESC",
    "priority": "SELECT * FROM v_retention_priority_list WHERE retention_priority = 'P1 - Retain now' "
                "ORDER BY revenue_at_risk DESC LIMIT 15",
}


def build_router(get_svc) -> APIRouter:
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    def home():
        return FileResponse(Path(__file__).parent / "static" / "dashboard.html")

    @router.get("/api/dashboard", tags=["dashboard"])
    def dashboard_data():
        out = {}
        with db.get_engine().connect() as conn:
            for name, sql in QUERIES.items():
                df = pd.read_sql(text(sql), conn)
                out[name] = df.astype(object).where(df.notna(), None).to_dict("records")
        m = get_svc().metrics
        out["churn_models"] = [{"model": k, "champion": k == m["churn"]["champion"], **v}
                               for k, v in m["churn"]["models"].items()]
        out["ltv_models"] = [{"model": k, "champion": k == m["ltv"]["champion"], **v}
                             for k, v in m["ltv"]["models"].items()]
        return out

    return router
