"""PostgreSQL data-warehouse layer (SQLAlchemy). Works with any SQLAlchemy URL."""
import logging
from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, Table, Text,
                        create_engine, text)

from src import config

log = logging.getLogger(__name__)
metadata = MetaData()

_RAW_COLS = ["customer_id", "gender", "senior_citizen", "partner", "dependents", "tenure",
             "phone_service", "multiple_lines", "internet_service", "online_security",
             "online_backup", "device_protection", "tech_support", "streaming_tv",
             "streaming_movies", "contract", "paperless_billing", "payment_method",
             "monthly_charges", "total_charges", "churn"]

# Landing zone: untouched CSV values as text (TotalCharges contains blanks)
raw_telco_customers = Table("raw_telco_customers", metadata,
                            *[Column(c, Text) for c in _RAW_COLS])

customers_clean = Table(
    "customers_clean", metadata,
    Column("customer_id", Text, primary_key=True),
    *[Column(c, Text) for c in ["gender", "partner", "dependents", "phone_service",
                                "multiple_lines", "internet_service", "online_security",
                                "online_backup", "device_protection", "tech_support",
                                "streaming_tv", "streaming_movies", "contract",
                                "paperless_billing", "payment_method"]],
    Column("senior_citizen", Integer), Column("tenure", Integer),
    Column("monthly_charges", Float), Column("total_charges", Float),
    Column("churn", Integer),
)

churn_predictions = Table(
    "churn_predictions", metadata,
    Column("customer_id", Text, primary_key=True),
    Column("churn_probability", Float), Column("predicted_churn", Integer),
    Column("risk_band", Text), Column("predicted_future_revenue", Float),
    Column("predicted_ltv", Float), Column("ltv_segment", Text),
    Column("revenue_at_risk", Float), Column("retention_priority", Text),
    Column("scored_at", DateTime, default=datetime.utcnow),
)

model_metrics = Table(
    "model_metrics", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_at", DateTime, default=datetime.utcnow),
    Column("task", Text), Column("model_name", Text), Column("split", Text),
    Column("metric", Text), Column("value", Float), Column("is_champion", Integer),
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(config.DATABASE_URL, pool_pre_ping=True, future=True)
    return _engine


def init_db():
    """Create tables if missing (idempotent)."""
    metadata.create_all(get_engine())


def replace_rows(table: Table, df, chunksize: int = 1000):
    """Truncate-and-load. Keeps the table object (and dependent views) intact."""
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(table.delete())
    df.to_sql(table.name, eng, if_exists="append", index=False, chunksize=chunksize)


def apply_views():
    """(Re)create reporting views used by Superset / Metabase."""
    lines = (config.SQL_DIR / "views.sql").read_text().splitlines()
    sql = "\n".join(l for l in lines if not l.strip().startswith("--"))
    stmts = [s.strip() for s in sql.split(";") if s.strip()]
    with get_engine().begin() as conn:
        for s in stmts:
            conn.execute(text(s))
    log.info("Applied %d view statements", len(stmts))


def load_customers():
    import pandas as pd
    df = pd.read_sql(customers_clean.select(), get_engine())
    df.columns = [str(c) for c in df.columns]
    return df


def save_metrics(task: str, rows: list):
    """rows: dicts with model_name, split, metric, value, is_champion."""
    import pandas as pd
    df = pd.DataFrame(rows)
    df["task"], df["run_at"] = task, datetime.utcnow()
    with get_engine().begin() as conn:
        conn.execute(model_metrics.delete().where(model_metrics.c.task == task))
    df.to_sql("model_metrics", get_engine(), if_exists="append", index=False)
