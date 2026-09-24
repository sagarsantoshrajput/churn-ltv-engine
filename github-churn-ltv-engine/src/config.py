"""Central configuration. Override anything via environment variables / .env"""
import os
from pathlib import Path

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SQL_DIR = ROOT / "sql"

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://churn:churn@localhost:5432/churn_db"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
LTV_HORIZON_MONTHS = int(os.getenv("LTV_HORIZON_MONTHS", 36))  # forward revenue window

# Churn-probability -> risk band
RISK_HIGH = float(os.getenv("RISK_HIGH", 0.60))
RISK_MEDIUM = float(os.getenv("RISK_MEDIUM", 0.30))

for _d in (MODELS_DIR, REPORTS_DIR, DATA_RAW.parent):
    _d.mkdir(parents=True, exist_ok=True)
