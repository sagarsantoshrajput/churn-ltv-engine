"""Week 1 (Day 1-2): load the Telco dataset into PostgreSQL (raw -> clean)."""
import logging
import urllib.request

import pandas as pd

from src import config, db
from src.preprocessing import clean, normalize_columns, quality_report

log = logging.getLogger(__name__)


def ensure_dataset():
    if not config.DATA_RAW.exists():
        log.info("Downloading dataset from %s", config.DATASET_URL)
        urllib.request.urlretrieve(config.DATASET_URL, config.DATA_RAW)
    return config.DATA_RAW


def run():
    db.init_db()
    raw = normalize_columns(pd.read_csv(ensure_dataset(), dtype=str))
    db.replace_rows(db.raw_telco_customers, raw[[c.name for c in db.raw_telco_customers.columns]])
    log.info("Loaded %d rows into raw_telco_customers", len(raw))

    df = clean(raw)
    qa = quality_report(df)
    log.info("Data-quality report: %s", qa)
    assert qa["duplicate_ids"] == 0, "duplicate customer ids found"
    assert qa["negative_charges"] == 0, "negative charges found"
    db.replace_rows(db.customers_clean, df[[c.name for c in db.customers_clean.columns]])
    db.apply_views()
    log.info("Loaded %d rows into customers_clean and created reporting views", len(df))
    return qa


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
