-- Reference DDL (PostgreSQL). The pipeline creates these tables automatically via
-- SQLAlchemy (src/db.py); this file documents the warehouse layout.

CREATE TABLE IF NOT EXISTS raw_telco_customers (      -- landing zone (text as-is)
    customer_id TEXT, gender TEXT, senior_citizen TEXT, partner TEXT, dependents TEXT,
    tenure TEXT, phone_service TEXT, multiple_lines TEXT, internet_service TEXT,
    online_security TEXT, online_backup TEXT, device_protection TEXT, tech_support TEXT,
    streaming_tv TEXT, streaming_movies TEXT, contract TEXT, paperless_billing TEXT,
    payment_method TEXT, monthly_charges TEXT, total_charges TEXT, churn TEXT
);

CREATE TABLE IF NOT EXISTS customers_clean (          -- typed, validated, imputed
    customer_id TEXT PRIMARY KEY, gender TEXT, partner TEXT, dependents TEXT,
    phone_service TEXT, multiple_lines TEXT, internet_service TEXT, online_security TEXT,
    online_backup TEXT, device_protection TEXT, tech_support TEXT, streaming_tv TEXT,
    streaming_movies TEXT, contract TEXT, paperless_billing TEXT, payment_method TEXT,
    senior_citizen INTEGER, tenure INTEGER, monthly_charges DOUBLE PRECISION,
    total_charges DOUBLE PRECISION, churn INTEGER
);

CREATE TABLE IF NOT EXISTS churn_predictions (        -- model output for dashboards
    customer_id TEXT PRIMARY KEY, churn_probability DOUBLE PRECISION, predicted_churn INTEGER,
    risk_band TEXT, predicted_future_revenue DOUBLE PRECISION, predicted_ltv DOUBLE PRECISION,
    ltv_segment TEXT, revenue_at_risk DOUBLE PRECISION, retention_priority TEXT,
    scored_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY, run_at TIMESTAMP, task TEXT, model_name TEXT, split TEXT,
    metric TEXT, value DOUBLE PRECISION, is_champion INTEGER
);
