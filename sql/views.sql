-- Reporting views (portable across PostgreSQL / SQLite). Statements separated by ';'

DROP VIEW IF EXISTS v_churn_by_contract;
CREATE VIEW v_churn_by_contract AS
SELECT contract,
       COUNT(*) AS customers,
       SUM(churn) AS churned,
       ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct,
       ROUND(CAST(AVG(monthly_charges) AS NUMERIC), 2) AS avg_monthly_charges
FROM customers_clean GROUP BY contract;

DROP VIEW IF EXISTS v_churn_by_tenure;
CREATE VIEW v_churn_by_tenure AS
SELECT CASE WHEN tenure <= 12 THEN '1) 0-12m' WHEN tenure <= 24 THEN '2) 13-24m'
            WHEN tenure <= 48 THEN '3) 25-48m' ELSE '4) 49m+' END AS tenure_bucket,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers_clean GROUP BY 1;

DROP VIEW IF EXISTS v_churn_by_internet_service;
CREATE VIEW v_churn_by_internet_service AS
SELECT internet_service, COUNT(*) AS customers,
       ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers_clean GROUP BY internet_service;

DROP VIEW IF EXISTS v_churn_by_payment_method;
CREATE VIEW v_churn_by_payment_method AS
SELECT payment_method, COUNT(*) AS customers,
       ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers_clean GROUP BY payment_method;

DROP VIEW IF EXISTS v_risk_summary;
CREATE VIEW v_risk_summary AS
SELECT risk_band, COUNT(*) AS customers,
       ROUND(CAST(AVG(churn_probability) AS NUMERIC), 3) AS avg_churn_probability,
       ROUND(CAST(SUM(revenue_at_risk) AS NUMERIC), 0) AS revenue_at_risk
FROM churn_predictions GROUP BY risk_band;

DROP VIEW IF EXISTS v_ltv_segment_summary;
CREATE VIEW v_ltv_segment_summary AS
SELECT ltv_segment, COUNT(*) AS customers,
       ROUND(CAST(AVG(predicted_ltv) AS NUMERIC), 2) AS avg_ltv,
       ROUND(CAST(SUM(predicted_ltv) AS NUMERIC), 0) AS total_ltv,
       ROUND(CAST(AVG(churn_probability) AS NUMERIC), 3) AS avg_churn_probability,
       ROUND(CAST(SUM(revenue_at_risk) AS NUMERIC), 0) AS revenue_at_risk
FROM churn_predictions GROUP BY ltv_segment;

DROP VIEW IF EXISTS v_risk_ltv_matrix;
CREATE VIEW v_risk_ltv_matrix AS
SELECT risk_band, ltv_segment, COUNT(*) AS customers,
       ROUND(CAST(SUM(revenue_at_risk) AS NUMERIC), 0) AS revenue_at_risk
FROM churn_predictions GROUP BY risk_band, ltv_segment;

DROP VIEW IF EXISTS v_retention_priority_list;
CREATE VIEW v_retention_priority_list AS
SELECT p.customer_id, p.retention_priority, p.risk_band, p.ltv_segment,
       ROUND(CAST(p.churn_probability AS NUMERIC), 3) AS churn_probability,
       ROUND(CAST(p.predicted_ltv AS NUMERIC), 2) AS predicted_ltv,
       ROUND(CAST(p.revenue_at_risk AS NUMERIC), 2) AS revenue_at_risk,
       c.contract, c.tenure, c.monthly_charges, c.internet_service, c.payment_method
FROM churn_predictions p JOIN customers_clean c ON c.customer_id = p.customer_id;

DROP VIEW IF EXISTS v_kpis;
CREATE VIEW v_kpis AS
SELECT (SELECT COUNT(*) FROM customers_clean) AS total_customers,
       (SELECT ROUND(100.0 * SUM(churn) / COUNT(*), 2) FROM customers_clean) AS actual_churn_rate_pct,
       (SELECT COUNT(*) FROM churn_predictions WHERE risk_band = 'High') AS high_risk_customers,
       (SELECT ROUND(CAST(SUM(revenue_at_risk) AS NUMERIC), 0) FROM churn_predictions) AS total_revenue_at_risk,
       (SELECT ROUND(CAST(AVG(predicted_ltv) AS NUMERIC), 2) FROM churn_predictions) AS avg_predicted_ltv
