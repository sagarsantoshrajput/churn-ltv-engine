# Dashboards (Week 4, Day 1-5)

All dashboard data comes from PostgreSQL views created by `sql/views.sql`; the API adds live what-if scoring.

## Connect Metabase (included in docker-compose, http://localhost:3000)
Admin -> Add database -> PostgreSQL -> host `db` (or `localhost` outside Docker), port `5432`,
db `churn_db`, user/password `churn` / `churn`.

## Connect Apache Superset (alternative)
Data -> Databases -> + Database -> PostgreSQL, SQLAlchemy URI:
`postgresql+psycopg2://churn:churn@<host>:5432/churn_db`, then add the `v_*` views as datasets.

## Dashboard 1 - Global Churn Risk
| Card | Source | Chart |
|---|---|---|
| KPI tiles | `v_kpis` | Number |
| Risk band mix + revenue at risk | `v_risk_summary` | Donut / bar |
| Churn rate by contract | `v_churn_by_contract` | Bar |
| Churn rate by tenure bucket | `v_churn_by_tenure` | Line/bar |
| Churn by internet service / payment | `v_churn_by_internet_service`, `v_churn_by_payment_method` | Bar |

## Dashboard 2 - LTV Segmentation & Retention Targeting
| Card | Source | Chart |
|---|---|---|
| Customers, avg LTV, total LTV per tier | `v_ltv_segment_summary` | Table / bar |
| Risk x LTV heat-map | `v_risk_ltv_matrix` | Pivot (rows=ltv_segment, cols=risk_band, value=revenue_at_risk) |
| **Call list: P1 - Retain now** | `v_retention_priority_list` filtered `retention_priority = 'P1 - Retain now'` sorted by `revenue_at_risk` | Table (export CSV for marketing) |

Add dashboard filters on `contract`, `ltv_segment`, `risk_band`.

## API-connected usage
The model API (`POST /predict`, `/predict/batch`) can be called from Superset/Metabase-adjacent tools
(e.g. Superset dataset SQL is DB-only, so predictions are published to `churn_predictions` by
`python -m src.score_customers`; schedule it nightly with cron/Airflow after new data lands).
