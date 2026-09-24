# GitHub protocol (required by Zaalima)

1. Create the GitHub repository **before** starting: `git init && git remote add origin <url>`.
2. Branching: `main` (stable) <- `develop` <- `feature/<topic>` (e.g. `feature/eda`, `feature/xgboost-shap`).
   Open a PR per feature; squash-merge into `develop`; tag weekly releases `v0.1 ... v1.0`.
3. **Commit every day** with Conventional Commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
   Commit history is the daily-productivity metric, so commit real work as you do it.

## Suggested daily commit plan (maps to the 4-week timeline)
| Week | Day | Suggested commit(s) |
|---|---|---|
| 1 | 1-2 | `chore: project skeleton, docker-compose postgres` / `feat(db): SQLAlchemy models + ingest` |
| 1 | 3-5 | `feat(eda): seaborn EDA charts` / `docs: churn drivers findings` |
| 1 | 6-7 | `feat(preprocessing): missing values + encoding` / `docs: baseline analytics report` |
| 2 | 1-3 | `feat(features): usage-vs-base-charge, add-on count, tenure bucket` |
| 2 | 4-6 | `feat(models): LR/RF/XGBoost + precision/recall/F1` / `feat(models): threshold tuning` |
| 2 | 7 | `feat(explain): SHAP importance + per-customer drivers` |
| 3 | 1-3 | `feat(ltv): survival-based target + regression models` |
| 3 | 4-7 | `feat(api): /predict` / `feat(api): batch + csv` / `test(api): pytest suite` |
| 4 | 1-3 | `feat(sql): reporting views` / `docs: Metabase/Superset connection` |
| 4 | 4-5 | `feat(dashboards): churn-risk + LTV dashboards` |
| 4 | 6-7 | `build: Dockerfile + compose` / `docs: technical documentation` |
