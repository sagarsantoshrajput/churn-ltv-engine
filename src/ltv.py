"""Week 3 (Day 1-3): Customer Lifetime Value.

LTV = revenue already collected (total_charges) + forecast future revenue.
Future revenue target = monthly_charges x E[remaining months over the next N months], where
survival is estimated per contract type with a smoothed Kaplan-Meier hazard curve
(tenure = observation time, churn = event). A regression model (Ridge / Random Forest /
XGBoost, chosen by CV RMSE) then learns to forecast that value from customer attributes, so
new customers can be scored through the API without the survival tables.
"""
import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src import config, db
from src.features import ALL_FEATURES, build_preprocessor, engineer_features

log = logging.getLogger(__name__)
OUT = config.REPORTS_DIR / "model"


def survival_curves(df, horizon):
    """Smoothed Kaplan-Meier survival S(t) per contract, extended `horizon` months past max tenure."""
    max_t = int(df["tenure"].max())
    curves = {}
    for contract, g in df.groupby("contract"):
        t, e = g["tenure"].clip(lower=1).to_numpy(), g["churn"].to_numpy()
        haz = np.array([((t == k) & (e == 1)).sum() / max((t >= k).sum(), 1) for k in range(1, max_t + 1)])
        haz = np.convolve(np.pad(haz, 3, mode="edge"), np.ones(7) / 7, mode="valid")  # smooth noise
        full = np.concatenate([[0.0], haz, np.full(horizon + 1, haz[-12:].mean())])
        curves[contract] = np.cumprod(1 - full).tolist()
    return curves


def expected_remaining_months(tenure, contract, curves, horizon):
    out = np.empty(len(tenure))
    for i, (t, c) in enumerate(zip(tenure, contract)):
        S = np.asarray(curves.get(c, curves["Month-to-month"]))
        t0 = int(min(t, len(S) - horizon - 2))
        out[i] = S[t0 + 1: t0 + horizon + 1].sum() / S[t0]
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    H = config.LTV_HORIZON_MONTHS
    df = engineer_features(db.load_customers())
    curves = survival_curves(df, H)
    df["exp_remaining_months"] = expected_remaining_months(df["tenure"], df["contract"], curves, H)
    df["future_revenue"] = df["monthly_charges"] * df["exp_remaining_months"]
    log.info("Avg expected remaining months (%d-mo horizon): %s", H,
             df.groupby("contract")["exp_remaining_months"].mean().round(1).to_dict())

    X, y = df[ALL_FEATURES], df["future_revenue"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
                                          stratify=df["contract"])
    rs = config.RANDOM_STATE
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=rs),
        "xgboost": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, n_jobs=-1, random_state=rs),
    }
    results, fitted = {}, {}
    for name, reg in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("reg", reg)])
        cv_rmse = -cross_val_score(pipe, Xtr, ytr, cv=5, scoring="neg_root_mean_squared_error").mean()
        pipe.fit(Xtr, ytr)
        pred = np.clip(pipe.predict(Xte), 0, None)
        results[name] = {"cv_rmse": cv_rmse, "mae": mean_absolute_error(yte, pred),
                         "rmse": mean_squared_error(yte, pred) ** 0.5, "r2": r2_score(yte, pred)}
        fitted[name] = pipe
        log.info("%-14s %s", name, {k: round(v, 3) for k, v in results[name].items()})
    champion = min(results, key=lambda n: results[n]["cv_rmse"])
    log.info("LTV champion: %s", champion)

    # LTV tiers = quartiles of predicted LTV across the current customer base
    pipe = fitted[champion]
    future = np.clip(pipe.predict(X), 0, None)
    ltv = df["total_charges"].to_numpy() + future
    edges = np.quantile(ltv, [0.25, 0.5, 0.75]).round(2).tolist()

    joblib.dump({"pipeline": pipe, "segment_edges": edges, "model_name": champion, "horizon_months": H},
                config.MODELS_DIR / "ltv_model.joblib")
    (config.MODELS_DIR / "ltv_metrics.json").write_text(json.dumps(
        {"champion": champion, "horizon_months": H, "segment_edges": edges, "models": results}, indent=2, default=float))
    (config.MODELS_DIR / "survival_curves.json").write_text(json.dumps(curves))

    pred = np.clip(pipe.predict(Xte), 0, None)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(yte, pred, s=6, alpha=0.4); lim = [0, max(yte.max(), pred.max())]
    ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("Target future revenue ($)"); ax[0].set_ylabel("Predicted ($)")
    ax[0].set_title(f"LTV regression ({champion}) - test set")
    ax[1].hist(ltv, bins=50, color="#4c72b0"); [ax[1].axvline(e, color="r", ls="--") for e in edges]
    ax[1].set_title("Predicted LTV distribution + tier cut-points"); ax[1].set_xlabel("LTV ($)")
    plt.tight_layout(); plt.savefig(OUT / "ltv_model.png", dpi=120); plt.close()

    db.save_metrics("ltv", [dict(model_name=n, split="test", metric=k, value=float(v), is_champion=int(n == champion))
                            for n, m in results.items() for k, v in m.items()])
    return results, champion


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
