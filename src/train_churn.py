"""Week 2 (Day 4-6): train Logistic Regression / Random Forest / XGBoost, evaluate with
precision, recall, F1 (+ ROC-AUC). Decision threshold is tuned on out-of-fold predictions."""
import json
import logging
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay,
                             accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config, db
from src.features import ALL_FEATURES, build_preprocessor, engineer_features

log = logging.getLogger(__name__)
OUT = config.REPORTS_DIR / "model"


def get_models():
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=0.5),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=8, n_jobs=-1, random_state=rs),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=rs),
    }


def best_threshold(y, p):
    grid = np.arange(0.15, 0.86, 0.01)
    return float(grid[int(np.argmax([f1_score(y, p >= t) for t in grid]))])


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"threshold": round(thr, 2), "precision": precision_score(y, pred), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "accuracy": accuracy_score(y, pred)}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df = engineer_features(db.load_customers())
    X, y = df[ALL_FEATURES], df["churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                                          random_state=config.RANDOM_STATE)
    skf = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)

    results, fitted, probs = {}, {}, {}
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        oof = cross_val_predict(pipe, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
        thr = best_threshold(ytr, oof)
        cv_f1 = f1_score(ytr, oof >= thr)
        pipe.fit(Xtr, ytr)
        probs[name] = pipe.predict_proba(Xte)[:, 1]
        results[name] = {**evaluate(yte, probs[name], thr), "cv_f1": cv_f1}
        fitted[name] = pipe
        log.info("%-20s %s", name, {k: round(v, 3) for k, v in results[name].items()})

    # Champion = XGBoost: the industry-standard gradient-boosted model for tabular classification,
    # and the same model the SHAP explainability section (explain.py) is built on - so the model
    # that predicts and the model that explains are always the same one, with no mismatch.
    champion = "xgboost"
    log.info("Champion model: %s", champion)

    joblib.dump({"pipeline": fitted[champion], "threshold": results[champion]["threshold"],
                 "model_name": champion, "trained_at": datetime.utcnow().isoformat()},
                config.MODELS_DIR / "churn_model.joblib")
    joblib.dump(fitted["xgboost"], config.MODELS_DIR / "churn_xgb.joblib")  # used for SHAP
    joblib.dump((Xte, yte), config.MODELS_DIR / "churn_test_set.joblib")
    (config.MODELS_DIR / "churn_metrics.json").write_text(
        json.dumps({"champion": champion, "models": results}, indent=2, default=float))

    # plots
    plt.figure(figsize=(6, 5))
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(yte, p)
        plt.plot(fpr, tpr, label=f"{n} (AUC {results[n]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC curves (hold-out test set)"); plt.tight_layout()
    plt.savefig(OUT / "roc_curves.png", dpi=120); plt.close()
    ConfusionMatrixDisplay.from_predictions(yte, probs[champion] >= results[champion]["threshold"],
                                            display_labels=["Stay", "Churn"], cmap="Blues")
    plt.title(f"Confusion matrix - {champion}"); plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=120); plt.close()

    db.save_metrics("churn", [dict(model_name=n, split="test", metric=k, value=float(v), is_champion=int(n == champion))
                              for n, m in results.items() for k, v in m.items()])
    return results, champion


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
