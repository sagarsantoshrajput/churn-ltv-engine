"""Week 2 (Day 7): SHAP explainability for business stakeholders.
Contributions are summed back from one-hot columns to the ORIGINAL business features
(e.g. all `contract_*` columns -> `contract`)."""
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

log = logging.getLogger(__name__)
OUT = config.REPORTS_DIR / "model"


class ChurnExplainer:
    def __init__(self, xgb_pipeline):
        self.prep = xgb_pipeline.named_steps["prep"]
        self.clf = xgb_pipeline.named_steps["clf"]
        self.explainer = shap.TreeExplainer(self.clf)
        ohe = self.prep.named_transformers_["cat"]
        self.blocks, i = {}, 0
        for c in NUMERIC_FEATURES:
            self.blocks[c] = [i]; i += 1
        for c, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
            self.blocks[c] = list(range(i, i + len(cats))); i += len(cats)
        self.feature_names = list(self.prep.get_feature_names_out())

    def transform(self, X):
        return self.prep.transform(X)

    def shap_values(self, Xt):
        sv = self.explainer.shap_values(Xt)
        return sv[1] if isinstance(sv, list) else sv

    def by_feature(self, sv) -> pd.DataFrame:
        return pd.DataFrame({c: sv[:, idx].sum(axis=1) for c, idx in self.blocks.items()})

    def top_drivers(self, X: pd.DataFrame, k: int = 5):
        """Per-row list of the k features pushing churn risk up/down (log-odds impact)."""
        agg = self.by_feature(self.shap_values(self.transform(X)))
        out = []
        for i in range(len(X)):
            row = agg.iloc[i]
            top = row.reindex(row.abs().sort_values(ascending=False).index)[:k]
            out.append([{"feature": f, "value": _py(X.iloc[i][f]), "impact": round(float(v), 4),
                         "effect": "increases churn risk" if v > 0 else "reduces churn risk"}
                        for f, v in top.items()])
        return out


def _py(v):
    return v.item() if hasattr(v, "item") else v


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    xgb_pipe = joblib.load(config.MODELS_DIR / "churn_xgb.joblib")
    Xte, _ = joblib.load(config.MODELS_DIR / "churn_test_set.joblib")
    ex = ChurnExplainer(xgb_pipe)
    Xt = ex.transform(Xte)
    sv = ex.shap_values(Xt)

    # global importance on original features
    agg = ex.by_feature(sv)
    imp = agg.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
    imp.columns = ["feature", "mean_abs_shap"]
    imp.to_csv(OUT / "shap_feature_importance.csv", index=False)

    plt.figure(figsize=(7, 6))
    top = imp.head(15).iloc[::-1]
    plt.barh(top.feature, top.mean_abs_shap, color="#4c72b0")
    plt.xlabel("mean |SHAP| (impact on churn log-odds)"); plt.title("What drives churn? (business features)")
    plt.tight_layout(); plt.savefig(OUT / "shap_importance_bar.png", dpi=120); plt.close()

    shap.summary_plot(sv, Xt, feature_names=ex.feature_names, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(OUT / "shap_beeswarm.png", dpi=120, bbox_inches="tight"); plt.close()
    log.info("Top churn drivers: %s", imp.head(5).feature.tolist())
    return imp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
