"""Week 1 (Day 3-7): EDA with Pandas + Seaborn, writes charts + baseline analytics report."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config, db
from src.features import engineer_features

log = logging.getLogger(__name__)
OUT = config.REPORTS_DIR / "eda"


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values) + "\n"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = engineer_features(db.load_customers())
    df["Churn"] = df["churn"].map({1: "Yes", 0: "No"})

    # 1. churn balance
    plt.figure(figsize=(4, 4)); sns.countplot(data=df, x="Churn", hue="Churn", legend=False)
    plt.title("Churn distribution"); plt.tight_layout(); plt.savefig(OUT / "churn_distribution.png", dpi=120); plt.close()

    # 2. churn rate by categorical drivers
    cats = ["contract", "internet_service", "payment_method", "tech_support", "paperless_billing", "tenure_bucket"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, c in zip(axes.ravel(), cats):
        rate = df.groupby(c)["churn"].mean().mul(100).sort_values()
        sns.barplot(x=rate.values, y=rate.index, ax=ax, color="#4c72b0")
        ax.set_title(f"Churn rate % by {c}"); ax.set_xlabel("%"); ax.set_ylabel("")
    plt.tight_layout(); plt.savefig(OUT / "churn_rate_by_category.png", dpi=120); plt.close()

    # 3. tenure & charges distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=36, multiple="stack", ax=axes[0])
    sns.boxplot(data=df, x="Churn", y="monthly_charges", hue="Churn", legend=False, ax=axes[1])
    sns.boxplot(data=df, x="contract", y="tenure", hue="Churn", ax=axes[2])
    axes[2].tick_params(axis="x", rotation=15)
    plt.tight_layout(); plt.savefig(OUT / "tenure_charges.png", dpi=120); plt.close()

    # 4. correlations (numeric + churn)
    num = ["tenure", "monthly_charges", "total_charges", "avg_monthly_spend", "num_addon_services",
           "is_autopay", "has_family", "senior_citizen", "churn"]
    corr = df[num].corr()
    plt.figure(figsize=(8, 6)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig(OUT / "correlations.png", dpi=120); plt.close()

    # baseline report
    by_contract = (df.groupby("contract").agg(customers=("churn", "size"), churn_rate_pct=("churn", lambda s: round(s.mean() * 100, 1)),
                                              avg_tenure=("tenure", "mean"), avg_monthly=("monthly_charges", "mean"))
                   .round(1).reset_index())
    by_tenure = df.groupby("tenure_bucket")["churn"].mean().mul(100).round(1).reset_index(name="churn_rate_pct")
    top_corr = corr["churn"].drop("churn").sort_values(key=abs, ascending=False).round(3).reset_index()
    top_corr.columns = ["feature", "corr_with_churn"]
    mtm = by_contract.loc[by_contract.contract == "Month-to-month", "churn_rate_pct"].iat[0]
    two = by_contract.loc[by_contract.contract == "Two year", "churn_rate_pct"].iat[0]
    report = f"""# Baseline Analytics Report - Telco Customer Churn

* Customers: **{len(df):,}**  |  Overall churn rate: **{df.churn.mean():.1%}**
* Avg tenure: {df.tenure.mean():.1f} months | Avg monthly charge: ${df.monthly_charges.mean():.2f}

## Key findings
* Month-to-month customers churn at **{mtm}%** vs **{two}%** on two-year contracts.
* Customers in their first 12 months are the most volatile (see tenure table).
* Tenure is the strongest negative correlate of churn; monthly charges is positively related.

## Churn by contract
{_md_table(by_contract)}
## Churn by tenure bucket
{_md_table(by_tenure)}
## Correlation with churn
{_md_table(top_corr)}
Charts: `churn_distribution.png`, `churn_rate_by_category.png`, `tenure_charges.png`, `correlations.png`.
"""
    (OUT / "baseline_report.md").write_text(report)
    log.info("EDA written to %s", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
