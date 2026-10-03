"""Cleaning: type fixes, missing-value handling, harmonised categories."""
import numpy as np
import pandas as pd

COLUMN_MAP = {
    "customerID": "customer_id", "SeniorCitizen": "senior_citizen", "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines", "InternetService": "internet_service",
    "OnlineSecurity": "online_security", "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection", "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv", "StreamingMovies": "streaming_movies", "Contract": "contract",
    "PaperlessBilling": "paperless_billing", "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges", "TotalCharges": "total_charges", "Churn": "churn",
    "Gender": "gender", "Partner": "partner", "Dependents": "dependents", "Tenure": "tenure",
}
NO_SERVICE_COLS = ["multiple_lines", "online_security", "online_backup", "device_protection",
                   "tech_support", "streaming_tv", "streaming_movies"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept original Telco headers (CustomerID/TotalCharges...) or snake_case."""
    return df.rename(columns=COLUMN_MAP).rename(columns=lambda c: c.strip().lower())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df).copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()

    for c in ("tenure", "monthly_charges"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "total_charges" not in df:
        df["total_charges"] = np.nan
    # TotalCharges arrives as text with blank strings for brand-new customers
    df["total_charges"] = pd.to_numeric(
        df["total_charges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
    miss = df["total_charges"].isna()
    df.loc[miss, "total_charges"] = df.loc[miss, "tenure"] * df.loc[miss, "monthly_charges"]

    df["senior_citizen"] = (df["senior_citizen"].replace({"Yes": 1, "No": 0})
                            .astype(float).astype(int))
    df["tenure"] = df["tenure"].astype(int)
    for c in NO_SERVICE_COLS:  # "No internet/phone service" is just "No" for the add-on
        df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    if "churn" in df:
        df["churn"] = df["churn"].replace({"Yes": 1, "No": 0}).astype(int)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "duplicate_ids": int(df["customer_id"].duplicated().sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v},
        "churn_rate": round(float(df["churn"].mean()), 4) if "churn" in df else None,
        "negative_charges": int((df["monthly_charges"] < 0).sum()),
    }
