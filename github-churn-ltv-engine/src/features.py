"""Feature engineering + categorical encoding / scaling."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADDONS = ["online_security", "online_backup", "device_protection", "tech_support",
          "streaming_tv", "streaming_movies"]

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges", "avg_monthly_spend",
                    "spend_vs_base_ratio", "num_addon_services", "charge_per_service",
                    "senior_citizen", "is_autopay", "has_family"]
CATEGORICAL_FEATURES = ["gender", "partner", "dependents", "phone_service", "multiple_lines",
                        "internet_service", "online_security", "online_backup",
                        "device_protection", "tech_support", "streaming_tv", "streaming_movies",
                        "contract", "paperless_billing", "payment_method", "tenure_bucket"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tenure = df["tenure"].clip(lower=1)
    # average monthly usage (billed) vs. base plan charge
    df["avg_monthly_spend"] = df["total_charges"] / tenure
    df["spend_vs_base_ratio"] = df["avg_monthly_spend"] / df["monthly_charges"].replace(0, np.nan)
    df["spend_vs_base_ratio"] = df["spend_vs_base_ratio"].fillna(1.0)
    df["num_addon_services"] = (df[ADDONS] == "Yes").sum(axis=1)
    df["charge_per_service"] = df["monthly_charges"] / (1 + df["num_addon_services"])
    df["is_autopay"] = df["payment_method"].str.contains("automatic", case=False).astype(int)
    df["has_family"] = ((df["partner"] == "Yes") | (df["dependents"] == "Yes")).astype(int)
    df["tenure_bucket"] = pd.cut(df["tenure"], [-1, 12, 24, 48, np.inf],
                                 labels=["0-12m", "13-24m", "25-48m", "49m+"]).astype(str)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Numeric -> StandardScaler, categorical -> dense One-Hot (unknowns ignored)."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
