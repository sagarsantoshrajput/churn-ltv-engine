"""Business rules: risk bands, LTV tiers, retention priority."""
import numpy as np
import pandas as pd

from src import config

LTV_LABELS = ["Bronze", "Silver", "Gold", "Platinum"]


def risk_band(p: pd.Series) -> pd.Series:
    return pd.Series(np.select([p >= config.RISK_HIGH, p >= config.RISK_MEDIUM],
                               ["High", "Medium"], default="Low"), index=p.index)


def ltv_segment(ltv: pd.Series, edges: list) -> pd.Series:
    """edges = 3 quartile cut-points learned on the training population."""
    idx = np.searchsorted(np.asarray(edges), ltv.to_numpy(), side="right")
    return pd.Series(np.array(LTV_LABELS)[idx], index=ltv.index)


def retention_priority(band: pd.Series, seg: pd.Series) -> pd.Series:
    high_value = seg.isin(["Gold", "Platinum"])
    return pd.Series(np.select(
        [(band == "High") & high_value,
         ((band == "Medium") & high_value) | ((band == "High") & ~high_value)],
        ["P1 - Retain now", "P2 - Nurture"], default="P3 - Monitor"), index=band.index)
