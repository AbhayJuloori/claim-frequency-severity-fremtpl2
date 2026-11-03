"""Load freMTPL2 frequency and severity tables and build one policy level frame."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 0
CAT_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
NUM_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
FEATURES = CAT_FEATURES + NUM_FEATURES


def download(data_dir: str = "data") -> None:
    from sklearn.datasets import fetch_openml

    out = Path(data_dir)
    out.mkdir(exist_ok=True)
    fetch_openml(data_id=41214, as_frame=True, parser="pandas").frame.to_parquet(out / "freq.parquet")
    fetch_openml(data_id=41215, as_frame=True, parser="pandas").frame.to_parquet(out / "sev.parquet")


def prepare(freq: pd.DataFrame, sev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join claim amounts onto policies and apply the cleaning rules, returning (df, audit)."""
    df = freq.copy()
    df["IDpol"] = df["IDpol"].astype(np.int64)
    sev = sev.assign(IDpol=sev["IDpol"].astype(np.int64))
    amounts = sev.groupby("IDpol")["ClaimAmount"].sum()
    audit = {
        "policies": int(len(df)),
        "claims_in_freq": int(df["ClaimNb"].sum()),
        "claim_rows_in_sev": int(len(sev)),
        "sev_policies_missing_from_freq": int((~amounts.index.isin(df["IDpol"])).sum()),
    }
    df["ClaimAmount"] = df["IDpol"].map(amounts).fillna(0.0)
    # Policies that report claims but have no amount in the severity table cannot be
    # used for severity or pure premium; treat them as claim free (as the reference
    # scikit-learn study does) and count how many.
    no_amount = (df["ClaimNb"] > 0) & (df["ClaimAmount"] == 0)
    audit["claims_without_amount_policies"] = int(no_amount.sum())
    df.loc[no_amount, "ClaimNb"] = 0
    # A handful of policies report 5+ claims in a year; cap at 4. Exposure over a year is a
    # data error; cap at 1. Very large single losses are capped at 200K for severity.
    audit["claimnb_over_4"] = int((df["ClaimNb"] > 4).sum())
    audit["exposure_over_1"] = int((df["Exposure"] > 1).sum())
    audit["amount_over_200k"] = int((df["ClaimAmount"] > 200_000).sum())
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200_000)
    df["LogDensity"] = np.log(df["Density"].astype(float))
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str).str.strip("'").astype("category")
    for c in ["VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        df[c] = df[c].astype(float)
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["AvgClaimAmount"] = np.where(df["ClaimNb"] > 0, df["ClaimAmount"] / df["ClaimNb"].clip(lower=1), 0.0)
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
    audit["claims_after_cleaning"] = int(df["ClaimNb"].sum())
    audit["exposure_years"] = float(df["Exposure"].sum())
    return df, audit


def split(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Policy level 75/25 holdout. Each row is one policy, so no policy is in both."""
    return train_test_split(df, test_size=0.25, random_state=seed)


def load(data_dir: str = "data") -> tuple[pd.DataFrame, dict]:
    d = Path(data_dir)
    return prepare(pd.read_parquet(d / "freq.parquet"), pd.read_parquet(d / "sev.parquet"))
