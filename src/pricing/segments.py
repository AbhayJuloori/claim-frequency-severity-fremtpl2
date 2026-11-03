"""Holdout segment table: observed vs predicted frequency and pure premium by rating factor."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pricing import data as D

BANDS = {
    "DrivAgeBand": ("DrivAge", [17, 21, 26, 31, 41, 51, 61, 71, 101], ["18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"]),
    "VehAgeBand": ("VehAge", [-1, 0, 2, 5, 10, 15, 101], ["0", "1-2", "3-5", "6-10", "11-15", "16+"]),
    "BonusMalusBand": ("BonusMalus", [49, 50, 60, 80, 100, 231], ["50", "51-60", "61-80", "81-100", "101+"]),
}
DIMENSIONS = ["Area", "VehGas", "Region", "DrivAgeBand", "VehAgeBand", "BonusMalusBand"]


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, (col, edges, labels) in BANDS.items():
        df[name] = pd.cut(df[col], bins=edges, labels=labels, right=True).astype(str)
    return df


def predict(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for kind in ["glm", "lgb"]:
        out[f"{kind}_freq"] = np.clip(models[f"{kind}_freq"].predict(df[D.FEATURES]), 1e-6, None)
        out[f"{kind}_sev"] = np.clip(models[f"{kind}_sev"].predict(df[D.FEATURES]), 1.0, None)
        out[f"{kind}_pp"] = out[f"{kind}_freq"] * out[f"{kind}_sev"]
    return out


def segment_table(test: pd.DataFrame, preds: pd.DataFrame, dim: str) -> list[dict]:
    t = add_bands(test).join(preds)
    e = t["Exposure"]
    t = t.assign(w_glm_freq=t.glm_freq * e, w_lgb_freq=t.lgb_freq * e, w_glm_pp=t.glm_pp * e, w_lgb_pp=t.lgb_pp * e)
    g = t.groupby(dim, observed=True).agg(
        policies=("IDpol", "size"), exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"),
        losses=("ClaimAmount", "sum"), w_glm_freq=("w_glm_freq", "sum"), w_lgb_freq=("w_lgb_freq", "sum"),
        w_glm_pp=("w_glm_pp", "sum"), w_lgb_pp=("w_lgb_pp", "sum"))
    overall_pp = t["w_lgb_pp"].sum() / e.sum()
    res = pd.DataFrame({
        "segment": g.index.astype(str), "policies": g.policies, "exposure": g.exposure.round(1),
        "claims": g.claims, "observed_frequency": (g.claims / g.exposure).round(4),
        "glm_frequency": (g.w_glm_freq / g.exposure).round(4), "lgb_frequency": (g.w_lgb_freq / g.exposure).round(4),
        "observed_pure_premium": (g.losses / g.exposure).round(0),
        "glm_pure_premium": (g.w_glm_pp / g.exposure).round(0), "lgb_pure_premium": (g.w_lgb_pp / g.exposure).round(0),
        "lgb_relativity": (g.w_lgb_pp / g.exposure / overall_pp).round(2),
    })
    return res.to_dict(orient="records")


def build(test: pd.DataFrame, models: dict) -> dict:
    preds = predict(models, test)
    return {dim: segment_table(test, preds, dim) for dim in DIMENSIONS}


def main(data_dir: str = "data", art: str = "artifacts") -> dict:
    df, _ = D.load(data_dir)
    ids = pd.read_parquet(Path(art) / "test_ids.parquet")["IDpol"]
    test = df[df["IDpol"].isin(ids)]
    models = joblib.load(Path(art) / "models.joblib")
    segs = build(test, models)
    (Path(art) / "segments.json").write_text(json.dumps(segs, indent=1))
    for dim in ["DrivAgeBand", "BonusMalusBand"]:
        print(pd.DataFrame(segs[dim]).to_string(index=False))
    return segs


if __name__ == "__main__":
    main(*sys.argv[1:])
