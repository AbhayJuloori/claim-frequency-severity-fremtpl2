"""FastAPI backend: pure premium predictions, holdout segment tables, verified notes."""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pricing import data as D

ART = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
app = FastAPI(title="Claim frequency & severity pricing API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class Policy(BaseModel):
    Area: Literal["A", "B", "C", "D", "E", "F"]
    VehBrand: str = Field(examples=["B12"])
    VehGas: Literal["Regular", "Diesel"]
    Region: str = Field(examples=["R11"])
    VehPower: int = Field(ge=4, le=15)
    VehAge: int = Field(ge=0, le=100)
    DrivAge: int = Field(ge=18, le=100)
    BonusMalus: int = Field(ge=50, le=230)
    Density: int = Field(ge=1, le=30000)


@lru_cache
def models() -> dict:
    return joblib.load(ART / "models.joblib")


@lru_cache
def artifact(name: str):
    return json.loads((ART / name).read_text())


@app.get("/health")
def health() -> dict:
    return {"ok": (ART / "models.joblib").exists()}


@app.get("/metrics")
def metrics() -> dict:
    m = artifact("metrics.json")
    return {k: m[k] for k in ["audit", "n_train", "n_test", "frequency", "severity", "pure_premium", "deciles"]}


@app.get("/segments-all")
def segments_all() -> dict:
    return artifact("segments.json")


@app.get("/segments/{dimension}")
def segments(dimension: str) -> list[dict]:
    segs = artifact("segments.json")
    if dimension not in segs:
        raise HTTPException(404, f"unknown dimension; use one of {sorted(segs)}")
    return segs[dimension]


@app.get("/notes")
def notes() -> list[dict]:
    # Only verified notes carry prose; rejected ones return note = null with the reasons.
    return [{"dimension": n["dimension"], "verified": n["verified"], "note": n["note"], "model": n["model"],
             "problems": n["attempts"][-1]["problems"]} for n in artifact("notes.json")]


@app.post("/predict")
def predict(policies: list[Policy]) -> list[dict]:
    if not policies:
        return []
    df = pd.DataFrame([p.model_dump() for p in policies])
    df["LogDensity"] = np.log(df["Density"].astype(float))
    for c in D.CAT_FEATURES:
        df[c] = pd.Categorical(df[c], categories=models()["lgb_freq"].booster_.pandas_categorical[D.CAT_FEATURES.index(c)])
    for c in ["VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        df[c] = df[c].astype(float)
    out = []
    for i in range(len(df)):
        row = df.iloc[[i]][D.FEATURES]
        res = {}
        for kind in ["glm", "lgb"]:
            f = float(np.clip(models()[f"{kind}_freq"].predict(row)[0], 1e-6, None))
            s = float(np.clip(models()[f"{kind}_sev"].predict(row)[0], 1.0, None))
            res[kind] = {"frequency": round(f, 5), "severity": round(s, 2), "pure_premium": round(f * s, 2)}
        out.append(res)
    return out
