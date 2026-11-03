import json

import numpy as np
import pandas as pd
import pytest

from pricing import data as D, models as M, segments as S


def fake_raw(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    bm = rng.choice([50, 50, 50, 60, 80, 100, 120], n)
    drv = rng.integers(18, 85, n)
    exposure = rng.uniform(0.05, 1.2, n)
    lam = 0.06 * np.exp(0.02 * (bm - 50) - 0.01 * (drv - 40)) * np.minimum(exposure, 1)
    claims = rng.poisson(lam)
    freq = pd.DataFrame({
        "IDpol": np.arange(1, n + 1, dtype=float), "ClaimNb": claims, "Exposure": exposure,
        "Area": pd.Categorical(rng.choice(list("ABCDEF"), n)), "VehPower": rng.integers(4, 12, n),
        "VehAge": rng.integers(0, 20, n), "DrivAge": drv, "BonusMalus": bm,
        "VehBrand": pd.Categorical(rng.choice(["B1", "B2", "B12"], n)),
        "VehGas": rng.choice(["'Regular'", "'Diesel'"], n), "Density": rng.integers(10, 5000, n),
        "Region": pd.Categorical(rng.choice(["R11", "R24", "R82"], n)),
    })
    rows = []
    for pid, k in zip(freq.IDpol, freq.ClaimNb):
        rows += [{"IDpol": pid, "ClaimAmount": float(rng.gamma(2, 900))} for _ in range(int(k))]
    sev = pd.DataFrame(rows)
    # One policy with claims but no amount, one amount for an unknown policy.
    with_claim = freq.index[freq.ClaimNb > 0][0]
    sev = sev[sev.IDpol != freq.loc[with_claim, "IDpol"]]
    sev = pd.concat([sev, pd.DataFrame({"IDpol": [999999.0], "ClaimAmount": [100.0]})], ignore_index=True)
    return freq, sev


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    art = tmp_path_factory.mktemp("art")
    df, audit = D.prepare(*fake_raw())
    train, test = D.split(df)
    M.LGB_BASE.update(n_estimators=50)
    models = M.train_all(train)
    import joblib
    joblib.dump(models, art / "models.joblib")
    metrics = {"audit": audit, "n_train": len(train), "n_test": len(test), **M.evaluate(models, train, test)}
    (art / "metrics.json").write_text(json.dumps(metrics))
    (art / "segments.json").write_text(json.dumps(S.build(test, models)))
    (art / "notes.json").write_text(json.dumps([{"dimension": "Area", "verified": False, "note": None, "model": "x",
                                                  "attempts": [{"text": "t", "ok": False, "problems": ["9.9"]}]}]))
    return art
