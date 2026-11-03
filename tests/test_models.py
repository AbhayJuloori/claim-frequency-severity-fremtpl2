import numpy as np
import pandas as pd
import pytest

from pricing import data as D, models as M
from tests.conftest import fake_raw


def test_prepare_applies_cleaning_rules():
    freq, sev = fake_raw()
    freq.loc[0, ["ClaimNb", "Exposure"]] = [11, 2.0]
    df, audit = D.prepare(freq, sev)
    assert audit["claims_without_amount_policies"] >= 1
    assert audit["sev_policies_missing_from_freq"] == 1
    assert df["ClaimNb"].max() <= 4 and df["Exposure"].max() <= 1
    assert ((df.ClaimNb > 0) == (df.ClaimAmount > 0)).all()
    assert df["VehGas"].isin(["Regular", "Diesel"]).all()


def test_split_has_no_policy_in_both():
    df, _ = D.prepare(*fake_raw())
    tr, te = D.split(df)
    assert not set(tr.IDpol) & set(te.IDpol) and len(tr) + len(te) == len(df)


def test_gini_perfect_random_and_reversed():
    rng = np.random.default_rng(1)
    y = rng.poisson(0.2, 5000).astype(float)
    w = np.ones_like(y)
    assert M.gini(y, y, w) > 0.5
    assert abs(M.gini(y, rng.random(5000), w)) < 0.05
    assert M.gini(y, -y, w) < -0.5


def test_deciles_are_equal_weight_and_preserve_totals():
    rng = np.random.default_rng(2)
    p = rng.gamma(2, 0.05, 1000)
    w = rng.uniform(0.1, 1, 1000)
    d = M.deciles(p, p, w)
    assert len(d) == 10
    assert np.allclose(d.observed, d.predicted)
    assert d.weight.sum() == pytest.approx(w.sum())


def test_models_beat_dummy_and_calibrate(artifacts):
    import json
    m = json.loads((artifacts / "metrics.json").read_text())
    assert m["frequency"]["glm"]["poisson_deviance"] < m["frequency"]["dummy"]["poisson_deviance"]
    assert m["frequency"]["glm"]["gini"] > 0.1
    for kind in ["glm", "lgb"]:
        assert 0.8 < m["pure_premium"][kind]["ratio_pred_to_obs"] < 1.25
