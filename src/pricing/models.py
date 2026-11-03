"""Frequency (Poisson) and severity (gamma) models: GLMs against LightGBM.

Frequency target is ClaimNb / Exposure with Exposure as the sample weight, which is the
same likelihood as a Poisson GLM on ClaimNb with log(Exposure) as an offset.
Severity target is the average claim amount on policies with claims, weighted by ClaimNb.
"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import auc, mean_gamma_deviance, mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from pricing import data as D

LGB_BASE = dict(learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, n_estimators=3000, verbose=-1, random_state=D.SEED)


def glm_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("bins", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile", subsample=None),
         ["VehAge", "DrivAge"]),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
        ("num", StandardScaler(), ["BonusMalus", "LogDensity"]),
    ])


def fit_glm_freq(train: pd.DataFrame):
    m = make_pipeline(glm_preprocessor(), PoissonRegressor(alpha=1e-12, solver="newton-cholesky", max_iter=300))
    return m.fit(train[D.FEATURES], train["Frequency"], poissonregressor__sample_weight=train["Exposure"])


def fit_glm_sev(train: pd.DataFrame):
    pos = train[train["ClaimNb"] > 0]
    m = make_pipeline(glm_preprocessor(), GammaRegressor(alpha=10.0, solver="newton-cholesky", max_iter=300))
    return m.fit(pos[D.FEATURES], pos["AvgClaimAmount"], gammaregressor__sample_weight=pos["ClaimNb"])


def _fit_lgb(x, y, w, objective: str, seed: int = D.SEED) -> lgb.LGBMRegressor:
    # Early stopping on a slice of the training set only; the holdout is never touched.
    xt, xv, yt, yv, wt, wv = train_test_split(x, y, w, test_size=0.1, random_state=seed)
    m = lgb.LGBMRegressor(objective=objective, **LGB_BASE)
    m.fit(xt, yt, sample_weight=wt, eval_set=[(xv, yv)], eval_sample_weight=[wv],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def fit_lgb_freq(train: pd.DataFrame) -> lgb.LGBMRegressor:
    return _fit_lgb(train[D.FEATURES], train["Frequency"], train["Exposure"], "poisson")


def fit_lgb_sev(train: pd.DataFrame) -> lgb.LGBMRegressor:
    pos = train[train["ClaimNb"] > 0]
    return _fit_lgb(pos[D.FEATURES], pos["AvgClaimAmount"], pos["ClaimNb"], "gamma")


# ------------------------------------------------------------------ metrics
def gini(y_true, y_pred, exposure) -> float:
    """Gini from the ordered Lorenz curve: rank by predicted risk, cumulate observed losses."""
    order = np.argsort(y_pred)
    cum_exposure = np.cumsum(exposure[order]) / exposure.sum()
    cum_loss = np.cumsum((y_true * exposure)[order]) / (y_true * exposure).sum()
    cum_exposure = np.concatenate([[0], cum_exposure])
    cum_loss = np.concatenate([[0], cum_loss])
    return float(1 - 2 * auc(cum_exposure, cum_loss))


def deciles(y_true, y_pred, weight) -> pd.DataFrame:
    """Observed vs predicted by decile of predicted risk, equal weight per decile."""
    order = np.argsort(y_pred, kind="stable")
    w = np.asarray(weight)[order]
    edges = np.cumsum(w) / w.sum()
    dec = np.minimum((edges * 10).astype(int), 9)
    frame = pd.DataFrame({"decile": dec + 1, "w": w, "obs": np.asarray(y_true)[order] * w,
                          "pred": np.asarray(y_pred)[order] * w})
    out = frame.groupby("decile").sum()
    return pd.DataFrame({"weight": out.w, "observed": out.obs / out.w, "predicted": out.pred / out.w}).reset_index()


def evaluate(models: dict, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    res = {}
    exp_te = test["Exposure"].to_numpy()
    dummy_freq = np.full(len(test), np.average(train["Frequency"], weights=train["Exposure"]))
    res["frequency"] = {"dummy": {"poisson_deviance": float(mean_poisson_deviance(test["Frequency"], dummy_freq, sample_weight=exp_te))}}
    pos_te = test[test["ClaimNb"] > 0]
    pos_tr = train[train["ClaimNb"] > 0]
    dummy_sev = np.full(len(pos_te), np.average(pos_tr["AvgClaimAmount"], weights=pos_tr["ClaimNb"]))
    res["severity"] = {"dummy": {"gamma_deviance": float(mean_gamma_deviance(pos_te["AvgClaimAmount"], dummy_sev, sample_weight=pos_te["ClaimNb"]))}}
    res["pure_premium"] = {}
    res["deciles"] = {}
    for kind in ["glm", "lgb"]:
        f = np.clip(models[f"{kind}_freq"].predict(test[D.FEATURES]), 1e-6, None)
        s = np.clip(models[f"{kind}_sev"].predict(test[D.FEATURES]), 1.0, None)
        s_pos = np.clip(models[f"{kind}_sev"].predict(pos_te[D.FEATURES]), 1.0, None)
        dev = mean_poisson_deviance(test["Frequency"], f, sample_weight=exp_te)
        res["frequency"][kind] = {
            "poisson_deviance": float(dev),
            "d2": float(1 - dev / res["frequency"]["dummy"]["poisson_deviance"]),
            "gini": gini(test["Frequency"].to_numpy(), f, exp_te),
            "predicted_claims": float((f * exp_te).sum()), "observed_claims": float(test["ClaimNb"].sum()),
        }
        gdev = mean_gamma_deviance(pos_te["AvgClaimAmount"], s_pos, sample_weight=pos_te["ClaimNb"])
        res["severity"][kind] = {
            "gamma_deviance": float(gdev),
            "d2": float(1 - gdev / res["severity"]["dummy"]["gamma_deviance"]),
            "mean_predicted": float(np.average(s_pos, weights=pos_te["ClaimNb"])),
            "mean_observed": float(np.average(pos_te["AvgClaimAmount"], weights=pos_te["ClaimNb"])),
        }
        pp = f * s
        res["pure_premium"][kind] = {
            "gini": gini(test["PurePremium"].to_numpy(), pp, exp_te),
            "predicted_total": float((pp * exp_te).sum()), "observed_total": float(test["ClaimAmount"].sum()),
            "ratio_pred_to_obs": float((pp * exp_te).sum() / test["ClaimAmount"].sum()),
        }
        res["deciles"][f"{kind}_frequency"] = deciles(test["Frequency"], f, exp_te).to_dict(orient="records")
        res["deciles"][f"{kind}_pure_premium"] = deciles(test["PurePremium"], pp, exp_te).to_dict(orient="records")
    return res


def train_all(train: pd.DataFrame) -> dict:
    return {"glm_freq": fit_glm_freq(train), "glm_sev": fit_glm_sev(train),
            "lgb_freq": fit_lgb_freq(train), "lgb_sev": fit_lgb_sev(train)}


def main(data_dir: str = "data", out_dir: str = "artifacts") -> dict:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    df, audit = D.load(data_dir)
    train, test = D.split(df)
    models = train_all(train)
    metrics = {"audit": audit, "n_train": int(len(train)), "n_test": int(len(test)),
               "lgb_trees": {"freq": int(models["lgb_freq"].best_iteration_), "sev": int(models["lgb_sev"].best_iteration_)},
               **evaluate(models, train, test)}
    joblib.dump(models, out / "models.joblib")
    test[["IDpol"]].to_parquet(out / "test_ids.parquet")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ["audit", "lgb_trees", "frequency", "severity", "pure_premium"]}, indent=2))
    return metrics


if __name__ == "__main__":
    main(*sys.argv[1:])
