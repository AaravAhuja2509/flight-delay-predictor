"""Train the two DEPLOYED models from the full 2017 dataset (5.6M flights).

The learning curve (src/learning_curve.py) showed test AUC peaks around 1M training rows and
does not improve beyond that, so each model is:
  1. fit on 1M Jan-Sep flights with early stopping on a September slice -> tree count, threshold,
     honest test score on ALL Oct-Dec flights
  2. refit on 1.15M flights drawn from all 12 months with that tree count (+15%) -> deployed model
The route/airline list for the app comes from all 5.6M flights (routes with 30+ flights).

Needs:  data/flights_2017_full.parquet, data/weather_2017.csv.gz
Run:    python src/train_full.py        (~15 min)
Writes: models/model.joblib, models/model_weather.joblib, models/meta.json, reports/metrics_full.json
"""
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import CATEGORICAL, FEATURES, TARGET, build_features
from learning_curve import BASE
from train import best_threshold
from weather import WEATHER_FEATURES, add_weather, prepare_weather

ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
SEED = 42


def main():
    t0 = time.time()
    raw = pd.read_parquet(ROOT / "data" / "flights_2017_full.parquet")
    for c in ("Carrier", "Origin", "Dest"):
        raw[c] = raw[c].astype(str)
    y = raw[TARGET].values.astype(np.int8)
    X = add_weather(build_features(raw), raw, prepare_weather(pd.read_csv(ROOT / "data" / "weather_2017.csv.gz")))
    wx = [f for f in WEATHER_FEATURES if X.loc[X.Month <= 9, f].nunique(dropna=True) > 1]
    num = [c for c in FEATURES + wx if c not in CATEGORICAL]
    X[num] = X[num].astype("float32")
    cats = {c: pd.CategoricalDtype(sorted(X[c].unique())) for c in CATEGORICAL}  # all airports/airlines seen in 2017
    for c, t in cats.items():
        X[c] = X[c].astype(t)

    month = X["Month"].values
    rng = np.random.RandomState(SEED)
    val = rng.choice(np.where(month == 9)[0], 150_000, replace=False)
    pool = np.setdiff1d(np.where(month <= 9)[0], val)
    rng.shuffle(pool)
    fit_idx, te = np.sort(pool[:1_000_000]), np.where(month >= 10)[0]
    every = np.arange(len(X))
    rng.shuffle(every)
    final_idx = np.sort(every[:1_150_000])

    report = {"test_rows": int(len(te)), "models": {}}
    for name, cols, path in (("schedule", FEATURES, "model.joblib"), ("weather", FEATURES + wx, "model_weather.joblib")):
        t = time.time()
        m = lgb.LGBMClassifier(**BASE).fit(X.iloc[fit_idx][cols], y[fit_idx], eval_set=[(X.iloc[val][cols], y[val])],
                                           eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        n = int(m.best_iteration_ or BASE["n_estimators"])
        thr = best_threshold(y[val], m.predict_proba(X.iloc[val][cols])[:, 1])
        p = m.predict_proba(X.iloc[te][cols])[:, 1]
        res = {"roc_auc": round(float(roc_auc_score(y[te], p)), 4), "pr_auc": round(float(average_precision_score(y[te], p)), 4),
               "trees": n, "threshold": round(thr, 4)}
        final = lgb.LGBMClassifier(**{**BASE, "n_estimators": int(n * 1.15)}).fit(X.iloc[final_idx][cols], y[final_idx])
        joblib.dump({"model": final, "cat_types": cats, "threshold": thr, "features": cols}, MODELS / path)
        report["models"][name] = res
        print(name, res, f"({(time.time()-t)/60:.1f} min)", flush=True)

    g = raw.groupby(["Origin", "Dest", "Carrier"]).size().reset_index(name="n")
    g = g[g.n >= 30]
    stats = raw.groupby(["Origin", "Dest"]).agg(Distance=("Distance", "median"), CRSElapsedTime=("CRSElapsedTime", "median"))
    keys = g[["Origin", "Dest"]].drop_duplicates()
    meta = {
        "routes": {f"{o}-{d}": {"distance": float(stats.loc[(o, d), "Distance"]), "elapsed": float(stats.loc[(o, d), "CRSElapsedTime"])}
                   for o, d in keys.itertuples(index=False)},
        "route_carriers": {k: sorted(v.Carrier.tolist()) for k, v in g.assign(k=g.Origin + "-" + g.Dest).groupby("k")},
        "threshold": report["models"]["schedule"]["threshold"],
        "base_rate": float(y.mean()),
        "trained_on": "BTS 2017, all 5.6M non-cancelled flights",
    }
    (MODELS / "meta.json").write_text(json.dumps(meta))
    report["routes"], report["route_airline_pairs"] = len(meta["routes"]), int(len(g))
    (REPORTS / "metrics_full.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1), f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
