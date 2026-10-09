"""Features that only become possible with the FULL flight schedule (not a sample).

Compares, on the same 1M training flights and the same test set (all Oct-Dec 2017):
  A. weather model (current)
  B. + typical congestion   - usual number of scheduled departures/arrivals at the airport
                              for that weekday + hour (a lookup table, so it works for any future date)
  C. + same-day congestion  - actual number scheduled that day/hour (needs that day's full schedule)
  D. + aircraft rotation    - which leg of the plane's day this is and the scheduled turnaround
                              since its previous flight (needs the tail number's schedule)

Run:  python src/feature_experiments.py      (needs the same files as src/learning_curve.py)
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import CATEGORICAL, FEATURES, TARGET, build_features
from learning_curve import BASE
from weather import WEATHER_FEATURES, add_weather, prepare_weather

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TRAIN_ROWS = 1_000_000


def hhmm_to_min(s):
    s = s.astype(int).clip(0, 2400) % 2400
    return (s // 100) * 60 + s % 100


def congestion(raw: pd.DataFrame) -> pd.DataFrame:
    """Scheduled departures at origin in the departure hour / arrivals at dest in the arrival hour."""
    dep_h = raw["CRSDepTime"].astype(int).clip(0, 2359) // 100
    arr_h = (raw["CRSArrTime"].astype(int).clip(0, 2400) % 2400) // 100
    dow = pd.to_datetime(raw["FlightDate"]).dt.dayofweek
    out = pd.DataFrame(index=raw.index)
    # same-day counts
    out["o_deps_hour_day"] = raw.groupby([raw.Origin, raw.FlightDate, dep_h])["Origin"].transform("size")
    out["d_arrs_hour_day"] = raw.groupby([raw.Dest, raw.FlightDate, arr_h])["Dest"].transform("size")
    out["o_deps_day"] = raw.groupby([raw.Origin, raw.FlightDate])["Origin"].transform("size")
    # typical counts: average of the same-day counts over all dates with that weekday
    n_days = raw.groupby(dow)["FlightDate"].transform("nunique")
    out["o_deps_hour_typ"] = raw.groupby([raw.Origin, dow, dep_h])["Origin"].transform("size") / n_days
    out["d_arrs_hour_typ"] = raw.groupby([raw.Dest, dow, arr_h])["Dest"].transform("size") / n_days
    out["o_deps_day_typ"] = raw.groupby([raw.Origin, dow])["Origin"].transform("size") / n_days
    return out.astype("float32")


def rotation(raw: pd.DataFrame) -> pd.DataFrame:
    """Leg number of the aircraft's day and scheduled minutes since its previous scheduled arrival."""
    r = raw[["TailNum", "FlightDate", "CRSDepTime", "CRSArrTime"]].copy()
    r["dep_m"], r["arr_m"] = hhmm_to_min(r.CRSDepTime), hhmm_to_min(r.CRSArrTime)
    r = r[r.TailNum.notna()].sort_values(["TailNum", "FlightDate", "dep_m"])
    g = r.groupby(["TailNum", "FlightDate"], sort=False)
    r["leg_of_day"] = g.cumcount() + 1
    r["legs_today"] = g["dep_m"].transform("size")
    r["turn_min"] = r["dep_m"] - g["arr_m"].shift(1)  # previous leg lands at this flight's origin -> same time zone
    r.loc[r["turn_min"] < 0, "turn_min"] = np.nan    # overnight / bad data
    out = pd.DataFrame(index=raw.index, columns=["leg_of_day", "legs_today", "turn_min"], dtype="float32")
    out.loc[r.index] = r[["leg_of_day", "legs_today", "turn_min"]].astype("float32").values
    return out.astype("float32")


def main():
    t0 = time.time()
    raw = pd.read_parquet(ROOT / "data" / "flights_2017_full.parquet")
    for c in ("Carrier", "Origin", "Dest"):
        raw[c] = raw[c].astype(str)
    y = raw[TARGET].values.astype(np.int8)
    X = build_features(raw)
    X = add_weather(X, raw, prepare_weather(pd.read_csv(ROOT / "data" / "weather_2017.csv.gz")))
    wx = [f for f in WEATHER_FEATURES if X.loc[X.Month <= 9, f].nunique(dropna=True) > 1]
    X = pd.concat([X[FEATURES + wx], congestion(raw), rotation(raw)], axis=1)
    num = [c for c in X.columns if c not in CATEGORICAL]
    X[num] = X[num].astype("float32")
    for c in CATEGORICAL:
        X[c] = X[c].astype(pd.CategoricalDtype(sorted(X.loc[X.Month <= 9, c].unique())))
    print(f"features built in {(time.time()-t0)/60:.1f} min", flush=True)

    month = X["Month"].values
    rng = np.random.RandomState(SEED)  # same split as learning_curve.py
    val = rng.choice(np.where(month == 9)[0], 150_000, replace=False)
    pool = np.setdiff1d(np.where(month <= 9)[0], val)
    rng.shuffle(pool)
    idx, te = np.sort(pool[:TRAIN_ROWS]), np.where(month >= 10)[0]

    base = FEATURES + wx
    typ = ["o_deps_hour_typ", "d_arrs_hour_typ", "o_deps_day_typ"]
    day = ["o_deps_hour_day", "d_arrs_hour_day", "o_deps_day"]
    rot = ["leg_of_day", "legs_today", "turn_min"]
    sets = [
        ("A. Weather model (current)", base, "yes"),
        ("B. + typical congestion", base + typ, "yes (lookup table)"),
        ("C. + same-day congestion", base + typ + day, "needs that day's schedule"),
        ("D. + aircraft rotation", base + typ + rot, "needs the plane's schedule"),
    ]
    results = []
    for name, cols, deployable in sets:
        t = time.time()
        m = lgb.LGBMClassifier(**BASE).fit(X.iloc[idx][cols], y[idx], eval_set=[(X.iloc[val][cols], y[val])],
                                           eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict_proba(X.iloc[te][cols])[:, 1]
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
        imp = (imp / imp.sum()).sort_values(ascending=False)
        r = {"name": name, "deployable": deployable, "roc_auc": round(float(roc_auc_score(y[te], p)), 4),
             "pr_auc": round(float(average_precision_score(y[te], p)), 4), "trees": int(m.best_iteration_),
             "top_new_features": {k: round(float(v), 4) for k, v in imp.items() if k not in base}}
        results.append(r)
        print(r, f"({(time.time()-t)/60:.1f} min)", flush=True)
    (ROOT / "reports" / "feature_experiments.json").write_text(json.dumps(
        {"train_rows": int(len(idx)), "test_rows": int(len(te)), "results": results}, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
