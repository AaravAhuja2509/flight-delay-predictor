"""Day-of prediction: what can we know 1 hour before departure, and how much does it help?

Prediction time is T-60 (T = scheduled departure, origin local clock). A feature is only used
if that information would really exist at T-60:

  inbound aircraft (same tail number, previous leg of the day)
    * prev_dep_delay   its departure delay, only if it has actually departed by T-60
    * prev_arr_delay   its arrival delay,   only if it has actually landed by T-60
    * inbound_slack    minutes between when the inbound plane is expected/actually at the gate
                       and our departure (actual arrival if landed, else scheduled arrival
                       pushed back by its departure delay, else plain scheduled turnaround)
  origin airport right now
    * apt_late_share   share of departures from the origin in the 2 hours before prediction time that
                       left 15+ minutes late, and apt_mean_delay their mean delay

Time zones: the previous leg departed from another airport, so its times are converted to our
origin's clock via its scheduled arrival time (which is in our origin's clock) minus its
scheduled flight time.  Nothing from the flight's own departure or arrival is ever used.

Needs:  a clone of github.com/bharathirajatut/flights-dataset (raw 2017 zips) and data/weather_2017.csv.gz
Run:    python src/dayof_experiment.py --src path/to/flights-dataset [--lead 180] [--only AE]
Writes: reports/dayof_experiment.json
"""
import argparse
import glob
import json
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from feature_experiments import hhmm_to_min
from features import CATEGORICAL, FEATURES, TARGET, build_features
from learning_curve import BASE
from weather import WEATHER_FEATURES, add_weather, prepare_weather

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LEAD = 60  # minutes before scheduled departure (override with --lead)
COLS = ["FlightDate", "Month", "Carrier", "TailNum", "FlightNum", "Origin", "Dest", "CRSDepTime", "CRSArrTime",
        "CRSElapsedTime", "Distance", "DepDelay", "ArrDelay", "ArrDel15", "Cancelled", "Diverted"]


def load_raw(src):
    parts = []
    for z in sorted(glob.glob(f"{src}/2017/*.zip")):
        with zipfile.ZipFile(z) as zf:
            parts.append(pd.read_csv(zf.open(next(n for n in zf.namelist() if n.endswith(".csv"))),
                                     usecols=COLS, low_memory=False))
    df = pd.concat(parts, ignore_index=True)
    df = df[df.CRSElapsedTime.notna()].reset_index(drop=True)
    for c in ("Carrier", "Origin", "Dest"):
        df[c] = df[c].astype(str)
    return df


def inbound_features(df):
    """Previous leg of the same aircraft, as known at T-60. Uses ALL scheduled flights incl. cancelled."""
    d = df[["TailNum", "FlightDate", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "DepDelay", "ArrDelay", "Cancelled"]].copy()
    d["T"] = hhmm_to_min(d.CRSDepTime).astype(float)       # our departure, origin clock
    d["sarr"] = hhmm_to_min(d.CRSArrTime).astype(float)    # our arrival, destination clock
    d = d[d.TailNum.notna()].sort_values(["TailNum", "FlightDate", "T"])
    g = d.groupby(["TailNum", "FlightDate"], sort=False)
    p_sarr = g["sarr"].shift(1)           # previous leg's scheduled arrival = at OUR origin, in our origin's clock
    p_elapsed = g["CRSElapsedTime"].shift(1)
    p_depdel, p_arrdel = g["DepDelay"].shift(1), g["ArrDelay"].shift(1)
    p_cancel = g["Cancelled"].shift(1)
    has_prev = p_sarr.notna() & (p_sarr <= d["T"]) & (p_cancel == 0)   # same-day previous leg that was flown
    cutoff = d["T"] - LEAD
    p_actual_dep = (p_sarr - p_elapsed) + p_depdel         # converted to our origin's clock
    p_actual_arr = p_sarr + p_arrdel
    departed = has_prev & p_depdel.notna() & (p_actual_dep <= cutoff)
    landed = has_prev & p_arrdel.notna() & (p_actual_arr <= cutoff)
    out = pd.DataFrame(index=d.index)
    out["prev_dep_delay"] = p_depdel.where(departed)
    out["prev_arr_delay"] = p_arrdel.where(landed)
    expected_in = np.where(landed, p_actual_arr, np.where(departed, p_sarr + p_depdel.clip(lower=0), p_sarr))
    out["inbound_slack"] = np.where(has_prev, d["T"] - expected_in, np.nan)
    out["inbound_state"] = np.select([landed, departed, has_prev], [3, 2, 1], 0)  # 0 first leg, 1 scheduled, 2 airborne, 3 landed
    return out.reindex(df.index).astype("float32")


def airport_state(df):
    """Departures from the origin that actually left in [T-180, T-60): share 15+ min late, mean delay."""
    d = df[["Origin", "FlightDate", "CRSDepTime", "DepDelay"]].copy()
    d["T"] = hhmm_to_min(d.CRSDepTime)
    gid = d.groupby(["Origin", "FlightDate"], sort=False).ngroup().values.astype(np.int64)
    flown = d.DepDelay.notna().values
    act = (d["T"] + d.DepDelay.fillna(0)).clip(-1000, 4000).values.astype(np.int64)
    OFF, SPAN = 2000, 10_000
    key = gid[flown] * SPAN + OFF + act[flown]
    order = np.argsort(key, kind="stable")
    key = key[order]
    late = (d.DepDelay.values[flown][order] >= 15).astype(np.int64)
    dly = d.DepDelay.values[flown][order].astype(float)
    c_late, c_dly = np.concatenate([[0], np.cumsum(late)]), np.concatenate([[0.0], np.cumsum(dly)])
    lo = np.searchsorted(key, gid * SPAN + OFF + d["T"].values - LEAD - 120, side="left")  # 2h window before prediction time
    hi = np.searchsorted(key, gid * SPAN + OFF + d["T"].values - LEAD, side="left")
    n = hi - lo
    out = pd.DataFrame(index=df.index)
    out["apt_recent_deps"] = n.astype("float32")
    with np.errstate(invalid="ignore", divide="ignore"):
        out["apt_late_share"] = np.where(n > 0, (c_late[hi] - c_late[lo]) / n, np.nan).astype("float32")
        out["apt_mean_delay"] = np.where(n > 0, (c_dly[hi] - c_dly[lo]) / n, np.nan).astype("float32")
    return out


def rotation_sched(df):
    d = df[["TailNum", "FlightDate", "CRSDepTime", "CRSArrTime"]].copy()
    d["dep_m"], d["arr_m"] = hhmm_to_min(d.CRSDepTime), hhmm_to_min(d.CRSArrTime)
    d = d[d.TailNum.notna()].sort_values(["TailNum", "FlightDate", "dep_m"])
    g = d.groupby(["TailNum", "FlightDate"], sort=False)
    out = pd.DataFrame(index=d.index)
    out["leg_of_day"] = g.cumcount() + 1
    turn = d["dep_m"] - g["arr_m"].shift(1)
    out["turn_min"] = turn.where(turn >= 0)
    return out.reindex(df.index).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--lead", type=int, default=60, help="minutes before departure the prediction is made")
    ap.add_argument("--only", help="run only feature sets whose name starts with these letters, e.g. AE")
    a = ap.parse_args()
    global LEAD
    LEAD = a.lead
    t0 = time.time()
    allf = load_raw(a.src)
    print(f"raw rows incl. cancelled/diverted: {len(allf):,}", flush=True)
    extra = pd.concat([inbound_features(allf), airport_state(allf), rotation_sched(allf)], axis=1)

    keep = (allf.Cancelled == 0) & (allf.Diverted == 0) & allf.ArrDel15.notna()
    raw, extra = allf[keep].reset_index(drop=True), extra[keep.values].reset_index(drop=True)
    y = raw[TARGET].values.astype(np.int8)
    X = add_weather(build_features(raw), raw, prepare_weather(pd.read_csv(ROOT / "data" / "weather_2017.csv.gz")))
    wx = [f for f in WEATHER_FEATURES if X.loc[X.Month <= 9, f].nunique(dropna=True) > 1]
    X = pd.concat([X[FEATURES + wx], extra], axis=1)
    num = [c for c in X.columns if c not in CATEGORICAL]
    X[num] = X[num].astype("float32")
    for c in CATEGORICAL:
        X[c] = X[c].astype(pd.CategoricalDtype(sorted(X.loc[X.Month <= 9, c].unique())))
    print(f"features built in {(time.time()-t0)/60:.1f} min; flights {len(X):,}", flush=True)
    cov = {k: round(float(X[k].notna().mean()), 3) for k in ["prev_dep_delay", "prev_arr_delay", "inbound_slack", "apt_late_share"]}
    print("coverage:", cov, flush=True)

    month = X["Month"].values
    rng = np.random.RandomState(SEED)
    val = rng.choice(np.where(month == 9)[0], 150_000, replace=False)
    pool = np.setdiff1d(np.where(month <= 9)[0], val)
    rng.shuffle(pool)
    idx, te = np.sort(pool[:1_000_000]), np.where(month >= 10)[0]

    base = FEATURES + wx
    rot = ["leg_of_day", "turn_min"]
    inb = ["prev_dep_delay", "prev_arr_delay", "inbound_slack", "inbound_state"]
    apt = ["apt_recent_deps", "apt_late_share", "apt_mean_delay"]
    sets = [("A. Current app model (schedule + weather)", base),
            ("B. + aircraft rotation (schedule)", base + rot),
            ("C. + inbound aircraft status at T-60", base + rot + inb),
            ("D. + origin airport status at T-60", base + rot + apt),
            ("E. All day-of features", base + rot + inb + apt)]
    if a.only:
        sets = [s_ for s_ in sets if s_[0][0] in a.only]
    results = []
    for name, cols in sets:
        t = time.time()
        m = lgb.LGBMClassifier(**BASE).fit(X.iloc[idx][cols], y[idx], eval_set=[(X.iloc[val][cols], y[val])],
                                           eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict_proba(X.iloc[te][cols])[:, 1]
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
        imp = (imp / imp.sum()).sort_values(ascending=False)
        r = {"name": name, "roc_auc": round(float(roc_auc_score(y[te], p)), 4),
             "pr_auc": round(float(average_precision_score(y[te], p)), 4), "trees": int(m.best_iteration_),
             "top_features": {k: round(float(v), 4) for k, v in imp.head(6).items()}}
        if name.startswith("E"):
            st = X.iloc[te]["inbound_state"].values
            r["auc_by_inbound_state"] = {lbl: round(float(roc_auc_score(y[te][st == s], p[st == s])), 4)
                                         for s, lbl in [(0, "first flight of the day"), (1, "inbound not departed yet"),
                                                        (2, "inbound in the air"), (3, "inbound landed")] if (st == s).sum() > 1000}
            r["share_by_inbound_state"] = {int(s): round(float((st == s).mean()), 3) for s in range(4)}
        results.append(r)
        print(r, f"({(time.time()-t)/60:.1f} min)", flush=True)
    out_name = "dayof_experiment.json" if LEAD == 60 and not a.only else f"dayof_experiment_lead{LEAD}.json"
    (ROOT / "reports" / out_name).write_text(json.dumps(
        {"lead_minutes": LEAD, "train_rows": int(len(idx)), "test_rows": int(len(te)), "coverage": cov, "results": results}, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
