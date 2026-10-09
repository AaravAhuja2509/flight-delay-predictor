"""How much does more training data help?  AUC vs number of training flights.

Same features (schedule + origin/destination weather) and the same test set for every run:
    train pool = Jan-Sep 2017 minus a held-out September validation slice (early stopping)
    test       = ALL Oct-Dec 2017 flights (~1.4M)

Needs:  data/flights_2017_full.parquet   (python scripts/make_dataset.py --full)
        data/weather_2017.csv.gz         (python scripts/fetch_weather.py)
Run:    python src/learning_curve.py            (about 40 min; add --extra for two more runs)
Writes: reports/learning_curve.json, reports/learning_curve.png
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import CATEGORICAL, FEATURES, TARGET, build_features
from weather import WEATHER_FEATURES, add_weather, prepare_weather

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEED = 42
SIZES = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, None]  # None = everything
BASE = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
            cat_smooth=50, min_data_per_group=200, n_estimators=4000, random_state=SEED, n_jobs=-1, verbose=-1)
BIG = {**BASE, "num_leaves": 255, "min_child_samples": 200}  # more capacity, only sensible with lots of data


def load():
    raw = pd.read_parquet(ROOT / "data" / "flights_2017_full.parquet")
    for c in ("Carrier", "Origin", "Dest"):
        raw[c] = raw[c].astype(str)
    X = build_features(raw)
    w = prepare_weather(pd.read_csv(ROOT / "data" / "weather_2017.csv.gz"))
    X = add_weather(X, raw, w)
    wx = [f for f in WEATHER_FEATURES if X.loc[X.Month <= 9, f].nunique(dropna=True) > 1]  # drop constant flags
    cols = FEATURES + wx
    X = X[cols]
    num = [c for c in cols if c not in CATEGORICAL]
    X[num] = X[num].astype("float32")
    cats = {c: pd.CategoricalDtype(sorted(X.loc[X.Month <= 9, c].unique())) for c in CATEGORICAL}
    for c, t in cats.items():
        X[c] = X[c].astype(t)
    return X, raw[TARGET].values.astype(np.int8), cols


def run(X, y, idx, val, te, params, cols):
    t = time.time()
    m = lgb.LGBMClassifier(**params).fit(X.iloc[idx][cols], y[idx], eval_set=[(X.iloc[val][cols], y[val])],
                                         eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict_proba(X.iloc[te][cols])[:, 1]
    return {"train_rows": int(len(idx)), "trees": int(m.best_iteration_ or params["n_estimators"]),
            "roc_auc": round(float(roc_auc_score(y[te], p)), 4),
            "pr_auc": round(float(average_precision_score(y[te], p)), 4),
            "minutes": round((time.time() - t) / 60, 1)}


def main():
    t0 = time.time()
    X, y, cols = load()
    month = X["Month"].values
    rng = np.random.RandomState(SEED)
    sep = np.where(month == 9)[0]
    val = rng.choice(sep, 150_000, replace=False)
    pool = np.setdiff1d(np.where(month <= 9)[0], val)
    te = np.where(month >= 10)[0]
    rng.shuffle(pool)  # nested subsets: each bigger run contains the smaller ones
    print(f"loaded in {(time.time()-t0)/60:.1f} min | pool {len(pool):,}  val {len(val):,}  test {len(te):,}", flush=True)

    out = {"test_rows": int(len(te)), "test_delay_rate": round(float(y[te].mean()), 4), "curve": [], "extra": []}
    for n in SIZES:
        idx = np.sort(pool[: (n or len(pool))])
        r = run(X, y, idx, val, te, BASE, cols)
        out["curve"].append(r)
        print("weather model", r, flush=True)

    if "--extra" not in sys.argv:  # two slow extra runs (~50 min), off by default
        out["extra"] = []
        (REPORTS / "learning_curve.json").write_text(json.dumps(out, indent=2))
        plot(out)
        return
    # Two extra points: schedule-only with all data, and a bigger model with all data
    all_idx = np.sort(pool)
    r = run(X, y, all_idx, val, te, BASE, FEATURES)
    out["extra"].append({"name": "Schedule only, all data", **r})
    print("schedule only, all data", r, flush=True)
    r = run(X, y, all_idx, val, te, BIG, cols)
    out["extra"].append({"name": "Weather, all data, 255 leaves", **r})
    print("bigger model, all data", r, flush=True)

    (REPORTS / "learning_curve.json").write_text(json.dumps(out, indent=2))
    plot(out)
    print(f"done in {(time.time()-t0)/60:.1f} min")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.DataFrame(out["curve"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(c.train_rows, c.roc_auc, "o-", color="#2a6fdb", label="Weather model (63 leaves)")
    for _, r in c.iterrows():
        ax.annotate(f"{r.roc_auc:.3f}", (r.train_rows, r.roc_auc), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    for e in out.get("extra", []):
        if e["name"].startswith("Weather, all data, 255"):
            ax.plot([e["train_rows"]], [e["roc_auc"]], "D", color="#d9480f", label=f"Bigger model, all data ({e['roc_auc']:.3f})")
    fe = REPORTS / "feature_experiments.json"
    if fe.exists():  # context: what new features do, versus more rows
        rot = json.loads(fe.read_text()).get("leakage_check", {}).get("D_full_schedule")
        if rot:
            ax.axhline(rot["roc_auc"], color="#2f9e44", ls="--", lw=1.2)
            ax.text(c.train_rows.min(), rot["roc_auc"] + 0.001, f"+ aircraft rotation features, 1M rows ({rot['roc_auc']:.3f})",
                    fontsize=8, color="#2f9e44", va="bottom")
    ax.set_ylim(0.67, 0.73)
    ax.axvline(500_000 * 0.75, color="#888", ls=":", lw=1)
    ax.text(500_000 * 0.75, ax.get_ylim()[0], "  old training set\n  (375k rows)", fontsize=7.5, color="#666", va="bottom")
    ax.set_xscale("log")
    ax.set(xlabel="Training flights (log scale)", ylabel="ROC-AUC on all Oct-Dec 2017 flights",
           title="More data barely helps; new features do")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "learning_curve.png", dpi=130)


if __name__ == "__main__":
    main()
