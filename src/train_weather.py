"""Round 2: add hourly weather at origin (departure hour) and destination (arrival hour).

Same split and same LightGBM settings as round 1 so the comparison is fair:
    train = Jan-Sep 2017, early-stopping validation = Sep, test = Oct-Dec.

Needs:  data/flight_delay_data.csv  and  data/weather_2017.csv.gz  (scripts/fetch_weather.py)
Run:    python src/train_weather.py
Writes: models/model_weather.joblib, reports/metrics_weather.json, reports/results_weather.png
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
from train import best_threshold, evaluate
from weather import WEATHER_FEATURES, add_weather, prepare_weather

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flight_delay_data.csv"
WEATHER = ROOT / "data" / "weather_2017.csv.gz"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEED = 42

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
              cat_smooth=50, min_data_per_group=200, n_estimators=3000, random_state=SEED, n_jobs=-1, verbose=-1)


def to_lgb(X, cat_types):
    X = X.copy()
    for c, t in cat_types.items():
        X[c] = X[c].astype(t)
    return X


def fit_eval(name, X, y, masks, cat_types):
    """Early-stop on Sep, refit on Jan-Sep with the best tree count, score Oct-Dec."""
    fit, val, tr, te = masks
    t = time.time()
    Xl = to_lgb(X, cat_types)
    m = lgb.LGBMClassifier(**PARAMS).fit(Xl[fit], y[fit], eval_set=[(Xl[val], y[val])], eval_metric="auc",
                                         callbacks=[lgb.early_stopping(50, verbose=False)])
    best_n = int(m.best_iteration_ or PARAMS["n_estimators"])
    thr = best_threshold(y[val], m.predict_proba(Xl[val])[:, 1])
    m = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": best_n}).fit(Xl[tr], y[tr])
    p = m.predict_proba(Xl[te])[:, 1]
    res = evaluate(f"{name} ({best_n} trees)", y[te], p, thr, time.time() - t)
    print(res, flush=True)
    return res, p, best_n, thr


def main():
    raw = pd.read_csv(DATA)
    y = raw[TARGET].astype(int).values
    sched = build_features(raw)

    print("preparing weather ...", flush=True)
    w = prepare_weather(pd.read_csv(WEATHER))
    Xw = add_weather(sched, raw, w)
    o_cov, d_cov = Xw["o_temp"].notna().mean(), Xw["d_temp"].notna().mean()
    print(f"weather coverage: origin {o_cov:.1%}  destination {d_cov:.1%}", flush=True)

    month = sched["Month"].values
    masks = (month <= 8, month == 9, month <= 9, month >= 10)
    te = masks[3]
    cat_tr = {c: pd.CategoricalDtype(sorted(sched.loc[masks[2], c].unique())) for c in CATEGORICAL}

    # The archive (reanalysis) never reports fog / thunderstorm / freezing-rain codes, so those
    # flags are always 0 in training. A constant column teaches the model nothing; drop it.
    wx = [f for f in WEATHER_FEATURES if Xw.loc[masks[2], f].nunique(dropna=True) > 1]
    dropped = sorted(set(WEATHER_FEATURES) - set(wx))
    print("constant in training, dropped:", dropped, flush=True)
    origin_only = FEATURES + [f for f in wx if f.startswith("o_")]
    runs = [
        ("Schedule only", FEATURES),
        ("+ origin weather", origin_only),
        ("+ origin & destination weather", FEATURES + wx),
    ]
    results, scores, best = [], {}, None
    for name, cols in runs:
        res, p, n, thr = fit_eval(name, Xw[cols], y, masks, cat_tr)
        results.append(res)
        scores[name] = p
        best = (cols, n, thr)  # last run = full weather model

    # How much does weather help on bad-weather days specifically?
    bad = (Xw["o_snow"].fillna(0) > 0) | (Xw["o_precip_3h"].fillna(0) >= 2) | (Xw["o_gust"].fillna(0) >= 60)
    bad_te = bad.values[te]
    split = {}
    for name in (runs[0][0], runs[-1][0]):
        for label, msk in (("bad_weather", bad_te), ("normal_weather", ~bad_te)):
            split.setdefault(label, {})[name] = round(float(roc_auc_score(y[te][msk], scores[name][msk])), 4)
    split["bad_weather_share_of_test"] = round(float(bad_te.mean()), 3)
    split["delay_rate_bad_weather"] = round(float(y[te][bad_te].mean()), 3)
    split["delay_rate_normal_weather"] = round(float(y[te][~bad_te].mean()), 3)
    print("\nAUC by weather on departure:", json.dumps(split, indent=1))

    # Deployed weather model: refit on all 12 months
    cols, n, thr = best
    cat_all = {c: pd.CategoricalDtype(sorted(sched[c].unique())) for c in CATEGORICAL}
    final = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": int(n * 1.15)}).fit(to_lgb(Xw[cols], cat_all), y)
    joblib.dump({"model": final, "cat_types": cat_all, "threshold": thr, "features": cols}, MODELS / "model_weather.joblib")

    imp = pd.Series(final.booster_.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).sort_values(ascending=False).round(4)
    (REPORTS / "metrics_weather.json").write_text(json.dumps(
        {"results": results, "by_weather": split, "coverage": {"origin": round(o_cov, 4), "dest": round(d_cov, 4)},
         "dropped_constant_features": dropped,
         "feature_importance_gain": imp.head(25).to_dict()}, indent=2))
    print("\nTop features:\n", imp.head(15))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for name, p in scores.items():
            fpr, tpr, _ = roc_curve(y[te], p)
            ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y[te], p):.3f})")
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[0].set(title="ROC, test Oct-Dec 2017", xlabel="False positive rate", ylabel="True positive rate")
        ax[0].legend(loc="lower right", fontsize=8)
        imp.head(12)[::-1].plot.barh(ax=ax[1])
        ax[1].set(title="Weather model: feature importance (share of gain)")
        plt.tight_layout()
        plt.savefig(REPORTS / "results_weather.png", dpi=130)
    except Exception as e:
        print("plot skipped:", e)

    print("\n" + pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
