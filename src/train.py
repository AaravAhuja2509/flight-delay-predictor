"""Train and compare models on schedule-only features (round 1).

Time-based split:  train = Jan-Sep, test = Oct-Dec.
LightGBM early stopping uses September as a validation slice, then refits on Jan-Sep.

Run:  python src/train.py
"""
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features import CATEGORICAL, FEATURES, NUMERIC, TARGET, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "flight_delay_data.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
SEED = 42


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[np.argmax(f1)])


def evaluate(name, y, p, thr, secs):
    pred = (p >= thr).astype(int)
    return {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "threshold": round(thr, 3),
        "train_seconds": round(secs, 1),
    }


def main():
    raw = pd.read_csv(DATA)
    X = build_features(raw)
    y = raw[TARGET].astype(int).values
    month = X["Month"].values

    tr, te = month <= 9, month >= 10
    fit, val = month <= 8, month == 9  # inner split for early stopping / thresholds
    print(f"rows: train={tr.sum():,} test={te.sum():,}  delay rate train={y[tr].mean():.3f} test={y[te].mean():.3f}")

    results, test_scores = [], {}

    # 0. Baseline: always predict the base rate
    t = time.time()
    dummy = DummyClassifier(strategy="prior").fit(X[tr], y[tr])
    p = dummy.predict_proba(X[te])[:, 1]
    results.append({"model": "Always 'on time' (baseline)", "roc_auc": 0.5, "pr_auc": round(float(y[te].mean()), 4),
                    "precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": None, "train_seconds": 0.0})

    # 1. Logistic regression (one-hot categoricals, scaled numerics)
    t = time.time()
    lr_cats = CATEGORICAL
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), lr_cats),
        ("num", StandardScaler(), NUMERIC),
    ])
    lr = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=300))])
    lr.fit(X.loc[fit], y[fit])
    thr = best_threshold(y[val], lr.predict_proba(X.loc[val])[:, 1])
    lr.fit(X.loc[tr], y[tr])
    p = lr.predict_proba(X.loc[te])[:, 1]
    results.append(evaluate("Logistic regression", y[te], p, thr, time.time() - t))
    test_scores["lr"] = p
    print(results[-1])

    # 2. Random forest on a 120k subsample (full data is slow)
    t = time.time()
    enc = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
                            remainder="passthrough")
    rf = Pipeline([("enc", enc), ("clf", RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=20, n_jobs=-1, random_state=SEED))])
    sub = np.where(fit)[0]
    sub = np.random.RandomState(SEED).choice(sub, 120_000, replace=False)
    rf.fit(X.iloc[sub], y[sub])
    thr = best_threshold(y[val], rf.predict_proba(X.loc[val])[:, 1])
    p = rf.predict_proba(X.loc[te])[:, 1]
    results.append(evaluate("Random forest (120k rows)", y[te], p, thr, time.time() - t))
    test_scores["rf"] = p
    print(results[-1])

    # 3. LightGBM with native categoricals
    t = time.time()
    Xl = X.copy()
    cat_types = {c: pd.CategoricalDtype(sorted(X.loc[tr, c].unique())) for c in CATEGORICAL}
    for c in CATEGORICAL:
        Xl[c] = Xl[c].astype(cat_types[c])
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                  cat_smooth=50, min_data_per_group=200, n_estimators=2000, random_state=SEED, n_jobs=-1, verbose=-1)
    gbm = lgb.LGBMClassifier(**params)
    gbm.fit(Xl.loc[fit], y[fit], eval_set=[(Xl.loc[val], y[val])], eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    best_n = int(gbm.best_iteration_ or params["n_estimators"])
    thr = best_threshold(y[val], gbm.predict_proba(Xl.loc[val])[:, 1])
    params["n_estimators"] = best_n
    gbm = lgb.LGBMClassifier(**params).fit(Xl.loc[tr], y[tr])  # refit on all training months
    p = gbm.predict_proba(Xl.loc[te])[:, 1]
    results.append(evaluate(f"LightGBM ({best_n} trees)", y[te], p, thr, time.time() - t))
    test_scores["lgbm"] = p
    print(results[-1])

    # Deployed model: refit on ALL 12 months (the test score above is from the Jan-Sep fit)
    allm = np.ones(len(X), dtype=bool)
    cat_types = {c: pd.CategoricalDtype(sorted(X[c].unique())) for c in CATEGORICAL}
    Xa = X.copy()
    for c in CATEGORICAL:
        Xa[c] = Xa[c].astype(cat_types[c])
    params["n_estimators"] = int(best_n * 1.15)
    final = lgb.LGBMClassifier(**params).fit(Xa, y)
    joblib.dump({"model": final, "cat_types": cat_types, "threshold": thr, "features": FEATURES}, MODELS / "model.joblib")

    hist = raw
    stats = hist.groupby(["Origin", "Dest"]).agg(n=("Distance", "size"), Distance=("Distance", "median"),
                                                  CRSElapsedTime=("CRSElapsedTime", "median")).reset_index()
    stats = stats[stats.n >= 30]
    carriers = hist.groupby(["Origin", "Dest", "Carrier"]).size().reset_index(name="n")
    carriers = carriers[carriers.n >= 30]
    meta = {
        "routes": {f"{r.Origin}-{r.Dest}": {"distance": float(r.Distance), "elapsed": float(r.CRSElapsedTime)}
                   for r in stats.itertuples()},
        "route_carriers": {k: sorted(g["Carrier"].tolist()) for k, g in
                           carriers.assign(k=carriers.Origin + "-" + carriers.Dest).groupby("k")},
        "threshold": thr,
        "base_rate": float(y.mean()),
    }
    (MODELS / "meta.json").write_text(json.dumps(meta))

    imp = pd.Series(final.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    imp = (imp / imp.sum()).round(4)
    (REPORTS / "metrics.json").write_text(json.dumps({"results": results, "feature_importance_gain": imp.to_dict()}, indent=2))
    print("\nTop features (share of gain):\n", imp.head(10))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for key, label in [("lr", "Logistic regression"), ("rf", "Random forest"), ("lgbm", "LightGBM")]:
            fpr, tpr, _ = roc_curve(y[te], test_scores[key])
            ax[0].plot(fpr, tpr, label=f"{label} ({roc_auc_score(y[te], test_scores[key]):.3f})")
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[0].set(title="ROC curve (test: Oct-Dec)", xlabel="False positive rate", ylabel="True positive rate")
        ax[0].legend(loc="lower right")
        imp.head(10)[::-1].plot.barh(ax=ax[1])
        ax[1].set(title="LightGBM feature importance (share of gain)")
        plt.tight_layout()
        plt.savefig(REPORTS / "results.png", dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)

    print("\n" + pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
