"""FastAPI service: POST /api/predict  ->  delay probability for a scheduled flight.

Run locally:  uvicorn app.main:app --reload
"""
import json
import sys
from datetime import date as Date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import build_features  # noqa: E402

bundle = joblib.load(ROOT / "models" / "model.joblib")
MODEL, CAT_TYPES, THRESHOLD = bundle["model"], bundle["cat_types"], bundle["threshold"]
META = json.loads((ROOT / "models" / "meta.json").read_text())
ROUTES, ROUTE_CARRIERS, BASE_RATE = META["routes"], META["route_carriers"], META["base_rate"]
LOW_CUTOFF = 0.12  # roughly the lowest-risk quarter of flights

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

app = FastAPI(title="Flight Delay Predictor")


class FlightIn(BaseModel):
    carrier: str = Field(min_length=2, max_length=3)
    origin: str = Field(min_length=3, max_length=3)
    dest: str = Field(min_length=3, max_length=3)
    date: Date
    dep_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="scheduled departure, local HH:MM")
    arr_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="scheduled arrival, local HH:MM")


def _hhmm(s: str) -> int:
    return int(s.replace(":", ""))


def _pretty_time(hhmm: int) -> str:
    h, m = divmod(hhmm, 100)
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _explain(row: pd.Series, flight: FlightIn) -> dict:
    """Readable labels for the top model factors (feature -> value shown to the user)."""
    return {
        "Carrier": f"Airline {flight.carrier}",
        "Origin": f"Departing {flight.origin}",
        "Dest": f"Arriving {flight.dest}",
        "Month": f"Travelling in {MONTHS[int(row['Month']) - 1]}",
        "DayOfWeek": f"Flying on a {DAYS[int(row['DayOfWeek']) - 1]}",
        "IsWeekend": "Weekend travel" if row["IsWeekend"] else "Weekday travel",
        "DepHour": f"Departure around {_pretty_time(_hhmm(flight.dep_time))}",
        "DepMinuteOfDay": f"Departure around {_pretty_time(_hhmm(flight.dep_time))}",
        "ArrHour": f"Arrival around {_pretty_time(_hhmm(flight.arr_time))}",
        "DaysToHoliday": "Close to a US holiday" if row["DaysToHoliday"] <= 3 else "Not near a US holiday",
        "IsHoliday": "On a US holiday" if row["IsHoliday"] else "Not on a US holiday",
        "CRSElapsedTime": "Scheduled flight length",
        "Distance": "Route distance",
    }


@app.get("/api/options")
def options():
    """Valid routes and the airlines that fly each one (the form only offers these)."""
    return {"routes": {k: sorted(v) for k, v in ROUTE_CARRIERS.items() if k in ROUTES}}


@app.post("/api/predict")
def predict(f: FlightIn):
    carrier, origin, dest = f.carrier.upper(), f.origin.upper(), f.dest.upper()
    route = f"{origin}-{dest}"
    if route not in ROUTES:
        raise HTTPException(422, f"No data for route {route}. Pick one of the routes in the list (US domestic only).")
    if carrier not in ROUTE_CARRIERS.get(route, []):
        raise HTTPException(422, f"{carrier} has no flights on {route} in the training data.")
    f = f.model_copy(update={"carrier": carrier, "origin": origin, "dest": dest})

    stats = ROUTES[route]
    row = pd.DataFrame([{
        "FlightDate": f.date, "Carrier": carrier, "Origin": origin, "Dest": dest,
        "CRSDepTime": _hhmm(f.dep_time), "CRSArrTime": _hhmm(f.arr_time),
        "CRSElapsedTime": stats["elapsed"], "Distance": stats["distance"],
    }])
    X = build_features(row)
    for c, t in CAT_TYPES.items():
        X[c] = X[c].astype(t)

    p = float(MODEL.predict_proba(X)[0, 1])
    contrib = MODEL.booster_.predict(X, pred_contrib=True)[0][:-1]  # per-feature log-odds push; last col = bias
    labels = _explain(X.iloc[0], f)
    ranked = sorted(zip(X.columns, contrib), key=lambda kv: -abs(kv[1]))
    factors, seen = [], set()
    for name, val in ranked:
        text = labels.get(name, name)
        if text in seen or abs(val) < 0.02:
            continue
        seen.add(text)
        factors.append({"text": text, "effect": "raises risk" if val > 0 else "lowers risk", "weight": round(float(val), 3)})
        if len(factors) == 4:
            break

    risk = "Low" if p < LOW_CUTOFF else ("Moderate" if p < THRESHOLD else "High")
    return {
        "probability": round(p, 4),
        "risk": risk,
        "likely_delayed": p >= THRESHOLD,
        "average_flight_probability": round(BASE_RATE, 3),
        "factors": factors,
        "route": route,
        "note": "Schedule-only model trained on 2017 US domestic flights. No weather or live data.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
