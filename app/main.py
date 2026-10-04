"""FastAPI service: POST /api/predict  ->  delay probability for a scheduled flight.

Two models:
  * weather model   - used when the flight is inside the 16-day forecast window and live
                      forecasts load (Open-Meteo, origin at departure + destination at arrival)
  * schedule model  - fallback for later dates, unknown airports, or if the forecast API fails

Run locally:  uvicorn app.main:app --reload
"""
import json
import sys
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
from features import build_features  # noqa: E402
from weather import WMO_TEXT, add_weather, flight_times, prepare_weather  # noqa: E402
import weather_live  # noqa: E402

SCHED = joblib.load(ROOT / "models" / "model.joblib")
WX_PATH = ROOT / "models" / "model_weather.joblib"
WX = joblib.load(WX_PATH) if WX_PATH.exists() else None
META = json.loads((ROOT / "models" / "meta.json").read_text())
ROUTES, ROUTE_CARRIERS, BASE_RATE = META["routes"], META["route_carriers"], META["base_rate"]
AIRPORTS = json.loads((ROOT / "models" / "airports.json").read_text())
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


def _in_forecast_window(d: Date) -> bool:
    today = Date.today()
    return today - timedelta(days=1) <= d <= today + timedelta(days=weather_live.FORECAST_DAYS - 1)


def _labels(row: pd.Series, f: FlightIn) -> dict:
    """Readable text for each model feature, used to explain the top factors."""
    dep, arr = _pretty_time(_hhmm(f.dep_time)), _pretty_time(_hhmm(f.arr_time))
    out = {
        "Carrier": f"Airline {f.carrier}",
        "Origin": f"Departing {f.origin}",
        "Dest": f"Arriving {f.dest}",
        "Month": f"Travelling in {MONTHS[int(row['Month']) - 1]}",
        "DayOfWeek": f"Flying on a {DAYS[int(row['DayOfWeek']) - 1]}",
        "IsWeekend": "Weekend travel" if row["IsWeekend"] else "Weekday travel",
        "DepHour": f"Departure around {dep}",
        "DepMinuteOfDay": f"Departure around {dep}",
        "ArrHour": f"Arrival around {arr}",
        "DaysToHoliday": "Close to a US holiday" if row["DaysToHoliday"] <= 3 else "Not near a US holiday",
        "IsHoliday": "On a US holiday" if row["IsHoliday"] else "Not on a US holiday",
        "CRSElapsedTime": "Scheduled flight length",
        "Distance": "Route distance",
    }
    for p, ap, when in (("o_", f.origin, "at departure"), ("d_", f.dest, "at arrival")):
        v = lambda k: row.get(p + k)  # noqa: E731
        if pd.isna(v("temp")):
            continue
        out.update({
            p + "precip": f"Rain/snow at {ap} {when} ({v('precip'):.1f} mm/h)",
            p + "precip_3h": f"Rain/snow at {ap} {when} ({v('precip_3h'):.1f} mm in 3h)",
            p + "snowfall_3h": f"Snowfall at {ap} {when} ({v('snowfall_3h'):.1f} cm in 3h)",
            p + "snow_depth": f"Snow on the ground at {ap}",
            p + "snow": f"Snow at {ap} {when}" if v("snow") else f"No snow at {ap}",
            p + "temp": f"Temperature at {ap} {when} ({v('temp'):.0f}°C)",
            p + "dewspread": f"Humidity / fog risk at {ap} {when}",
            p + "cloud_low": f"Low cloud at {ap} {when} ({v('cloud_low'):.0f}%)",
            p + "wind": f"Wind at {ap} {when} ({v('wind'):.0f} km/h)",
            p + "gust": f"Wind gusts at {ap} {when} ({v('gust'):.0f} km/h)",
            p + "pressure": f"Air pressure at {ap} {when}",
        })
    return out


def _conditions(raw: pd.DataFrame, prepared: pd.DataFrame, airport: str, when: pd.Timestamp) -> dict | None:
    """Short weather summary for one airport-hour, shown on the page."""
    r = raw[(raw["airport"] == airport) & (pd.to_datetime(raw["time"]) == when)]
    p = prepared[(prepared["airport"] == airport) & (prepared["time"] == when)]
    if r.empty or p.empty:
        return None
    code, p = r["weather_code"].iloc[0], p.iloc[0]
    return {
        "airport": airport,
        "time": when.strftime("%a %d %b, %H:00 local"),
        "conditions": WMO_TEXT.get(int(code), "Unknown") if pd.notna(code) else "Unknown",
        "temp_c": round(float(p["temp"]), 1),
        "precip_mm_3h": round(float(p["precip_3h"]), 1),
        "wind_kmh": round(float(p["wind"])),
        "gust_kmh": round(float(p["gust"])),
    }


def _predict(bundle: dict, X: pd.DataFrame):
    X = X[bundle["features"]].copy()
    for c, t in bundle["cat_types"].items():
        X[c] = X[c].astype(t)
    p = float(bundle["model"].predict_proba(X)[0, 1])
    contrib = bundle["model"].booster_.predict(X, pred_contrib=True)[0][:-1]  # last column = bias
    return p, list(zip(X.columns, contrib)), X.iloc[0]


@app.get("/api/options")
def options():
    """Valid routes and the airlines that fly each one (the form only offers these)."""
    return {"routes": {k: sorted(v) for k, v in ROUTE_CARRIERS.items() if k in ROUTES},
            "forecast_days": weather_live.FORECAST_DAYS, "weather_model": WX is not None}


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

    # Try the weather model first
    mode, weather_info, reason = "schedule", None, None
    if WX is None:
        reason = "Weather model not available."
    elif not _in_forecast_window(f.date):
        reason = f"Live forecasts only cover the next {weather_live.FORECAST_DAYS} days, so this uses the schedule-only model."
    elif origin not in AIRPORTS or dest not in AIRPORTS:
        reason = "No coordinates for one of these airports, so this uses the schedule-only model."
    else:
        raw = weather_live.fetch_many({a: AIRPORTS[a] for a in {origin, dest}})
        if raw is None:
            reason = "The weather forecast service didn't respond, so this uses the schedule-only model."
        else:
            prepared = prepare_weather(raw)
            Xw = add_weather(X, row, prepared)
            if pd.isna(Xw.loc[0, "o_temp"]):
                reason = "No forecast for the departure hour yet, so this uses the schedule-only model."
            else:
                X, mode = Xw, "weather"
                dep_dt, arr_dt = flight_times(row)
                weather_info = {"origin": _conditions(raw, prepared, origin, dep_dt[0]),
                                "dest": _conditions(raw, prepared, dest, arr_dt[0])}

    bundle = WX if mode == "weather" else SCHED
    p, contrib, xrow = _predict(bundle, X)
    labels = _labels(xrow, f)
    factors, seen = [], set()
    for name, val in sorted(contrib, key=lambda kv: -abs(kv[1])):
        text = labels.get(name)
        if text is None or text in seen or abs(val) < 0.02:
            continue
        seen.add(text)
        factors.append({"text": text, "effect": "raises risk" if val > 0 else "lowers risk", "weight": round(float(val), 3)})
        if len(factors) == 5:
            break

    thr = bundle["threshold"]
    risk = "Low" if p < LOW_CUTOFF else ("Moderate" if p < thr else "High")
    note = ("Uses the live weather forecast for both airports. Model trained on 2017 US domestic flights."
            if mode == "weather" else f"{reason} Model trained on 2017 US domestic flights.")
    return {
        "probability": round(p, 4),
        "risk": risk,
        "likely_delayed": p >= thr,
        "average_flight_probability": round(BASE_RATE, 3),
        "model": mode,
        "weather": weather_info,
        "factors": factors,
        "route": route,
        "note": note,
    }


@app.get("/health")
def health():
    return {"status": "ok", "weather_model": WX is not None}


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
