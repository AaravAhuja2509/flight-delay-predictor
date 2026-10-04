"""Live hourly forecasts from the Open-Meteo forecast API (free, no key).

Requested with the same variables and timezone=auto as the training archive, so the
same src/weather.py code turns them into model features.

Forecasts are cached per airport for CACHE_MINUTES; any network/API problem returns
None so the app can fall back to the schedule-only model instead of failing.
"""
import json
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from weather import API_VARS

URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 16   # Open-Meteo maximum
PAST_DAYS = 1        # so 3-hour rolling sums work for early-morning flights
CACHE_MINUTES = 30
TIMEOUT_S = 8

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_lock = threading.Lock()


def _request(lat: float, lon: float) -> dict:
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "hourly": ",".join(API_VARS),
                                "timezone": "auto", "forecast_days": FORECAST_DAYS, "past_days": PAST_DAYS})
    with urllib.request.urlopen(f"{URL}?{q}", timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


def fetch_airport(code: str, lat: float, lon: float) -> pd.DataFrame | None:
    """Hourly forecast for one airport (local time), or None if unavailable."""
    now = time.time()
    with _lock:
        hit = _cache.get(code)
        if hit and now - hit[0] < CACHE_MINUTES * 60:
            return hit[1]
    try:
        data = _request(lat, lon)
        if data.get("error"):
            return None
        df = pd.DataFrame(data["hourly"])
        if df.empty or "time" not in df:
            return None
        df.insert(0, "airport", code)
    except Exception:  # network error, timeout, bad JSON, rate limit ... -> caller falls back
        return None
    with _lock:
        _cache[code] = (now, df)
    return df


def fetch_many(airports: dict[str, dict]) -> pd.DataFrame | None:
    """airports: {code: {"lat":..,"lon":..}}. Returns combined raw hourly frame, or None if any fails."""
    with ThreadPoolExecutor(max_workers=len(airports) or 1) as ex:
        futures = {c: ex.submit(fetch_airport, c, a["lat"], a["lon"]) for c, a in airports.items()}
        frames = {c: f.result() for c, f in futures.items()}
    if any(f is None for f in frames.values()):
        return None
    return pd.concat(frames.values(), ignore_index=True)


def clear_cache():
    with _lock:
        _cache.clear()
