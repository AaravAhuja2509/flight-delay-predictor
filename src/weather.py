"""Weather features: turn hourly weather per airport into per-flight features.

The same code path is used for:
  * training  - historical hourly weather (Open-Meteo archive, scripts/fetch_weather.py)
  * serving   - live hourly forecasts (Open-Meteo forecast API, app/weather_live.py)

Both APIs return the same variable names and units, and both are requested with
timezone=auto, so times are LOCAL airport time, matching BTS scheduled times.
"""
import numpy as np
import pandas as pd

# Open-Meteo hourly variable names (identical in the archive and forecast APIs).
# Exactly 10 variables keeps each archive request at the minimum API-call weight.
API_VARS = [
    "temperature_2m", "dew_point_2m", "precipitation", "snowfall", "snow_depth",
    "weather_code", "cloud_cover_low", "wind_speed_10m", "wind_gusts_10m", "pressure_msl",
]

# Per-airport, per-hour features derived from the raw variables
BASE = ["temp", "dewspread", "precip", "precip_3h", "snowfall_3h", "snow_depth",
        "cloud_low", "wind", "gust", "pressure", "storm", "fog", "snow", "freezing"]
WEATHER_FEATURES = [f"o_{b}" for b in BASE] + [f"d_{b}" for b in BASE]  # o_ = origin at departure, d_ = destination at arrival

# WMO weather codes -> short text, used for display in the app
WMO_TEXT = {0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Freezing fog",
            51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
            61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Rain showers",
            81: "Rain showers", 82: "Violent rain showers", 85: "Snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail"}


def prepare_weather(raw: pd.DataFrame) -> pd.DataFrame:
    """raw: columns airport, time (local, hourly) + API_VARS.  Returns airport/time + BASE features."""
    w = raw.copy()
    w["time"] = pd.to_datetime(w["time"]).dt.floor("h")
    w = w.sort_values(["airport", "time"]).drop_duplicates(["airport", "time"], keep="last")
    g = w.groupby("airport", sort=False)
    code = w["weather_code"].fillna(-1)
    out = pd.DataFrame({
        "airport": w["airport"].values,
        "time": w["time"].values,
        "temp": w["temperature_2m"].values,
        "dewspread": (w["temperature_2m"] - w["dew_point_2m"]).values,  # small spread -> fog / low cloud risk
        "precip": w["precipitation"].values,
        "precip_3h": g["precipitation"].transform(lambda s: s.rolling(3, min_periods=1).sum()).values,
        "snowfall_3h": g["snowfall"].transform(lambda s: s.rolling(3, min_periods=1).sum()).values,
        "snow_depth": w["snow_depth"].values,
        "cloud_low": w["cloud_cover_low"].values,
        "wind": w["wind_speed_10m"].values,
        "gust": w["wind_gusts_10m"].values,
        "pressure": w["pressure_msl"].values,
        "storm": (code >= 95).astype(int).values,
        "fog": code.isin([45, 48]).astype(int).values,
        "snow": (code.between(71, 77) | code.isin([85, 86])).astype(int).values,
        "freezing": code.isin([56, 57, 66, 67]).astype(int).values,
    })
    return out


def flight_times(df: pd.DataFrame):
    """Local scheduled departure hour and arrival hour (arrival rolls to next day for overnight flights)."""
    date = pd.to_datetime(df["FlightDate"]).dt.normalize()
    dep = df["CRSDepTime"].astype(int).clip(0, 2359)
    arr = df["CRSArrTime"].astype(int).clip(0, 2400) % 2400
    dep_dt = date + pd.to_timedelta(dep // 100, unit="h")
    arr_dt = date + pd.to_timedelta(arr // 100, unit="h")
    # Arrival earlier in the clock than departure => lands the next day (red-eye).
    # Time zones can make short westbound flights look like this too, so only roll
    # over when the gap is large (arrival at least 6 clock-hours "before" departure).
    overnight = (dep - arr) >= 600
    arr_dt = arr_dt + pd.to_timedelta(overnight.astype(int), unit="D")
    return dep_dt, arr_dt


def add_weather(features: pd.DataFrame, df: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    """Append o_*/d_* weather columns to `features`. Missing airport/hour -> NaN (LightGBM handles NaN)."""
    dep_dt, arr_dt = flight_times(df)
    out = features.copy()
    idx = prepared.set_index(["airport", "time"])[BASE]
    for prefix, airports, times in (("o_", df["Origin"], dep_dt), ("d_", df["Dest"], arr_dt)):
        keys = pd.MultiIndex.from_arrays([airports.astype(str).values, times.values])
        block = idx.reindex(keys)
        for b in BASE:
            out[prefix + b] = block[b].values.astype(float)
    return out


def describe(prepared_row: pd.Series, code: float) -> dict:
    """Small human-readable summary of one airport-hour (for the app)."""
    return {
        "conditions": WMO_TEXT.get(int(code), "Unknown") if pd.notna(code) else "Unknown",
        "temp_c": None if pd.isna(prepared_row["temp"]) else round(float(prepared_row["temp"]), 1),
        "precip_mm_3h": None if pd.isna(prepared_row["precip_3h"]) else round(float(prepared_row["precip_3h"]), 1),
        "wind_kmh": None if pd.isna(prepared_row["wind"]) else round(float(prepared_row["wind"])),
        "gust_kmh": None if pd.isna(prepared_row["gust"]) else round(float(prepared_row["gust"])),
    }
