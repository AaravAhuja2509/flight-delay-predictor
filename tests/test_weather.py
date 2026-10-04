"""Checks that flights pick up weather from the right airport and the right hour.

Run:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import build_features  # noqa: E402
from weather import API_VARS, add_weather, flight_times, prepare_weather  # noqa: E402


def _weather(airport, start, hours, **overrides):
    t = pd.date_range(start, periods=hours, freq="h")
    w = pd.DataFrame({v: 0.0 for v in API_VARS}, index=range(hours))
    w["temperature_2m"] = np.arange(hours, dtype=float)  # temp == hour index, easy to check
    w["dew_point_2m"] = w["temperature_2m"] - 5
    for k, v in overrides.items():
        w[k] = v
    w.insert(0, "time", t)
    w.insert(0, "airport", airport)
    return w


FLIGHTS = pd.DataFrame({
    "FlightDate": ["2017-03-10", "2017-03-10"],
    "Carrier": ["DL", "AA"], "Origin": ["JFK", "LAX"], "Dest": ["LAX", "JFK"],
    "CRSDepTime": [1730, 2350],   # second flight is a red-eye
    "CRSArrTime": [2045, 805],
    "CRSElapsedTime": [375.0, 315.0], "Distance": [2475.0, 2475.0],
})


def test_overnight_arrival_rolls_to_next_day():
    dep, arr = flight_times(FLIGHTS)
    assert dep[0] == pd.Timestamp("2017-03-10 17:00") and arr[0] == pd.Timestamp("2017-03-10 20:00")
    assert dep[1] == pd.Timestamp("2017-03-10 23:00") and arr[1] == pd.Timestamp("2017-03-11 08:00")


def test_weather_joins_origin_at_departure_and_dest_at_arrival():
    precip = np.zeros(48)
    precip[[15, 16, 17]] = [1.0, 2.0, 3.0]  # rain at JFK 15:00-17:00 on Mar 10
    w = prepare_weather(pd.concat([
        _weather("JFK", "2017-03-10 00:00", 48, precipitation=precip),
        _weather("LAX", "2017-03-10 00:00", 48),
    ]))
    X = add_weather(build_features(FLIGHTS), FLIGHTS, w)
    # flight 0: JFK at 17:00 -> temp 17, 3h precip = 1+2+3; LAX at 20:00 -> temp 20
    assert X.loc[0, "o_temp"] == 17 and X.loc[0, "o_precip_3h"] == 6.0 and X.loc[0, "d_temp"] == 20
    # flight 1: LAX at 23:00 -> 23; arrives JFK next day 08:00 -> hour index 32
    assert X.loc[1, "o_temp"] == 23 and X.loc[1, "d_temp"] == 32
    assert (X["o_dewspread"] == 5).all()


def test_missing_airport_gives_nan_not_error():
    w = prepare_weather(_weather("JFK", "2017-03-10 00:00", 48))
    X = add_weather(build_features(FLIGHTS), FLIGHTS, w)
    assert X.loc[0, "o_temp"] == 17 and np.isnan(X.loc[0, "d_temp"])


def test_weather_codes_become_flags():
    codes = np.full(24, 3.0)
    codes[17] = 95   # thunderstorm at 17:00
    w = prepare_weather(_weather("JFK", "2017-03-10 00:00", 24, weather_code=codes))
    X = add_weather(build_features(FLIGHTS.iloc[:1]), FLIGHTS.iloc[:1], w)
    assert X.loc[0, "o_storm"] == 1 and X.loc[0, "o_fog"] == 0
