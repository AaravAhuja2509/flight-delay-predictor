"""Feature engineering shared by training (src/train.py) and serving (app/main.py).

Only information known BEFORE departure is used. Never add departure delay,
taxi times, wheels-off or delay-cause columns: that would be target leakage.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CATEGORICAL = ["Carrier", "Origin", "Dest"]  # Route / DayofMonth tested and dropped (no gain, overfit)
NUMERIC = [
    "Month", "DayOfWeek", "DepHour", "DepMinuteOfDay", "ArrHour",
    "CRSElapsedTime", "Distance", "IsHoliday", "DaysToHoliday", "IsWeekend",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "ArrDel15"

_cal = USFederalHolidayCalendar()
_holidays = _cal.holidays(start="2015-01-01", end="2035-12-31").values.astype("datetime64[D]")


def _days_to_nearest_holiday(dates: pd.Series) -> np.ndarray:
    d = dates.values.astype("datetime64[D]")
    idx = np.searchsorted(_holidays, d)
    left = _holidays[np.clip(idx - 1, 0, len(_holidays) - 1)]
    right = _holidays[np.clip(idx, 0, len(_holidays) - 1)]
    dist = np.minimum(np.abs((d - left).astype(int)), np.abs((right - d).astype(int)))
    return np.minimum(dist, 14)  # cap: only the run-up/aftermath of a holiday matters


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs: FlightDate, Carrier, Origin, Dest, CRSDepTime, CRSArrTime, CRSElapsedTime, Distance."""
    date = pd.to_datetime(df["FlightDate"])
    out = pd.DataFrame(index=df.index)
    out["Carrier"] = df["Carrier"].astype(str)
    out["Origin"] = df["Origin"].astype(str)
    out["Dest"] = df["Dest"].astype(str)
    out["Month"] = date.dt.month
    out["DayOfWeek"] = date.dt.dayofweek + 1  # 1=Mon .. 7=Sun, as in BTS
    out["IsWeekend"] = (out["DayOfWeek"] >= 6).astype(int)
    dep = df["CRSDepTime"].astype(int).clip(0, 2359)
    arr = df["CRSArrTime"].astype(int).clip(0, 2400) % 2400
    out["DepHour"] = dep // 100
    out["DepMinuteOfDay"] = (dep // 100) * 60 + dep % 100
    out["ArrHour"] = arr // 100
    out["CRSElapsedTime"] = df["CRSElapsedTime"].astype(float)
    out["Distance"] = df["Distance"].astype(float)
    dth = _days_to_nearest_holiday(date)
    out["DaysToHoliday"] = dth
    out["IsHoliday"] = (dth == 0).astype(int)
    return out[FEATURES]
