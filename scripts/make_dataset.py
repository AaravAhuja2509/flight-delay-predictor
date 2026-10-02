"""Rebuild data/flight_delay_data.csv from public BTS On-Time Performance files.

The monthly 2017 BTS zips are mirrored in github.com/bharathirajatut/flights-dataset.
Original source: https://www.transtats.bts.gov  (Reporting Carrier On-Time Performance).

Run:  python scripts/make_dataset.py
Keeps non-cancelled, non-diverted flights, only pre-departure columns + the target,
then takes a reproducible 500k-row random sample (seed 42).
"""
import glob
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "flight_delay_data.csv"
COLS = ["FlightDate", "Month", "DayofMonth", "DayOfWeek", "Carrier", "TailNum", "FlightNum", "Origin", "Dest",
        "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "Cancelled", "Diverted", "ArrDelay", "ArrDel15"]

with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["git", "clone", "--depth", "1", "-q",
                    "https://github.com/bharathirajatut/flights-dataset.git", tmp], check=True)
    parts = []
    for z in sorted(glob.glob(f"{tmp}/2017/*.zip")):
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            d = pd.read_csv(zf.open(name), usecols=COLS, low_memory=False)
        parts.append(d[(d.Cancelled == 0) & (d.Diverted == 0)].drop(columns=["Cancelled", "Diverted"]))
        print(Path(z).name, len(parts[-1]))

df = pd.concat(parts, ignore_index=True).dropna(subset=["ArrDel15", "CRSElapsedTime"])
df = df.sample(500_000, random_state=42).sort_values(["FlightDate", "CRSDepTime"]).reset_index(drop=True)
df["ArrDel15"] = df.ArrDel15.astype(int)
OUT.parent.mkdir(exist_ok=True)
df.to_csv(OUT, index=False)
print("saved", OUT, df.shape, "delay rate", round(df.ArrDel15.mean(), 3))
