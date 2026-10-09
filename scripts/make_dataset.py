"""Build the flight dataset from public BTS On-Time Performance files (2017).

The monthly 2017 BTS zips are mirrored in github.com/bharathirajatut/flights-dataset.
Original source: https://www.transtats.bts.gov  (Reporting Carrier On-Time Performance).

    python scripts/make_dataset.py           # 500k-row sample  -> data/flight_delay_data.csv
    python scripts/make_dataset.py --full    # all ~5.6M flights -> data/flights_2017_full.parquet
    python scripts/make_dataset.py --full --src path/to/flights-dataset   # reuse an existing clone

Keeps non-cancelled, non-diverted flights and only pre-departure columns + the target.
"""
import argparse
import glob
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["FlightDate", "Month", "DayofMonth", "DayOfWeek", "Carrier", "TailNum", "FlightNum", "Origin", "Dest",
        "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "Cancelled", "Diverted", "ArrDelay", "ArrDel15"]
DTYPES = {"Carrier": "category", "Origin": "category", "Dest": "category", "TailNum": "string"}


def read_months(src: str) -> pd.DataFrame:
    parts = []
    for z in sorted(glob.glob(f"{src}/2017/*.zip")):
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            d = pd.read_csv(zf.open(name), usecols=COLS, low_memory=False)
        parts.append(d[(d.Cancelled == 0) & (d.Diverted == 0)].drop(columns=["Cancelled", "Diverted"]))
        print(Path(z).name, len(parts[-1]), flush=True)
    df = pd.concat(parts, ignore_index=True).dropna(subset=["ArrDel15", "CRSElapsedTime"])
    df["ArrDel15"] = df.ArrDel15.astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="keep every flight (parquet) instead of a 500k sample")
    ap.add_argument("--src", help="existing clone of bharathirajatut/flights-dataset (skips the download)")
    a = ap.parse_args()
    (ROOT / "data").mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        src = a.src
        if not src:
            subprocess.run(["git", "clone", "--depth", "1", "-q",
                            "https://github.com/bharathirajatut/flights-dataset.git", tmp], check=True)
            src = tmp
        df = read_months(src)

    if a.full:
        out = ROOT / "data" / "flights_2017_full.parquet"
        df = df.sort_values(["FlightDate", "CRSDepTime"]).reset_index(drop=True)
        df.astype({k: v for k, v in DTYPES.items() if k in df}).to_parquet(out, index=False)
    else:
        out = ROOT / "data" / "flight_delay_data.csv"
        df = df.sample(500_000, random_state=42).sort_values(["FlightDate", "CRSDepTime"]).reset_index(drop=True)
        df.to_csv(out, index=False)
    print("saved", out, df.shape, "delay rate", round(df.ArrDel15.mean(), 3))


if __name__ == "__main__":
    main()
