"""Download 2017 hourly weather for the top-200 airports from the Open-Meteo archive API.

Run on a machine with normal internet access (free, no API key):
    python scripts/fetch_weather.py

* Resumable: each airport is cached in data/weather_cache/<IATA>.json, so you can stop
  and re-run at any time and it continues where it left off.
* Rate-limit safe: Open-Meteo's free tier allows ~5,000 weighted calls per hour and a
  full-year, 10-variable request costs ~26, so ~200 airports takes roughly 1-1.5 hours.
  On HTTP 429 the script waits and retries automatically.
* Output: data/weather_2017.csv.gz  (airport, local time, 10 hourly variables)
"""
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from weather import API_VARS  # noqa: E402

CACHE = ROOT / "data" / "weather_cache"
OUT = ROOT / "data" / "weather_2017.csv.gz"
URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2017-01-01", "2017-12-31"


def fetch(lat, lon):
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
                                "hourly": ",".join(API_VARS), "timezone": "auto"})
    for attempt in range(40):
        try:
            with urllib.request.urlopen(f"{URL}?{q}", timeout=120) as r:
                data = json.loads(r.read())
            if data.get("error"):
                raise RuntimeError(data.get("reason"))
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                wait = 60 if attempt < 3 else 300
                print(f"   HTTP {e.code}, waiting {wait}s (attempt {attempt + 1})", flush=True)
                time.sleep(wait)
                continue
            raise
        except (OSError, json.JSONDecodeError, http.client.HTTPException) as e:
            # dropped connection, timeout, or an empty/truncated response body
            print(f"   network error ({type(e).__name__}), retrying in 20s (attempt {attempt + 1})", flush=True)
            time.sleep(20)
    raise RuntimeError("gave up after 40 attempts (rate limiting or network errors)")


def main():
    airports = json.loads((ROOT / "models" / "airports.json").read_text())
    todo = [(k, v) for k, v in airports.items() if v["weather"]]
    CACHE.mkdir(parents=True, exist_ok=True)
    for i, (code, a) in enumerate(todo, 1):
        path = CACHE / f"{code}.json"
        if path.exists():
            continue
        print(f"[{i}/{len(todo)}] {code} {a['city']}", flush=True)
        data = fetch(a["lat"], a["lon"])
        path.write_text(json.dumps({"timezone": data.get("timezone"), "hourly": data["hourly"]}))
        time.sleep(1)

    frames = []
    for code, _ in todo:
        h = json.loads((CACHE / f"{code}.json").read_text())["hourly"]
        f = pd.DataFrame(h)
        f.insert(0, "airport", code)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"saved {OUT}  rows={len(df):,}  airports={df.airport.nunique()}")


if __name__ == "__main__":
    main()
