"""Look up a flight's route and scheduled times from its flight number (AeroDataBox API).

Setup (free "Basic" plan, about 400 API units a month):
  1. Sign up at https://rapidapi.com/aedbx-aedbx/api/aerodatabox and subscribe to Basic.
  2. Put your key in a file called .env in the project folder:
         AERODATABOX_KEY=your-rapidapi-key
     (.env is git-ignored, so the key never ends up on GitHub.)

Results are cached for CACHE_HOURS per flight number + date to save quota.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
HOST = "aerodatabox.p.rapidapi.com"
CACHE_HOURS = 6
TIMEOUT_S = 10

_cache: dict[tuple[str, str], tuple[float, list]] = {}
_lock = threading.Lock()


class LookupError_(Exception):
    """Lookup failed in a way the user should hear about (message is shown on the page)."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def api_key() -> str | None:
    return os.environ.get("AERODATABOX_KEY") or None


def normalise(number: str) -> str:
    """'dl 423', 'DL-0423', 'DL423' -> 'DL423'. Raises on anything that isn't a flight number."""
    s = re.sub(r"[\s\-]", "", number or "").upper()
    m = re.fullmatch(r"([A-Z0-9]{2})(\d{1,4})", s)
    if not m or m.group(1).isdigit():
        raise LookupError_("That doesn't look like a flight number. Use the airline code plus number, e.g. DL 423.", 422)
    return f"{m.group(1)}{int(m.group(2))}"


def _request(number: str, date: str) -> list:
    url = f"https://{HOST}/flights/number/{urllib.parse.quote(number)}/{date}?dateLocalRole=Departure"
    req = urllib.request.Request(url, headers={"X-RapidAPI-Key": api_key(), "X-RapidAPI-Host": HOST})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        body = r.read()
    return json.loads(body) if body.strip() else []


def _local(point: dict) -> datetime | None:
    """Scheduled time at that airport, as a naive local datetime."""
    st = point.get("scheduledTime") or {}
    if st.get("local"):                       # e.g. "2026-10-10 17:30-04:00"
        return datetime.strptime(st["local"][:16], "%Y-%m-%d %H:%M")
    if st.get("utc"):                         # e.g. "2026-10-10 21:30Z" -> convert with the airport's zone
        tz = (point.get("airport") or {}).get("timeZone")
        if tz:
            utc = datetime.strptime(st["utc"][:16], "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("UTC"))
            return utc.astimezone(ZoneInfo(tz)).replace(tzinfo=None)
    return None


def _parse(raw: list) -> list[dict]:
    legs = []
    # Prefer the operating flight over codeshare listings of the same leg
    ops = [f for f in raw if f.get("codeshareStatus") != "IsCodeshared"] or raw
    for f in ops:
        dep, arr = f.get("departure") or {}, f.get("arrival") or {}
        o, d = (dep.get("airport") or {}).get("iata"), (arr.get("airport") or {}).get("iata")
        td, ta = _local(dep), _local(arr)
        if not (o and d and td and ta):
            continue
        legs.append({
            "number": (f.get("number") or "").replace(" ", ""),
            "carrier": (f.get("airline") or {}).get("iata"),
            "airline_name": (f.get("airline") or {}).get("name"),
            "origin": o, "dest": d,
            "origin_name": (dep.get("airport") or {}).get("shortName") or (dep.get("airport") or {}).get("name"),
            "dest_name": (arr.get("airport") or {}).get("shortName") or (arr.get("airport") or {}).get("name"),
            "date": td.strftime("%Y-%m-%d"),
            "dep_time": td.strftime("%H:%M"),
            "arr_time": ta.strftime("%H:%M"),
            "status": f.get("status"),
        })
    # de-duplicate identical legs, keep departure order
    seen, out = set(), []
    for leg in sorted(legs, key=lambda x: (x["date"], x["dep_time"])):
        k = (leg["origin"], leg["dest"], leg["date"], leg["dep_time"])
        if k not in seen:
            seen.add(k)
            out.append(leg)
    return out


def lookup(number: str, date: str) -> list[dict]:
    """All legs of flight `number` departing on local `date` (YYYY-MM-DD)."""
    n = normalise(number)
    if not api_key():
        raise LookupError_("Flight number lookup isn't set up yet (no AERODATABOX_KEY). Enter the route and times instead.", 503)
    key, now = (n, date), time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_HOURS * 3600:
            return hit[1]
    try:
        raw = _request(n, date)
    except urllib.error.HTTPError as e:
        if e.code == 404 or e.code == 204:
            raw = []
        elif e.code in (401, 403):
            raise LookupError_("The flight lookup API key was rejected. Check AERODATABOX_KEY.", 503)
        elif e.code == 429:
            raise LookupError_("Flight lookup limit reached for now. Enter the route and times instead.", 503)
        else:
            raise LookupError_("The flight lookup service had a problem. Try again or enter the route and times.")
    except Exception:
        raise LookupError_("Couldn't reach the flight lookup service. Try again or enter the route and times.")
    legs = _parse(raw if isinstance(raw, list) else [])
    with _lock:
        _cache[key] = (now, legs)
    return legs


def clear_cache():
    with _lock:
        _cache.clear()
