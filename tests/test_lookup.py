"""Flight-number lookup tests with simulated AeroDataBox responses (no network, no key needed)."""
import sys
import urllib.error
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "app", ROOT / "src"):
    sys.path.insert(0, str(p))

import flight_lookup  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
DATE = "2026-10-10"


def leg(number, carrier, o, d, dep_local, arr_local, codeshare="IsOperator", local=True):
    def point(code, t, tz):
        st = {"local": t} if local else {"utc": t}
        return {"airport": {"iata": code, "shortName": code + " Intl", "timeZone": tz}, "scheduledTime": st}
    return {"number": number, "airline": {"iata": carrier, "name": carrier + " Airlines"}, "status": "Expected",
            "codeshareStatus": codeshare,
            "departure": point(o, dep_local, "America/New_York"), "arrival": point(d, arr_local, "America/Los_Angeles")}


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    flight_lookup.clear_cache()
    monkeypatch.setenv("AERODATABOX_KEY", "test-key")
    yield
    flight_lookup.clear_cache()


def fake(monkeypatch, response):
    calls = []

    def _req(number, date):
        calls.append((number, date))
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(flight_lookup, "_request", _req)
    return calls


@pytest.mark.parametrize("raw,expected", [("DL 423", "DL423"), ("dl-0423", "DL423"), ("B6 1", "B61"), ("9E5432", "9E5432")])
def test_normalise(raw, expected):
    assert flight_lookup.normalise(raw) == expected


@pytest.mark.parametrize("bad", ["", "DELTA", "123", "D", "DL12345", "12345"])
def test_normalise_rejects_junk(bad):
    with pytest.raises(flight_lookup.LookupError_):
        flight_lookup.normalise(bad)


def test_lookup_fills_route_and_local_times(monkeypatch):
    fake(monkeypatch, [leg("DL 423", "DL", "JFK", "LAX", "2026-10-10 17:30-04:00", "2026-10-10 20:45-07:00")])
    r = client.get("/api/flight-lookup", params={"number": "dl423", "date": DATE})
    assert r.status_code == 200
    (l,) = r.json()["legs"]
    assert (l["origin"], l["dest"], l["carrier"]) == ("JFK", "LAX", "DL")
    assert (l["date"], l["dep_time"], l["arr_time"]) == (DATE, "17:30", "20:45")
    assert l["supported"] is True


def test_utc_only_times_are_converted_to_airport_local(monkeypatch):
    fake(monkeypatch, [leg("DL 423", "DL", "JFK", "LAX", "2026-10-10 21:30Z", "2026-10-11 03:45Z", local=False)])
    (l,) = client.get("/api/flight-lookup", params={"number": "DL423", "date": DATE}).json()["legs"]
    assert (l["dep_time"], l["arr_time"]) == ("17:30", "20:45")   # EDT is UTC-4, PDT is UTC-7


def test_codeshare_listing_is_ignored_in_favour_of_operator(monkeypatch):
    fake(monkeypatch, [leg("AF 6720", "AF", "JFK", "LAX", "2026-10-10 17:30-04:00", "2026-10-10 20:45-07:00", "IsCodeshared"),
                       leg("DL 423", "DL", "JFK", "LAX", "2026-10-10 17:30-04:00", "2026-10-10 20:45-07:00")])
    legs = client.get("/api/flight-lookup", params={"number": "AF6720", "date": DATE}).json()["legs"]
    assert len(legs) == 1 and legs[0]["carrier"] == "DL"


def test_multi_leg_flight_returns_each_leg_in_order(monkeypatch):
    fake(monkeypatch, [leg("WN 100", "WN", "DAL", "HOU", "2026-10-10 12:00-05:00", "2026-10-10 13:05-05:00"),
                       leg("WN 100", "WN", "MDW", "DAL", "2026-10-10 08:00-05:00", "2026-10-10 10:30-05:00")])
    legs = client.get("/api/flight-lookup", params={"number": "WN100", "date": DATE}).json()["legs"]
    assert [(l["origin"], l["dest"]) for l in legs] == [("MDW", "DAL"), ("DAL", "HOU")]


def test_non_us_flight_is_flagged_unsupported(monkeypatch):
    fake(monkeypatch, [leg("AI 101", "AI", "DEL", "JFK", "2026-10-10 02:00+05:30", "2026-10-10 07:50-04:00")])
    (l,) = client.get("/api/flight-lookup", params={"number": "AI101", "date": DATE}).json()["legs"]
    assert l["supported"] is False and "US domestic" in l["reason"]


def test_unknown_flight_gives_404(monkeypatch):
    fake(monkeypatch, [])
    r = client.get("/api/flight-lookup", params={"number": "DL9999", "date": DATE})
    assert r.status_code == 404 and "No flight DL9999" in r.json()["detail"]


def test_missing_key_gives_clear_503(monkeypatch):
    monkeypatch.delenv("AERODATABOX_KEY")
    r = client.get("/api/flight-lookup", params={"number": "DL423", "date": DATE})
    assert r.status_code == 503 and "isn't set up" in r.json()["detail"]


def test_rate_limit_is_reported(monkeypatch):
    fake(monkeypatch, urllib.error.HTTPError("u", 429, "Too Many", {}, None))
    r = client.get("/api/flight-lookup", params={"number": "DL423", "date": DATE})
    assert r.status_code == 503 and "limit" in r.json()["detail"]


def test_results_are_cached(monkeypatch):
    calls = fake(monkeypatch, [leg("DL 423", "DL", "JFK", "LAX", "2026-10-10 17:30-04:00", "2026-10-10 20:45-07:00")])
    for _ in range(3):
        client.get("/api/flight-lookup", params={"number": "DL 423", "date": DATE})
    assert calls == [("DL423", DATE)]
