"""API tests with simulated Open-Meteo forecasts (no network needed).

Run:  python -m pytest -q
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))

import weather_live  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
SOON = (date.today() + timedelta(days=3)).isoformat()
FAR = (date.today() + timedelta(days=60)).isoformat()


def fake_forecast(precip=0.0, code=1, gust=20.0):
    """Builds a fake API response: 17 days of identical hourly weather starting yesterday."""
    t = pd.date_range(date.today() - timedelta(days=1), periods=17 * 24, freq="h")
    n = len(t)
    return {"hourly": {
        "time": t.strftime("%Y-%m-%dT%H:%M").tolist(),
        "temperature_2m": [12.0] * n, "dew_point_2m": [6.0] * n,
        "precipitation": [precip] * n, "snowfall": [0.0] * n, "snow_depth": [0.0] * n,
        "weather_code": [code] * n, "cloud_cover_low": [80.0 if precip else 10.0] * n,
        "wind_speed_10m": [gust / 2] * n, "wind_gusts_10m": [gust] * n, "pressure_msl": [1000.0 if precip else 1018.0] * n,
    }}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    weather_live.clear_cache()
    monkeypatch.setattr(weather_live, "_request", lambda lat, lon: fake_forecast())
    yield
    weather_live.clear_cache()


def body(**kw):
    b = dict(carrier="DL", origin="JFK", dest="LAX", date=SOON, dep_time="17:30", arr_time="20:45")
    b.update(kw)
    return b


def test_uses_weather_model_inside_forecast_window():
    r = client.post("/api/predict", json=body())
    assert r.status_code == 200
    d = r.json()
    assert d["model"] == "weather"
    assert d["weather"]["origin"]["airport"] == "JFK" and d["weather"]["dest"]["airport"] == "LAX"
    assert d["weather"]["origin"]["conditions"] == "Mostly clear"
    assert 0 < d["probability"] < 1


def test_falls_back_to_schedule_model_beyond_16_days():
    d = client.post("/api/predict", json=body(date=FAR)).json()
    assert d["model"] == "schedule" and d["weather"] is None
    assert "16 days" in d["note"]


def test_falls_back_when_forecast_service_fails(monkeypatch):
    def boom(lat, lon):
        raise TimeoutError("no response")
    monkeypatch.setattr(weather_live, "_request", boom)
    d = client.post("/api/predict", json=body()).json()
    assert d["model"] == "schedule"
    assert "didn't respond" in d["note"]


def test_heavy_rain_and_wind_raise_delay_risk(monkeypatch):
    clear = client.post("/api/predict", json=body()).json()["probability"]
    weather_live.clear_cache()
    monkeypatch.setattr(weather_live, "_request", lambda lat, lon: fake_forecast(precip=6.0, code=65, gust=70.0))
    storm = client.post("/api/predict", json=body()).json()
    assert storm["probability"] > clear
    assert storm["weather"]["origin"]["conditions"] == "Heavy rain"
    assert any("Rain/snow" in f["text"] for f in storm["factors"])


def test_forecast_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(weather_live, "_request", lambda lat, lon: calls.append(1) or fake_forecast())
    client.post("/api/predict", json=body())
    client.post("/api/predict", json=body())
    assert len(calls) == 2  # one per airport, second prediction served from cache


def test_bad_route_still_rejected():
    r = client.post("/api/predict", json=body(origin="BOM", dest="DEL"))
    assert r.status_code == 422
