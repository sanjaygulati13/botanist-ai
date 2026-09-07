"""Tests for the Botanist AI backend.

Run from the backend/ directory:

    pip install pytest
    python -m pytest test_main.py -v

Everything runs offline except TestWorldCities, which exercises the live
Open-Meteo geocoding API with 10 real cities (needs internet).
"""

import asyncio
import datetime
import json
import os
import sys
from urllib.parse import quote

import httpx
import pytest

# Importing main validates the environment, so give it safe defaults.
# A real .env still wins, since load_dotenv does not override set vars.
os.environ.setdefault("MODEL_NAME", "test-model")
os.environ.setdefault("LOCATION_NAME", "Fremont, CA")
os.environ.setdefault("SEARXNG_URL", "http://searxng.test/search")
os.environ.setdefault("LLM_URL", "http://llm.test/v1/chat/completions")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main  # noqa: E402


def request(path):
    """Send one GET through the app in-process; no server needed."""
    async def _do():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)
    return asyncio.run(_do())


SAMPLE_CARE = {
    "plant_name": "Monstera deliciosa",
    "seasonal_summary": "A short seasonal summary.",
    "care_details": {
        "watering": "Every two weeks.",
        "lighting": "Bright indirect light.",
        "humidity_temperature": "50-60% humidity.",
        "fertilizer": "Monthly, diluted.",
        "maintenance_tasks": ["Wipe the leaves", "Check the roots"],
    },
    "warning_signs": ["Yellowing leaves", "Brown leaf tips"],
}


@pytest.fixture
def offline(monkeypatch):
    """Point every external call at a canned answer; returns captured prompts."""
    prompts = []

    async def fake_searxng(query):
        return f"Source: test\nContent: care notes for: {query}"

    async def fake_weather():
        return {
            "current": {"temp": 20, "humidity": 50, "description": "clear sky"},
            "forecast": [{"temp": 22, "desc": "fog"}],
        }

    async def fake_llm(prompt):
        prompts.append(prompt)
        return SAMPLE_CARE

    monkeypatch.setattr(main, "fetch_searxng", fake_searxng)
    monkeypatch.setattr(main, "fetch_weather_forecast", fake_weather)
    monkeypatch.setattr(main, "query_llm", fake_llm)
    return prompts


class TestIndex:
    def test_serves_frontend(self):
        resp = request("/")
        assert resp.status_code == 200
        assert "Botanist AI" in resp.text


class TestResearch:
    def test_returns_care_guide(self, offline, monkeypatch):
        monkeypatch.setattr(main, "current_month", lambda: "January")
        resp = request("/api/research/monstera")
        assert resp.status_code == 200
        body = resp.json()
        assert body["plant_name"] == "Monstera deliciosa"
        assert body["warning_signs"] == SAMPLE_CARE["warning_signs"]
        # The prompt carried the requested month and the weather block.
        assert "January" in offline[0]
        assert "CURRENT WEATHER IN Fremont, CA" in offline[0]

    @pytest.mark.parametrize("plant,month", [
        ("Monstera", "January"),
        ("Rosemary", "July"),
        ("Orchid", "December"),
        ("Cactus", "March"),
        ("Fern", "June"),
        ("Aloe vera", "September"),
    ])
    def test_plants_and_months(self, offline, monkeypatch, plant, month):
        monkeypatch.setattr(main, "current_month", lambda: month)
        resp = request(f"/api/research/{quote(plant)}")
        assert resp.status_code == 200, resp.text
        assert month in offline[0]
        assert plant in offline[0]

    def test_no_search_results_404(self, offline, monkeypatch):
        async def empty(query):
            return ""
        monkeypatch.setattr(main, "fetch_searxng", empty)
        resp = request("/api/research/unknown-plant-xyz")
        assert resp.status_code == 404

    def test_llm_down_502(self, offline, monkeypatch):
        from fastapi import HTTPException

        async def down(prompt):
            raise HTTPException(status_code=502, detail="llm down")

        monkeypatch.setattr(main, "query_llm", down)
        resp = request("/api/research/monstera")
        assert resp.status_code == 502

    def test_misshapen_llm_output_is_coerced(self, offline, monkeypatch):
        async def misshapen(prompt):
            return {
                "plant_name": 123,
                "seasonal_summary": None,
                "care_details": {"watering": ["less", "often"]},
                "warning_signs": "yellow leaves",  # wrong type: a bare string
            }
        monkeypatch.setattr(main, "query_llm", misshapen)
        resp = request("/api/research/monstera")
        assert resp.status_code == 200
        body = resp.json()
        assert body["plant_name"] == "123"
        assert body["seasonal_summary"] == ""
        assert body["warning_signs"] == ["yellow leaves"]


class TestHealth:
    def _patch_probes(self, monkeypatch, searxng=True, llm=True, weather=True):
        def const(value):
            async def probe():
                return value
            return probe
        monkeypatch.setattr(main, "_probe_searxng", const(searxng))
        monkeypatch.setattr(main, "_probe_llm", const(llm))
        monkeypatch.setattr(main, "_probe_weather", const(weather))

    def test_health_ok(self, monkeypatch):
        self._patch_probes(monkeypatch)
        body = request("/api/health").json()
        assert body["status"] == "ok"
        assert body["checks"] == {"searxng": True, "llm": True, "weather": True}
        assert body["location"] == main.LOCATION_NAME

    def test_health_degraded(self, monkeypatch):
        self._patch_probes(monkeypatch, llm=False)
        body = request("/api/health").json()
        assert body["status"] == "degraded"
        assert body["checks"]["llm"] is False


class TestParseLlmJson:
    @pytest.mark.parametrize("raw,expected", [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure!\n```\n{"a": 1}\n```\nHope that helps.', {"a": 1}),
    ])
    def test_parses(self, raw, expected):
        assert main.parse_llm_json(raw) == expected

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            main.parse_llm_json("I cannot produce JSON.")


class TestWmoDesc:
    @pytest.mark.parametrize("code,expected", [
        (0, "clear sky"),
        (45, "fog"),
        (95, "thunderstorm"),
        (12345, "weather code 12345"),
        (None, "unknown"),
    ])
    def test_desc(self, code, expected):
        assert main._wmo_desc(code) == expected


class TestWeatherCache:
    def test_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "CACHE_FILE", tmp_path / "weather_cache.json")
        weather = {"current": {"temp": 20, "humidity": 50, "description": "clear sky"},
                   "forecast": []}
        main.write_weather_cache(weather)
        assert main.read_weather_cache() == weather

    def test_stale_cache_discarded(self, monkeypatch, tmp_path):
        cache_file = tmp_path / "weather_cache.json"
        monkeypatch.setattr(main, "CACHE_FILE", cache_file)
        stale = (datetime.datetime.now() - datetime.timedelta(hours=48)).isoformat()
        cache_file.write_text(json.dumps({"timestamp": stale, "weather": {"old": True}}))
        assert main.read_weather_cache() is None

    def test_corrupt_cache_discarded(self, monkeypatch, tmp_path):
        cache_file = tmp_path / "weather_cache.json"
        monkeypatch.setattr(main, "CACHE_FILE", cache_file)
        cache_file.write_text("not json at all")
        assert main.read_weather_cache() is None


class TestWorldCities:
    """Live geocoding: 10 cities on 6 continents must land in the right area.

    These hit the real Open-Meteo API and need internet access.
    """

    CITIES = [
        ("Fremont, CA", (37.4, 37.7), (-122.2, -121.8)),
        ("London, UK", (51.4, 51.7), (-0.3, 0.2)),
        ("Tokyo, Japan", (35.5, 35.9), (139.4, 140.2)),
        ("Sydney, Australia", (-34.1, -33.6), (150.8, 151.6)),
        ("Nairobi, Kenya", (-1.5, -1.0), (36.5, 37.2)),
        ("Sao Paulo, Brazil", (-23.8, -23.3), (-47.0, -46.3)),
        ("Mumbai, India", (18.8, 19.4), (72.6, 73.2)),
        ("Cairo, Egypt", (29.8, 30.3), (30.9, 31.6)),
        ("Reykjavik, Iceland", (64.0, 64.4), (-22.2, -21.6)),
        ("Cape Town, South Africa", (-34.2, -33.6), (18.1, 18.8)),
    ]

    @pytest.mark.parametrize("city,lat_range,lon_range", CITIES, ids=[c[0] for c in CITIES])
    def test_resolves_to_right_area(self, monkeypatch, city, lat_range, lon_range):
        monkeypatch.setattr(main, "LOCATION_NAME", city)
        result = asyncio.run(main.resolve_location())
        assert result is not None, f"could not geocode {city!r}"
        lat, lon = result
        assert lat_range[0] <= lat <= lat_range[1], f"{city}: lat {lat} outside {lat_range}"
        assert lon_range[0] <= lon <= lon_range[1], f"{city}: lon {lon} outside {lon_range}"
