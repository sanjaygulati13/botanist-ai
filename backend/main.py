"""Botanist AI backend.

Serves the single-page frontend and exposes /api/research/{plant}, which
combines SearXNG search results, a
current weather forecast, and an LLM served through any
OpenAI-compatible endpoint into a structured, season-aware plant
care guide.

Run with:  python main.py
All configuration comes from the .env file at the repo root (see .env.example).
"""

import asyncio
import datetime
import json
import logging
import os
import pathlib
import re
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # repo root

# .env lives at the repo root; the CWD lookup is a fallback for odd setups.
load_dotenv(BASE_DIR / ".env")
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("botanist")

# --- Configuration (values come from .env) ---

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888/search")
# Any OpenAI-compatible /v1/chat/completions endpoint (vLLM, LM Studio,
# llama.cpp server, ...).
LLM_URL = os.getenv("LLM_URL", "http://localhost:8080/v1/chat/completions")
# Weather comes from Open-Meteo: free, no API key required.
OM_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OM_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MODEL_NAME = os.getenv("MODEL_NAME", "")
if not MODEL_NAME:
    raise RuntimeError("MODEL_NAME is required; set it in .env (see .env.example)")
try:
    PORT = int(os.getenv("PORT", "4004"))
except ValueError as exc:
    raise RuntimeError(f"PORT must be an integer, got {os.getenv('PORT')!r}") from exc

# City for the weather lookup, geocoded automatically (see resolve_location).
LOCATION_NAME = os.getenv("LOCATION_NAME", "")
if not LOCATION_NAME:
    raise RuntimeError("LOCATION_NAME is required; set a city in .env (e.g. 'Fremont, CA')")

# Weather is cached for 24h so repeated queries do not burn API quota.
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "weather_cache.json"
CACHE_TTL = datetime.timedelta(hours=24)

app = FastAPI(title="Botanist AI")

# The API has no auth or cookies, so credentials stay off; combining
# allow_credentials with a wildcard origin is a misconfiguration we want
# to avoid.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


class PlantCareResponse(BaseModel):
    """Shape of the JSON returned by /api/research/{plant}."""
    plant_name: str
    seasonal_summary: str
    care_details: Dict[str, Any]
    warning_signs: List[str]


def read_weather_cache() -> Optional[Dict[str, Any]]:
    """Return the cached forecast if it is younger than CACHE_TTL, else None."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        fetched_at = datetime.datetime.fromisoformat(data["timestamp"])
        if datetime.datetime.now() - fetched_at < CACHE_TTL:
            return data["weather"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A half-written or stale cache file should never take the app down.
        log.warning("Discarding unreadable weather cache: %s", exc)
    return None


def write_weather_cache(weather: Dict[str, Any]) -> None:
    """Persist the latest forecast; a failure here is logged, not fatal."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"timestamp": datetime.datetime.now().isoformat(), "weather": weather}, f)
    except OSError as exc:
        log.warning("Could not write weather cache: %s", exc)


# WMO weather interpretation codes (what Open-Meteo returns) -> short phrase.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    56: "freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def _wmo_desc(code: Optional[int]) -> str:
    """Human phrase for a WMO weather code; unknown codes stay visible, not hidden."""
    return _WMO.get(code, f"weather code {code}") if code is not None else "unknown"


async def resolve_location() -> Optional[tuple[float, float]]:
    """Geocode the configured city via Open-Meteo (no API key needed).

    Returns (lat, lon), or None if the city cannot be resolved.
    """
    city = LOCATION_NAME.split(",")[0].strip()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(OM_GEOCODE_URL, params={"name": city, "count": 1, "language": "en"})
            resp.raise_for_status()
            results = resp.json().get("results") or []
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Geocoding failed for %r: %s", city, exc)
        return None
    if not results:
        log.warning("Geocoding found no match for %r", city)
        return None
    place = results[0]
    log.info("Resolved %r -> %s (%s)", LOCATION_NAME, place["name"], place.get("country_code", "?"))
    return place["latitude"], place["longitude"]


async def fetch_weather_forecast() -> Optional[Dict[str, Any]]:
    """Current conditions + 10-day forecast from Open-Meteo (no API key).

    Returns None on any failure; the endpoint still works, it just
    omits the weather context.
    """
    cached = read_weather_cache()
    if cached is not None:
        return cached

    loc = await resolve_location()
    if loc is None:
        return None
    lat, lon = loc
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code",
        "daily": "temperature_2m_max,weather_code",
        "timezone": "auto",
        "forecast_days": 10,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(OM_FORECAST_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        info = {
            "current": {
                "temp": data["current"]["temperature_2m"],
                "humidity": data["current"]["relative_humidity_2m"],
                "description": _wmo_desc(data["current"].get("weather_code")),
            },
            "forecast": [
                {"temp": t, "desc": _wmo_desc(c)}
                for t, c in zip(data["daily"]["temperature_2m_max"], data["daily"]["weather_code"])
            ],
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        log.warning("Open-Meteo forecast failed: %s", exc)
        return None

    write_weather_cache(info)
    return info


async def fetch_searxng(query: str) -> str:
    """Run one SearXNG query and flatten the results into prompt-ready text.

    Returns an empty string on any failure so the caller can judge how
    much context it actually has.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(SEARXNG_URL, params={"q": query, "format": "json", "language": "en"})
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("SearXNG query failed (%r): %s", query, exc)
        return ""

    snippets = []
    for item in data.get("results", []):
        title = item.get("title", "")
        content = item.get("content", "")
        if title or content:
            snippets.append(f"Source: {title}\nContent: {content}")
    return "\n\n".join(snippets)


# Some models wrap the JSON in markdown fences despite response_format, so
# strip one fence pair before parsing.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_llm_json(content: str) -> Dict[str, Any]:
    """Parse the LLM answer, stripping one pair of markdown fences if present.

    Raises ValueError when the content is not JSON.
    """
    match = _FENCE_RE.search(content)
    if match:
        content = match.group(1).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("content is not valid JSON") from exc


async def query_llm(prompt: str) -> Dict[str, Any]:
    """Send the assembled prompt to the LLM and parse its JSON answer.

    Raises HTTPException(502) with a user-safe message on any failure;
    the full error goes to the log only.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert botanist. Respond with a single valid JSON object and nothing else."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(LLM_URL, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
        log.error("LLM request failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The local LLM did not return a usable response. Check that your LLM gateway is running.",
        ) from exc

    try:
        return parse_llm_json(content)
    except ValueError as exc:
        log.error("LLM returned unparseable JSON: %.500s", content)
        raise HTTPException(status_code=502, detail="The LLM response was not valid JSON; please try again.") from exc


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the single-page frontend."""
    return FileResponse(BASE_DIR / "frontend" / "src" / "index.html")


async def _probe_searxng() -> bool:
    """SearXNG is up if a trivial JSON query answers within 3s."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(SEARXNG_URL, params={"q": "test", "format": "json"})
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def _probe_llm() -> bool:
    """LLM gateway is up if the standard /v1/models probe answers (4xx still counts as up)."""
    base = LLM_URL
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")] + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(base)
        return resp.status_code < 500
    except httpx.HTTPError:
        return False


async def _probe_weather() -> bool:
    """Open-Meteo is up if it can geocode the configured city."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(OM_GEOCODE_URL, params={"name": LOCATION_NAME.split(",")[0].strip(), "count": 1})
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


@app.get("/api/health")
async def health():
    """Liveness of the app plus each external dependency.

    Always returns 200; read `status` ("ok" or "degraded") for the verdict.
    """
    searxng_ok, llm_ok, weather_ok = await asyncio.gather(
        _probe_searxng(), _probe_llm(), _probe_weather()
    )
    return {
        "status": "ok" if (searxng_ok and llm_ok and weather_ok) else "degraded",
        "checks": {"searxng": searxng_ok, "llm": llm_ok, "weather": weather_ok},
        "location": LOCATION_NAME,
        "model": MODEL_NAME,
    }


def current_month() -> str:
    """Name of the current month; tests monkeypatch this to cover other months."""
    return datetime.datetime.now().strftime("%B")


# LLM answers are only loosely typed; these coercers keep a wrong-shaped
# answer from slipping past pydantic or crashing the frontend.
def _as_str(value, default: str) -> str:
    if isinstance(value, (str, int, float)) and str(value).strip():
        return str(value)
    return default


def _as_dict(value) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_str_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


@app.get("/api/research/{plant_name}", response_model=PlantCareResponse)
async def research_plant(plant_name: str):
    """Build a season-aware care guide for a plant from search + weather + LLM."""
    month = current_month()

    # Weather and the three searches are independent, so run them together.
    search_queries = [
        f"{plant_name} care requirements light water soil humidity",
        f"{plant_name} care in {month}",
        f"{plant_name} common pests and diseases",
    ]
    weather_info, *search_results = await asyncio.gather(
        fetch_weather_forecast(),
        *(fetch_searxng(q) for q in search_queries),
    )
    combined_context = "\n\n".join(search_results)

    if not combined_context.strip():
        raise HTTPException(status_code=404, detail=f"Could not find enough information about '{plant_name}'.")

    weather_context = ""
    if weather_info:
        curr = weather_info["current"]
        trend = ", ".join(f"{d['temp']}°C ({d['desc']})" for d in weather_info.get("forecast", []))
        weather_context = (
            f"\n\nCURRENT WEATHER IN {LOCATION_NAME}: "
            f"Temp: {curr['temp']}°C, Humidity: {curr['humidity']}%, Condition: {curr['description']}."
            f"\nNext 10 days trend: {trend}"
        )

    prompt = f"""Use the search data below to write a care guide for '{plant_name}' in {month}{weather_context}

SEARCH DATA:
{combined_context}

Respond with a single JSON object in exactly this shape:
{{
  "plant_name": "{plant_name}",
  "seasonal_summary": "How {month} affects this plant, factoring in the weather above.",
  "care_details": {{
    "watering": "Specific guidance for {month}.",
    "lighting": "Light requirements.",
    "humidity_temperature": "Ideal humidity and temperature.",
    "fertilizer": "Feeding schedule for this time of year.",
    "maintenance_tasks": ["task", "task"]
  }},
  "warning_signs": ["sign", "sign"]
}}"""

    llm_data = await query_llm(prompt)
    # Coerce to the response types so a misshapen LLM answer cannot slip
    # past validation or crash the frontend.
    return PlantCareResponse(
        plant_name=_as_str(llm_data.get("plant_name"), plant_name),
        seasonal_summary=_as_str(llm_data.get("seasonal_summary"), ""),
        care_details=_as_dict(llm_data.get("care_details")),
        warning_signs=_as_str_list(llm_data.get("warning_signs")),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
