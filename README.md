# Botanist AI Dashboard

Type a plant name and the app compiles a season-aware care guide from web
search results plus the live weather and 10-day forecast for your location:
watering, light, humidity, fertilizer, monthly maintenance, and warning signs.

## Why this repo exists

The entire project — FastAPI backend, React frontend, systemd deploy script,
the lot — was written by [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B),
running QAT 4-bit with a 96K context window and Q4_0 KV cache on a single 16 GB GPU.
Blank screen to working app took approx an hour.

I manually did the review pass: hardening error handling, sorting out the
configuration, and fixing a couple of real bugs the first draft had (LLM call error handling, weather location issues).
Nothing structural — the first draft of every file came from the model, and it held up.

This is the proof that if guided properly: a 12B local model can produce a complete, runnable
project, not just plausible snippets. And a small note in the model's favour:
the model that *wrote* this app is not the one it *calls* — at runtime it
works with any OpenAI-compatible LLM endpoint, local or hosted.

## Stack

- **Backend:** FastAPI + Uvicorn (Python)
- **Frontend:** React (single HTML file, no build step)
- **Search:** SearXNG instance
- **LLM:** any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server, ollama, ...)
- **Weather:** Open-Meteo (free, no API key)

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /` | The dashboard UI |
| `GET /api/research/{plant}` | Season-aware care guide (JSON) |
| `GET /api/health` | Dependency health; always 200, `status` is `ok` or `degraded` |

## Project layout

```
.
├── backend/main.py         # FastAPI app (API + serves the frontend)
├── backend/test_main.py    # test suite
├── frontend/src/index.html # single-page React UI
├── deploy.sh               # systemd installer (run with sudo)
├── requirements.txt        # backend dependencies
└── .env.example            # configuration template (copy to .env)
```

## Running locally

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
python3 main.py
```

App starts on `http://localhost:4004`.

## Config

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `MODEL_NAME` | Model ID served by your LLM endpoint (e.g. `gemma-4-12b`) |
| `LOCATION_NAME` | City for weather + shown in responses (e.g. `Fremont, CA`); geocoded automatically |
| `SEARXNG_URL` | Your SearXNG instance URL |
| `LLM_URL` | OpenAI-compatible chat completions URL of your LLM endpoint |

## Tests

```bash
cd backend
pip install pytest
python -m pytest test_main.py -v
```

All tests run offline except `TestWorldCities`, which checks live geocoding
for 10 real cities around the world.

## Deploy as systemd service

```bash
sudo bash deploy.sh
```

Sets up `botanist-ai-dashboard.service` (port from `PORT` in `.env`, default 4004).

## License

MIT — see [LICENSE](LICENSE).
