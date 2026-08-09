<p align="center">
  <img src="docs/logo/framesleuth-logo-256.png" alt="Framesleuth" width="128" height="128" />
</p>

# Framesleuth

<p align="center">
  <a href="https://github.com/thestackhub1/framesleuth-agent/actions/workflows/ci.yml"><img src="https://github.com/thestackhub1/framesleuth-agent/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0" /></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg" alt="Python 3.11 | 3.12" />
</p>

**Local video → structured context for coding agents, exposed over MCP.**

Feed Framesleuth a video and it reads it frame by frame, folds in optional browser
sidecars, and produces a structured Context Bundle. Any video works: a bug recording, a
feature demo, a design walkthrough, a Loom, a phone capture.

The bundle is served over MCP, so a VS Code agent, another coding agent, or your own
system can drive the analysis and use the result to fix a bug, change a feature, or build
something new, grounded in what the video actually shows.

Capture happens outside this repo, which holds the analysis agent only. A browser capture
extension can record a session and post the video plus sidecars to the local API.

Everything runs locally. Nothing leaves your machine.

## Quick start

> Going from a video to a grounded change inside VS Code? See
> [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md): connect the
> bundled MCP server, then turn a recording into a fix, a feature, or a new build.

### Fastest: one command with Docker

One command brings up the model server, the models, and the API. No Python, no
virtualenv, no manual model setup.

```bash
git clone https://github.com/thestackhub1/framesleuth-agent.git
cd framesleuth-agent
docker compose up            # or: ./scripts/dev_up.sh
```

Compose picks up `docker-compose.override.yml` automatically; that file adds the Ollama
server, the model-pull job, and the model volume. The first run pulls the vision and
coder models (`qwen2.5vl` and `qwen2.5-coder:7b`, ~11 GB total) into a Docker volume,
then starts the backend on `http://127.0.0.1:8010`. Later runs are instant. It's ready
when the health check says `healthy`:

```bash
curl -s http://127.0.0.1:8010/v1/healthz | python -m json.tool   # "status": "healthy"
```

That's the whole setup. Run your first analysis below, or connect the MCP server in your
editor ([VS Code & Claude](docs/use-with-vscode-and-claude.md)).

```bash
docker compose logs -f                  # follow progress / model download
docker compose down --remove-orphans    # stop  (add -v to also delete model volumes)
```

The stack runs its own Ollama on the internal Docker network and never publishes its
port, so it won't clash with a native Ollama on `:11434`. The only host port is the API
on `:8010`.

> Already running Ollama natively with the models pulled? The Docker stack ships its own
> Ollama and would download them again. Use the direct path below instead. It reuses your
> existing Ollama and is faster, especially on macOS, where Docker can't reach the GPU.
>
> On macOS, or anywhere without a GPU, Docker runs the models on CPU and the vision model
> is slow. On Linux with an NVIDIA GPU, uncomment the `deploy:` block on the `ollama`
> service in `docker-compose.override.yml`.
>
> To run only the backend container against a native or external model server, use
> `docker compose -f docker-compose.yml up`. The base compose file defaults to native
> Ollama on `http://host.docker.internal:11434`; override `VLM_URL` and `CODER_URL` for
> another server.
>
> **Docker users: don't `cp .env.example .env`.** If you already did, comment out
> `VLM_URL` and `CODER_URL` in it. Compose reads `.env` and those values beat the
> defaults above, and `.env.example` ships the native `127.0.0.1`, which inside a
> container means the container itself. The symptom is a backend that starts cleanly and
> then can't reach any model.

### Run your first analysis (curl)

Once the API reports healthy, either setup path, three calls take you from a video to a
Context Bundle. Analysis is async: submit, poll, read.

No recording handy? Generate a throwaway one. It exercises the whole pipeline and takes
about a second.

```bash
uv run python scripts/make_sample_video.py     # writes sample.mp4
```

```bash
# 1. Submit any screen recording (mp4/webm). Returns 202 { job_id, ... }
JOB=$(curl -s -F "video=@sample.mp4" http://127.0.0.1:8010/v1/analyze \
  | python -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

# 2. Poll until state is "done" (queued → running → done)
curl -s "http://127.0.0.1:8010/v1/jobs/$JOB" | python -m json.tool

# 3. Read the Context Bundle
curl -s "http://127.0.0.1:8010/v1/report/$JOB" | python -m json.tool
```

Step 1 takes optional form fields: `-F intent="why does save hang?"`,
`-F skill=bug_report`, `-F action=fix`. `GET /v1/skills` and `/v1/actions` list the
choices. Prefer a UI? The [Postman collection](postman/README.md) chains these calls for
you.

### Run it directly (no Docker — fastest on macOS, best for development)

You need Python 3.11+, [`uv`](https://docs.astral.sh/uv/), 8 GB+ RAM, and a local model
server. ffmpeg isn't required, since PyAV bundles its own; if `ffprobe` happens to be on
PATH it's used to detect an audio stream.

```bash
git clone https://github.com/thestackhub1/framesleuth-agent.git
cd framesleuth-agent

# 1. Models — native Ollama (uses the Mac GPU) is the quick path
ollama serve &                                  # skip if already running
ollama pull qwen2.5vl && ollama pull qwen2.5-coder:7b

# 2. Install — from uv.lock, so you get the versions CI actually tested
uv sync --frozen --extra dev
source .venv/bin/activate
python scripts/download_models.py               # optional: pre-warm ASR + check servers

# 3. Configure + start the API (binds 127.0.0.1:8010)
cp .env.example .env                            # already defaults to the Ollama path above
framesleuth-api                                 # or: uvicorn framesleuth.service.api:app --port 8010

# 4. Verify  (says so either way — a silent command is not a passing check)
curl -s http://127.0.0.1:11434/v1/models | grep -q qwen2.5vl \
  && echo "VLM ready" || echo "VLM NOT ready — run: ollama pull qwen2.5vl"
curl -s http://127.0.0.1:8010/v1/healthz | python -m json.tool   # status: healthy, vlm: ready
```

When `/v1/healthz` shows `vlm: ready`, recordings get a real classification
(`analysis_quality.level` of `full` or `partial`). `ready` means the server answered *and*
listed your `VLM_MODEL`. If the model was never pulled you get `vlm: degraded` with
`model '<name>' not loaded` instead.

With no vision model reachable at all, Framesleuth degrades gracefully. It still produces
a valid Context Bundle from the browser sidecars (console errors, failed requests, clicks)
and records what was thin in `analysis_quality`. Narrate while you record and the audio
transcript (`asr`) stage contributes too.

> Something not working? Run the setup doctor. It runs under a plain `python3` even when
> your virtualenv is broken, and prints a one-line fix for each problem: a stale or
> missing venv, `framesleuth-api` not on PATH, ffmpeg and render prerequisites, an
> unreachable backend or model server, a wrong `VLM_URL`.
>
> ```bash
> python3 scripts/doctor.py
> ```
>
> The common one: `command not found: framesleuth-api`, or a `uv pip install` error about
> a missing interpreter, means your active venv was deleted or moved. Fix it from the
> framesleuth directory:
> `deactivate; unset VIRTUAL_ENV; uv sync --frozen --extra dev; source .venv/bin/activate`.

**Stop**

```bash
# Stop the backend: Ctrl+C in its terminal, or
pkill -f framesleuth-api

# Stop Ollama (optional — leaving it running keeps the model warm)
pkill -f "ollama serve"              # macOS app users: quit Ollama from the menu bar
```

## Architecture

```
Any video (mp4/webm) + optional sidecars
    ↓
Local Analysis Service (pipeline)
    ├─ Preprocess (PyAV: duration/fps/dims)
    ├─ Transcript (faster-whisper)
    ├─ Keyframes (visual-delta change scoring)
    ├─ Understanding (local vision model — Qwen2.5-VL by default)
    ├─ Fusion + Classification
    ├─ Extraction → Context Bundle
    ├─ Summarize (skill/system-prompt-driven)
    └─ Grounding (workspace search)
    ↓
Context Bundle
    ↓
MCP server + local HTTP API
    └─ consumed by any MCP client (VS Code agent, other agents, capture extension)
```

## Features

- **Frame-by-frame understanding** with a local vision model (Qwen2.5-VL by default; engine-agnostic)
- **Adaptive keyframe selection.** Coverage-binned and visual-salience-ranked (AKS-style), with a build-aware budget for feature and design videos. Perceptual-hash dedup drops near-identical frames so the VLM budget goes on distinct content.
- **Bug *and* build.** A `feature` class plus a structured build context: screens, UI components, a screen-to-screen user flow, design notes, and where to implement. An agent can build from it, not only diagnose.
- **Error detection and extraction** from console, OCR, and UI state
- **Corpus-aware grounding.** Error symbols or feature/UI nouns resolve to ranked `file:line` hits. Definitions are preferred, distinctive symbols weighted via IDF plus whole-word match, `.gitignore` respected, and the search bounded for large repos.
- **Trust signals.** Per-field confidence, where agreeing signals across modalities corroborate each other, plus a task-aware `actionability` (ready/thin/insufficient) alongside the pipeline quality level.
- **Redaction-first design.** Secrets (passwords, tokens, keys) and PII (emails, Luhn-valid card numbers, SSNs/phones, cloud keys) are scrubbed from OCR, captions, the transcript, and the raw sidecar streams **before** any of it reaches a model or is persisted. That covers the bundle and the sibling `timeline.json`, `sidecars.json` and `transcript.json`.
- **Observability.** Per-stage timings land on every bundle (`stage_timings`) and live on `GET /v1/jobs/{id}`, so you can see where analysis time went.
- **Job lifecycle and delivery.** Cooperative cancellation (`DELETE /v1/jobs/{id}`, checked between frames), a hard per-job timeout (`JOB_TIMEOUT_S`), crash recovery that fails orphaned jobs on restart rather than leaving zombies, SSE progress with explicit terminal events (`GET /v1/jobs/{id}/events`), a completion webhook (`WEBHOOK_URL`), real queue depth in `/healthz`, and TTL retention cleanup (`BUNDLE_TTL_DAYS`) swept at startup and periodically (`RETENTION_SWEEP_INTERVAL_S`).
- **Interaction overlay.** A click/cursor sidecar with coordinates draws a marker on the matching keyframe, so the model sees *where* the user acted.
- **Cleaner transcripts.** faster-whisper voice-activity filtering (`ASR_VAD_FILTER`) drops silence before decoding; the detected or forced language is recorded.
- **OCR backstop** *(optional `ocr` extra)*. A sparse VLM OCR on an error frame gets a second, independent Tesseract reading. Without the extra it's a no-op.
- **No data leaves your machine.** Fully local, no telemetry, no cloud APIs.
- **Engine-agnostic.** Swap Ollama, llama.cpp, or vLLM via config only.
- **Works on *any* video**, not just bug recordings. A demo, a walkthrough, a talk, a
  phone clip: each yields a faithful summary and a timeline of key moments (`summary`,
  `key_moments[]`) rather than something forced into a bug shape. The bug-only fields
  (severity, expected/actual, repro steps) stay `null` instead of carrying fabricated
  placeholders.
- **Structured output.** A canonical Context Bundle with evidence citations.
- **Configurable response.** Pick a summary skill and an action mode
  (`fix`/`implement`/`design`/`summarize`/`explain`/`triage`/`test`/`report`/`reproduce`,
  auto-picked from the classification), plus a machine-readable `suggested_actions` menu
  and on-demand artifact renderers (markdown, GitHub issue, test plan).
- **Eval harness.** Model-free classification, grounding, citation and faithfulness
  suites (`python scripts/eval_harness.py --behavioral`) run in CI on every push and PR:
  a GitHub Actions 3.11/3.12 matrix of ruff, black, mypy `--strict`, pytest behind a
  coverage gate, the eval harness against per-metric thresholds in `evals/`, and an
  OpenAPI-freshness check, plus a separate security job running `pip-audit` and
  `pre-commit`. The faithfulness suite proves every emitted key moment and step cites
  real, resolvable evidence.
- **Resilient.** Handles no-audio videos, weak local models, and low-confidence cases.
- **HTML → video (frame-by-frame).** Turn a self-contained HTML animation (CSS/JS/canvas)
  into MP4, GIF, or WebM via the `render_html_video` MCP tool or `POST /v1/render-html`.
  Frames are captured one at a time under a paused virtual clock and encoded to a
  color-correct H.264 MP4 (`yuv420p`+`bt709`, near-lossless): full color, no dropped
  frames, no quality loss, up to 4K and 5–60 fps. The Docker image includes it by default
  (headless Chromium + ffmpeg). On the direct path, add the `render` extra (see below);
  without it the endpoint returns `503` with an actionable message.

### Enable & troubleshoot HTML → video

> On **Docker** (`docker compose up`) this already works; the image bakes in Playwright,
> Chromium and ffmpeg. Build with `--build-arg INSTALL_RENDER=false` for a slimmer image
> without it. The steps below are for the direct path.

Playwright lives in an optional `[render]` extra rather than core, because it pulls a
~150 MB headless-Chromium browser the video→bundle pipeline never needs. (`av`, `opencv`
and `faster-whisper` are core.) Install the extra and you're done. The Chromium build
downloads on your first render, so there's no separate `playwright install chromium`
step:

```bash
# In the same environment the server runs in:
uv sync --frozen --extra dev --extra render   # or --all-extras
# ffmpeg must be on PATH (brew install ffmpeg / apt-get install ffmpeg)

# Restart framesleuth-api, then verify (Chromium fetches itself on first render):
curl -s http://127.0.0.1:8010/v1/healthz | python -m json.tool
# → "render": {"playwright": true, "chromium": <true after first render>, "ffmpeg": true}
```

Set `FRAMESLEUTH_AUTO_INSTALL_BROWSER=0` to disable the auto-download and run
`playwright install chromium` yourself, e.g. in a locked-down environment.

> The other optional extra is `ocr`. For the dedicated OCR backstop on error frames, run
> `uv sync --frozen --extra dev --extra ocr` and put the `tesseract` binary on PATH
> (`brew install tesseract` / `apt-get install tesseract-ocr`). Absent, it's a no-op: the
> VLM still does OCR, and the backstop only adds a second reading. Use `".[all]"` for
> dev + render + ocr.

If `render.ready` is `false`, ask `/v1/version` for the details. `/v1/healthz` is public,
so it omits the `hint` and `python` fields rather than publish the server's filesystem
layout to an unauthenticated caller:

```bash
curl -s http://127.0.0.1:8010/v1/version | python -m json.tool
# → "render": {"ready": false, "hint": "...", "python": "/path/to/the/interpreter", ...}
# With API_TOKEN set, this endpoint is token-gated:
#   curl -s -H "Authorization: Bearer $API_TOKEN" http://127.0.0.1:8010/v1/version
```

`render.hint` tells you what's missing. When you followed the steps and still get
*"Playwright is not installed"*, it's usually one of two things: `framesleuth-api` is
running from a different environment than the one you installed into (`render.python`
names the interpreter it uses), or the server wasn't restarted.

## Project structure

```
framesleuth/
├── framesleuth/              # Main package
│   ├── config.py            # Typed config (pydantic-settings)
│   ├── schemas.py           # Data contracts (Context Bundle, enums)
│   ├── errors.py            # Exception taxonomy
│   ├── logging_config.py    # Structured JSON logging, job-id correlation
│   ├── prompts.py           # VLM / classify / summary / fix prompt templates
│   ├── skills.py            # Built-in summary skills (summary, bug_report, ...)
│   ├── actions.py           # Action modes (fix/explain/triage/...) + suggested-actions menu
│   ├── render.py            # Artifact renderers (markdown / GitHub issue / test plan)
│   ├── clients/             # VLM, coder HTTP clients (OpenAI-compatible)
│   ├── pipeline/            # preprocess, asr, scenes, understand, fusion, classify,
│   │                        #   bug_extract, build_context, confidence, dedup, overlay,
│   │                        #   ocr, redact, summarize, sidecars, grounding, gif,
│   │                        #   atomic, html_render
│   ├── eval/                # harness.py — model-free behavioral suites
│   ├── orchestrator/        # graph.py — linear async stage pipeline
│   ├── jobs/                # store.py — SQLite job state + bundle index
│   ├── service/             # FastAPI HTTP endpoints
│   └── mcp_server/          # framesleuth MCP server (VS Code + any MCP client)
├── tests/                   # pytest tests + fixtures
├── scripts/                 # doctor.py (setup check), download_models.py, dev_up.sh,
│                            #   eval_harness.py, export_openapi.py
├── evals/                   # thresholds.json + baseline.json (the CI quality gate)
├── openapi.json             # generated API schema — the contract clients build from
├── postman/                 # HTTP API collection + environment
├── docs/                    # capabilities, use-with-vscode-and-claude, web-integration
└── pyproject.toml           # Dependencies and tool config
```

## Development

### Run tests
```bash
pytest tests/ -q                                        # fast: no coverage gate
pytest tests/ -q --cov=framesleuth --cov-fail-under=75  # what CI enforces
```

### Regenerate the API schema (after changing any route)
```bash
python scripts/export_openapi.py --out openapi.json
```
CI fails if this file is stale; the website generates its typed client from it.

### Run the eval gates
```bash
python scripts/eval_harness.py --behavioral   # see evals/README.md
```

### Code quality
```bash
ruff check framesleuth tests
black --check framesleuth tests
mypy --strict framesleuth
```

### Set up pre-commit hooks
```bash
pre-commit install
```

Docs, a short and focused set:

- [Capabilities](docs/capabilities.md). One reference for every input, output, skill, action, renderer, HTTP endpoint and MCP tool.
- [Use with VS Code & Claude (MCP)](docs/use-with-vscode-and-claude.md). Connect the `framesleuth` MCP server to Copilot, Claude Code, and Claude Desktop.
- [Web App Integration (end-to-end)](docs/web-integration.md). Embed Framesleuth behind your own backend with an agent loop.
- [Postman Collection](postman/README.md). Exercise the HTTP API end to end, in the app or headless with Newman.
- [Runbook & Troubleshooting](runbook.md). Setup, health checks, and common issues.

## License

Apache-2.0

---

## Capture client

Bug capture lives outside this repo. Any screen recording works, so you can drive the
agent with your own video file. A browser capture extension can also record a session,
collect browser sidecars (console errors, failed requests, clicks), and post the video
plus sidecars to this agent's local API.

CORS is an exact allowlist. The local dev origins `http://localhost:3000` and
`http://127.0.0.1:3000` are on by default; set `ALLOW_LOCAL_DEV_ORIGINS=false` on a
hardened deployment to drop them. `chrome-extension://` origins come from the IDs you
list in `CHROME_EXTENSION_IDS`, empty by default, so a capture extension has to add its
own. Everything else goes in `WEB_ORIGINS`, also **empty by default: no remote site is
trusted, framesleuth.com included**. The agent answers Chrome's Private Network Access
preflight, so an allowed origin can drive a backend running locally.

To let the hosted "Try it" widget talk to your agent, opt in explicitly:

```bash
WEB_ORIGINS=https://framesleuth.com,https://www.framesleuth.com
```

The agent stays bound to loopback; CORS only controls which browser origins may *read*
its responses.

**Set `API_TOKEN` for anything beyond a single-user laptop.** With a token set, every
`/v1` endpoint except `/v1/healthz` requires `Authorization: Bearer <token>`. CORS won't
stop another local process, or a DNS-rebinding page, from *sending* requests to loopback.
A token will. The Docker stack reads it from `.env`, and publishes the API on
`127.0.0.1` only.

Status: backend, pipeline and MCP server are complete.

Questions? Open an issue, or check [runbook.md](runbook.md) for common ones.
