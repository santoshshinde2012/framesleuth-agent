# Framesleuth capabilities

The single reference for **everything the agent can do** — its inputs, its outputs,
and every surface it exposes. For setup and deeper walkthroughs, follow the links at
the end.

> **What it is, in one line:** Framesleuth takes *any* video — a bug recording, a
> feature demo, a design walkthrough, a Loom, a phone capture (plus optional browser
> sidecars) — and produces a structured, evidence-cited **Context Bundle**, then
> exposes it over an HTTP API and an MCP server so any coding agent can read it and
> act: **fix a bug, add or change a feature, or build something new**. Everything
> runs locally; nothing leaves the machine.

---

## 1. What the pipeline does

A single linear pass (with one bounded resample step). A missing *model* degrades
gracefully — the run still produces a valid bundle and says what was thin via
`analysis_quality`. **Rejected input does not degrade**: an unreadable container or a
recording longer than `MAX_DURATION_S` fails the job with `unsupported_media` /
`duration_exceeded`, rather than silently producing a single-frame bundle that looks
like a successful analysis.

| Stage | Capability |
|---|---|
| **Preprocess** | Probe the container (recovers duration even from header-less browser WebM); sample frames at a bounded budget; **perceptual-hash dedup** collapses near-identical keyframes before the VLM. |
| **Transcribe (ASR)** | Local `faster-whisper` narration → timestamped transcript; **voice-activity filter** drops silence and the detected/forced `language` is recorded. |
| **Understand (VLM)** | Per-keyframe `caption` / `ocr_text` / `ui_action` / `is_error_state` / `reason`, run **concurrently**. Click/cursor sidecars are **overlaid** onto frames; sparse error frames are **re-OCR'd at full resolution** (with an optional dedicated **OCR backstop**). |
| **Classify** | Deterministic, auditable bug/feature/tutorial/demo/feedback/other scoring (build/feature intent comes from the request + narration); an **ambiguous** result triggers a bounded **resample** + an optional **model tie-breaker**. |
| **Build context** | For feature/build/demo videos: structured UI extraction (components, layout, screen names, design notes) + a screen-to-screen user flow → a buildable spec. |
| **Summarize** | A skill-shaped narrative (video + audio) plus distilled, timestamped **`key_moments`** — the deliverable for a general (non-bug) video. |
| **Extract** | Assemble the canonical Context Bundle with evidence citations, an `analysis_quality` trust signal, per-field confidence (with cross-modal corroboration), and `stage_timings`. |
| **Redact** | Strip secrets (tokens, keys, passwords) **and PII** (emails, Luhn-valid cards, SSNs/phones, cloud keys) from text **before** persistence. |
| **Ground** | Read-only, `.gitignore`-aware workspace search → ranked candidate `file:line` (definitions preferred, IDF + whole-word + camel/snake query expansion), bounded for large repos. |
| **Resolve action** | Pick what a downstream agent should *do* (see Actions), auto-selected from the classification (general video → `summarize`). |

---

## 2. Inputs

Supplied to `POST /v1/analyze` (form fields) or the `analyze_video` MCP tool (args).

| Input | Required | Purpose |
|---|---|---|
| `video` | ✅ | The recording (`.mp4` / `.webm` / `.mkv` / `.mov` / `.avi`). |
| `sidecars` | — | JSON array of browser events (console errors, failed network requests, clicks, env). Strong evidence even with no vision model. |
| `intent` | — | The user's request in their own words; recorded and woven into the fix-prompt. |
| `skill` | — | Summary **style** (see Skills). Defaults to `summary`. |
| `system_prompt` | — | Fully custom summary system prompt; overrides `skill`. |
| `action` | — | What the agent should **do** (see Actions). Auto-picked from the classification when omitted. |
| `action_prompt` | — | Fully custom action task; overrides `action`. |
| `capture_options` | — | (HTTP only) Arbitrary capture metadata, stored beside the bundle for provenance. |
| `repo_root` | — | Workspace to ground error text against. Over **HTTP** this is refused with `403 grounding_disabled` unless the operator sets `GROUNDING_ROOTS`, because grounding reads source files and returns matching lines to the caller. Over **MCP** the caller is your own editor, so only the path validity is checked. |

**Limits:** `MAX_UPLOAD_MB` (default 512), `MAX_DURATION_S` (default 600 s),
`MAX_QUEUED_JOBS` (default 32). Exceeding the duration or uploading something that is
not a video container **fails the job** (`duration_exceeded` / `unsupported_media`) —
it is not degraded.

**Idempotency:** identical bytes *and identical options* (intent, skill, action,
custom prompts, sidecar content, repo root) return the cached report. Changing any
option produces a new job.

---

## 3. Output — the Context Bundle

`GET /v1/report/{id}` (or `get_report`) returns this stable, versioned JSON.

| Field | What it is |
|---|---|
| `id`, `schema_version`, `source_video`, `duration_s`, `created_at` | Provenance. |
| `classification` | `label` (bug/feature/tutorial/demo/feedback/other) + `confidence` + `alt_labels`. |
| `title`, `severity`, `priority`, `suspected_component`, `reproducibility` | Triage headline. **Bug-only** — `null` for a general (non-bug) video. |
| `environment` | OS / app / browser / version from sidecars or OCR. |
| `preconditions`, `expected_behavior`, `actual_behavior` | The behavioral story. **Bug-only** — `null` for a general video (no fabricated placeholders). |
| `repro_steps[]` | Numbered, **evidence-cited** steps. Observed steps for general video; empty when none were shown. |
| `summary`, `key_moments[]` | The **deliverable for any video**: a narrative summary plus salient timestamped moments (`t` / `label` / `kind` of scene·action·speech·error / `evidence`). |
| `error_evidence[]` | Timestamped console / OCR / network / UI errors. |
| `keyframe_refs[]` | The frames the model read (resolvable images). |
| `code_candidates[]` | Ranked `file:line` locations from grounding (definitions preferred). |
| `build_context` | For feature/build/demo videos: `screens[]`, `components[]`, `user_flow[]`, `design_notes[]`, `data_models[]`, `is_greenfield`, `target_locations[]` — a buildable spec. Null for pure bugs. |
| `field_confidence` | Per-field confidence 0-1 (title, repro_steps, severity, build_context…) so consumers know which claims to trust. |
| `skill` | The summary style/skill used to produce `summary`. |
| `action`, `action_prompt` | Resolved response mode (and custom task, if any). |
| `suggested_actions[]` | Machine-readable next-step menu (`action` / `label` / `rationale` / `ref`). |
| `user_intent` | The request you passed. |
| `analysis_quality` | **Trust signal** — `level` (`full`/`partial`/`degraded`) + `degraded_stages` + `warnings` + `evidence_counts` + `actionability` (`ready`/`thin`/`insufficient` for the resolved action). Read this first. |
| `stage_timings` | Per-stage wall-clock seconds (`preprocess`, `asr`, `understand`, `summarize`, `grounding`…) — see where the analysis time went. Also surfaced live on `GET /v1/jobs/{id}` as `metrics`. |
| `provenance` | Which models, prompt version + fingerprint, agent version, and workspace `repo_root`/`git_commit` produced this bundle — so a quality change can be attributed to a prompt edit or a model swap, and `code_candidates` line numbers stay verifiable later. |
| `redactions[]` | What was scrubbed (secrets **and** PII: emails, Luhn-valid cards, SSNs/phones, cloud keys). |
| `transcript_path`, `timeline_path` | Sibling artifacts (`transcript.json`, `timeline.json`). |

Sibling files written next to `bundle.json`: `source.<ext>`, `keyframes/*.png`,
`transcript.json`, `timeline.json`, `sidecars.json`, `metrics.json`, plus
`capture_options.json` (when that form field was supplied) and any cached
`preview-<key>.gif`. Every one is written atomically, so a crash mid-write can never
leave truncated JSON at a path a reader trusts.

---

## 4. Skills — how the summary reads

Pick with `skill`, or override with `system_prompt`. List live via `GET /v1/skills`.

| Skill | Output |
|---|---|
| `summary` *(default)* | Narrative + ordered steps + any issues. |
| `bug_report` | QA bug report (title, repro, expected/actual, severity). |
| `tutorial` | Step-by-step how-to of what was demonstrated. |
| `action_items` | Decisions + follow-ups (owners when shown). |
| `release_notes` | Short, user-facing change notes. |

---

## 5. Actions — what the agent should do

Pick with `action`, or override with `action_prompt`. List live via `GET /v1/actions`.
With no `action`, one is **auto-picked** from the classification (bug→`fix`,
feature→`implement`, tutorial/demo→`explain`, feedback→`report`, other→`summarize`).

| Action | The downstream agent is told to… |
|---|---|
| `fix` | Diagnose the root cause and propose/make a minimal fix. |
| `implement` | Build or extend the feature shown, using the build context as a spec. |
| `design` | Propose a UI/component/data design from what was shown — no code yet. |
| `summarize` | Summarize/analyze **any** video — overview, key moments, takeaways. The default for a general (non-bug) video. |
| `explain` | Explain what happened — no code changes. |
| `triage` | Assess severity/priority and route it — no fix. |
| `test` | Write a failing regression test that reproduces it. |
| `report` | Produce a ready-to-paste issue/PR description. |
| `reproduce` | Produce minimal exact steps / a script to reproduce. |

---

## 6. Renderers — shareable artifacts

Turn any finished report into a consumable artifact (no new info — projected from the
bundle). Via the `render(report_id, format)` MCP tool or the resources below.

| Format | Output |
|---|---|
| `markdown` | A shareable markdown report. |
| `issue` | GitHub-issue text (title + labels + body). |
| `test-plan` | A framework-agnostic regression test plan. |

### HTML animation → video (frame-by-frame)

A separate capability (`POST /v1/render-html` / the `render_html_video` MCP tool):
turn a self-contained HTML/CSS/JS animation into a downloadable clip. It captures
the animation **frame-by-frame** in headless Chromium using a paused virtual clock
(advanced one frame budget at a time), so every frame is a lossless full-resolution
PNG at an exact timestamp — **no dropped frames and no color loss**, unlike screen
recording. The PNG sequence is then encoded to a color-correct **H.264 MP4**
(`yuv420p` + `bt709`, near-lossless CRF, `+faststart`), **VP9 WebM**, or a
palette-based **GIF**. Output up to **4K**, **5–60 fps**, ≤ 30 s. If deterministic
capture is unavailable on the running Chromium it falls back to real-time recording.
Optional capability — needs the `render` extra (Playwright) + `ffmpeg`; Chromium
auto-downloads on first use.

---

## 7. HTTP API

Loopback only (`127.0.0.1:8010` by default). Front it with your own backend; never
expose it to the browser/internet directly.

| Method · Path | Returns |
|---|---|
| `GET /v1/healthz` | VLM / coder / storage readiness + overall `status`, plus a `render` block for optional HTML→video readiness (`playwright`, `playwright_version`, `chromium`, `ffmpeg`, `ready`). **Always public**, even when `API_TOKEN` is set — so a client can tell "agent down" from "agent needs a token". For that reason the verbose `python`/`hint` diagnostics are never included here; they live on the token-gated `/v1/version`. |
| `GET /v1/version` | Build + **capability advertisement** — `{version, schema_version, api_version, capabilities}`. Feature-detect once instead of probing endpoints and guessing whether a 404/503 means "unsupported". |
| `GET /v1/skills` | Built-in summary skills + default. |
| `GET /v1/actions` | Built-in action modes + default + `auto` flag. |
| `POST /v1/analyze` | Queues the pipeline (background, bounded by `MAX_CONCURRENT_JOBS`) → `202 {job_id, status: "queued", idempotent}` (`idempotent` is a **boolean**). Errors: `400 bad_sidecars` (malformed sidecar JSON), `403 grounding_disabled` (a `repo_root` without `GROUNDING_ROOTS`), `413 upload_too_large`, `415 unsupported_media` (the bytes are not a video container — checked at ingest, before anything is queued), `429 queue_full` (more than `MAX_QUEUED_JOBS` waiting). Poll `/v1/jobs/{id}` (or stream `/events`) for completion. |
| `GET /v1/jobs` | Paginated job history, newest first. Filter with a repeatable `state=` query param. |
| `GET /v1/jobs/{id}` | Lifecycle state + progress + error + per-stage `metrics`. |
| `GET /v1/jobs/{id}/events` | **Server-Sent Events** progress stream (JSON snapshots) that closes on a terminal state — follow a job without polling. |
| `DELETE /v1/jobs/{id}` | Request **cooperative cancellation**; the pipeline stops at the next stage boundary and the job becomes `cancelled`. `409` if already terminal. |
| `GET /v1/report/{id}` | The full Context Bundle. |
| `GET /v1/report/{id}/render?format=` | The bundle projected as `markdown` / `issue` / `test-plan` text — the same renderers the MCP `render` tool calls, so the two surfaces cannot drift. Add `download=true` for a file response. |
| `GET /v1/report/{id}/prompt` | The grounded, evidence-only action prompt (the MCP `fix-prompt` resource over HTTP). |
| `GET /v1/report/{id}/keyframes/{index}` | One analyzed keyframe image (PNG). The bundle cites `frame:N` throughout; this resolves N. |
| `GET /v1/report/{id}/timeline` | The fused, cited event timeline (scenes + transcript + console/network). |
| `GET /v1/report/{id}/transcript` | The redacted audio transcript. |
| `DELETE /v1/report/{id}` | Purge a finished report (bundle directory + job row). |
| `GET /v1/video/{id}` | The stored source recording (correct media type). |
| `GET /v1/gif/{id}` | An animated GIF preview of the recording (`image/gif`). Optional `fps`/`width`/`start`/`end` query params (clamped); rendered on demand and cached on disk per parameter set. |
| `POST /v1/render-html` | Render an HTML document (CSS/JS/canvas animation) to a clip — **frame-by-frame, full color, no quality loss** (up to 4K, 5–60 fps). JSON body `{html, format: mp4\|gif\|webm, duration_s, fps, width, height}`; returns the encoded file. Optional capability — needs the `render` extra (Playwright) + `ffmpeg`; returns `503` with an actionable message when unavailable. Check `GET /v1/healthz` → `render.ready` first. |

---

## 8. MCP server (`framesleuth`)

Stdio (`framesleuth-mcp`). No tool ever **edits** your workspace — source changes
happen only through the calling agent's reviewed apply flow. Workspace *reads* are
confined to an explicit `repo_root`, never inferred from the server's working
directory. Run
`framesleuth-mcp --print-config` to print a ready-to-paste client config with an
**absolute** command path (works in any client/scope; avoids the
`${workspaceFolder}` ENOENT gotcha — see docs/use-with-vscode-and-claude.md).

**Tools (14):** `analyze_video`, `list_skills`, `list_actions`, `list_reports`,
`get_report(view=full|slim)`, `get_suggested_actions`, `get_repro_steps`,
`get_error_evidence`, `get_timeline`, `get_keyframe_image`,
`get_video_gif(fps,width,start,end)`, `locate_in_code`, `render(format)`,
`render_html_video(html,format,duration_s,fps,width,height)`.

Every tool declares MCP **annotations** (`readOnlyHint` / `destructiveHint` /
`idempotentHint` / `openWorldHint`), so a client knows which calls are safe to
make without asking the user. The eleven pure readers are `readOnly`;
`analyze_video`, `get_video_gif` and `render_html_video` write only inside
Framesleuth's own bundle directory and are marked non-destructive. The server
also ships an `instructions` block describing the analyze → report → act flow.

**Resources (4):** `framesleuth://report/{id}/summary` · `…/fix-prompt` · `…/markdown` ·
`…/issue`.

**Prompt (1):** `fix_from_video(report_id)` — the grounded, action-aware action prompt.

See [Use with VS Code & Claude](use-with-vscode-and-claude.md) for client setup.

---

## 9. Cross-cutting capabilities

- **Graceful degradation** — every stage can fail independently; the bundle stays
  well-formed and `analysis_quality` reports `full` / `partial` / `degraded` with
  warnings, so a thin recording is never mistaken for "nothing wrong".
- **Evidence by structure** — every claim carries a citation (frame / sidecar /
  transcript); uncited claims are dropped.
- **Redaction-first** — secrets **and PII** (emails, Luhn-valid cards, SSNs/phones,
  cloud keys) are scrubbed from text before anything is persisted or returned
  (`REDACT_PII`; text-only — pixel-level redaction is not yet built).
- **Interaction overlay** — a click/cursor sidecar with coordinates draws a marker on
  the matching keyframe so the model sees *where* the user acted (`OVERLAY_INTERACTIONS`).
- **OCR backstop** *(optional `ocr` extra)* — a sparse VLM OCR on an error frame gets a
  second, independent reading from Tesseract; a no-op when the extra is absent.
- **Better transcripts** — faster-whisper's voice-activity filter (`ASR_VAD_FILTER`)
  removes silence before decoding, and the detected/forced `language` is recorded.
- **Corpus-aware grounding** — `.gitignore`-respecting, IDF + whole-word ranked, with
  camelCase/snake-case query expansion and a `GROUNDING_MAX_FILES` bound.
- **Engine-agnostic** — swap Ollama / llama.cpp / vLLM via config (`VLM_URL`,
  `VLM_MODEL`, …); no code change.
- **Job lifecycle** — cancel (`DELETE`), SSE progress (`/events`), a completion
  **webhook** (`WEBHOOK_URL`), and TTL retention cleanup (`BUNDLE_TTL_DAYS`).
- **Idempotent & self-cleaning** — identical uploads *with identical options* return
  the cached report; per-job scratch files are removed after each run.
- **Local & private** — all models run on `localhost`; no telemetry, no cloud calls.

### Error responses

Every non-2xx response is the same flat shape — no `detail` wrapper:

```json
{ "error": "Job abc not found", "code": "job_not_found",
  "hint": "Check that the job ID is correct.", "request_id": "12f825a4-…" }
```

Branch on `code`, never on `error`. `request_id` also comes back as the
`X-Request-ID` response header (exposed to browsers via CORS), so one value ties a
user's bug report to the exact server-side log lines. Send your own `X-Request-ID`
and it is echoed back.

Tuning knobs (concurrency, JSON/schema mode, token caps, resample, classification
tie-breaker, frame resolution, keyframe dedup, ASR model/VAD, grounding roots, queue
caps, retry, retention, webhook) are documented in [`.env.example`](../.env.example).

---

## Where to go next

- [Use with VS Code & Claude (MCP)](use-with-vscode-and-claude.md) — connect an agent + the end-to-end test recipe.
- [Web App Integration](web-integration.md) — embed Framesleuth behind your own backend with an agent loop.
- [Postman Collection](../postman/README.md) — exercise the HTTP API end-to-end.
- [Runbook & Troubleshooting](../runbook.md) — setup, health checks, and common issues.
