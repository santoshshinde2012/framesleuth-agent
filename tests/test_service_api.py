"""Tests for FastAPI service endpoints."""

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from framesleuth.config import Settings
from framesleuth.schemas import JobState
from framesleuth.service.api import create_app

_EXT_ID = "abcdefghijklmnopabcdefghijklmnop"

# /v1/analyze sniffs the first chunk for a container signature, so a payload of
# arbitrary bytes is now rejected at ingest with 415. These helpers produce the
# smallest input that gets past that gate: a real ISO-BMFF `ftyp` box header.
_MP4_MAGIC = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


def _video(payload: bytes = b"") -> bytes:
    """Minimal bytes that sniff as an MP4, plus optional distinguishing content."""
    return _MP4_MAGIC + payload


def _upload(name: str = "sample.mp4", payload: bytes = b"") -> dict[str, tuple[str, bytes, str]]:
    """A ready-to-post ``files=`` mapping for /v1/analyze."""
    return {"video": (name, _video(payload), "video/mp4")}


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        BUNDLE_DIR=tmp_path / "bundles",
        DATABASE_PATH=tmp_path / "jobs.db",
        MAX_UPLOAD_MB=10,
        CHROME_EXTENSION_IDS=_EXT_ID,
    )


def _wait_terminal(client: TestClient, job_id: str, timeout_s: float = 5.0) -> dict:
    """Poll a job until it reaches a terminal state, then return it.

    Analyses now run as tasks the app owns (so shutdown can drain them) rather
    than as Starlette BackgroundTasks pinned to the request, so completion is
    genuinely concurrent with the response and a test must wait for it instead
    of assuming the next call sees a finished job.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["state"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached a terminal state: {job['state']}")


def test_analyze_and_report_roundtrip(tmp_path: Path) -> None:
    """Analyze endpoint should create idempotent job and report should be retrievable."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(
            json.dumps({"id": job_id, "title": "test", "source_video": source_video}),
            encoding="utf-8",
        )
        (bundle_dir / "metrics.json").write_text(
            json.dumps({"stages": {"preprocess": 0.1, "understand": 1.2}, "degraded": []}),
            encoding="utf-8",
        )
        await state.store.update_job(
            job_id,
            state=JobState.DONE,
            progress_pct=100,
            bundle_path=str(bundle_path),
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run

        response = client.post(
            "/v1/analyze",
            files=_upload("sample.mp4", b"video-content"),
        )
        # Analysis is accepted (202) and runs in the background; the TestClient
        # drains the background task before returning, so the job is done by now.
        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "queued"
        job_id = payload["job_id"]

        job = client.get(f"/v1/jobs/{job_id}")
        assert job.status_code == 200
        assert job.json()["state"] == "done"
        # Per-stage metrics are surfaced to the poller for observability.
        assert job.json()["metrics"]["stages"]["understand"] == 1.2

        report = client.get(f"/v1/report/{job_id}")
        assert report.status_code == 200
        assert report.json()["id"] == job_id

        # The temp upload is cleaned up, not left accumulating in the bundle dir.
        assert not list(app.state.app_state.settings.BUNDLE_DIR.glob("upload-*"))

        # Same upload should hit idempotency path.
        second = client.post(
            "/v1/analyze",
            files=_upload("sample.mp4", b"video-content"),
        )
        assert second.status_code == 202
        assert second.json()["idempotent"] is True


def test_analyze_records_typed_failure_in_background(tmp_path: Path) -> None:
    """A typed pipeline failure marks the job FAILED with a structured error."""
    from framesleuth.errors import UnsupportedMediaError

    app = create_app(_make_settings(tmp_path))

    async def boom(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        raise UnsupportedMediaError("bad codec")

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = boom
        resp = client.post("/v1/analyze", files=_upload("x.mp4", b"vid"))
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        job = _wait_terminal(client, job_id)
        assert job["state"] == "failed"
        assert job["error"]["code"] == "unsupported_media"

        # The temp upload is still cleaned up on the failure path.
        assert not list(app.state.app_state.settings.BUNDLE_DIR.glob("upload-*"))


def test_get_video_serves_correct_media_type(tmp_path: Path) -> None:
    """The stored source is served with a content-type matching its container."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        # An mp4 source must not be advertised as webm.
        (bundle_dir / "source.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post("/v1/analyze", files=_upload("bug.mp4", b"vid")).json()["job_id"]
        _wait_terminal(client, job_id)

        video = client.get(f"/v1/video/{job_id}")
        assert video.status_code == 200
        assert video.headers["content-type"] == "video/mp4"


def _write_sample_video(path: Path, *, frames: int = 16, fps: int = 8) -> None:
    """Encode a tiny synthetic mp4 for endpoints that need a real recording."""
    import av
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 128
        stream.height = 96
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            arr = np.full((96, 128, 3), (i * 12) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_get_gif_renders_and_caches_preview(tmp_path: Path) -> None:
    """GET /v1/gif encodes a GIF from the stored source and caches it on disk."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        _write_sample_video(bundle_dir / "source.mp4")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post("/v1/analyze", files=_upload("bug.mp4", b"vid")).json()["job_id"]
        _wait_terminal(client, job_id)

        resp = client.get(f"/v1/gif/{job_id}", params={"fps": 6, "width": 96})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/gif"
        assert resp.content[:6] in (b"GIF87a", b"GIF89a")

        # The render is cached on disk keyed by its parameters.
        bundle_dir = app.state.app_state.settings.BUNDLE_DIR / job_id
        cached = list(bundle_dir.glob("preview-*.gif"))
        assert len(cached) == 1

        # A second identical request reuses the cache (still 200, same file).
        again = client.get(f"/v1/gif/{job_id}", params={"fps": 6, "width": 96})
        assert again.status_code == 200
        assert len(list(bundle_dir.glob("preview-*.gif"))) == 1


def test_get_gif_returns_404_when_no_source(tmp_path: Path) -> None:
    """A job without a stored recording yields a 404, not a 500."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post("/v1/analyze", files=_upload("bug.mp4", b"vid")).json()["job_id"]
        _wait_terminal(client, job_id)

        resp = client.get(f"/v1/gif/{job_id}")
        assert resp.status_code == 404
        assert resp.json()["code"] == "missing_video"


def test_skills_endpoint_lists_builtins(tmp_path: Path) -> None:
    """GET /v1/skills returns the default and the built-in catalog."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        r = client.get("/v1/skills")
        assert r.status_code == 200
        body = r.json()
        assert body["default"] == "summary"
        names = {s["name"] for s in body["skills"]}
        assert {"summary", "bug_report", "tutorial"} <= names


def test_actions_endpoint_lists_builtins(tmp_path: Path) -> None:
    """GET /v1/actions returns the default, auto flag, and the built-in catalog."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        r = client.get("/v1/actions")
        assert r.status_code == 200
        body = r.json()
        assert body["default"] == "fix"
        assert body["auto"] is True
        names = {a["name"] for a in body["actions"]}
        assert {"fix", "explain", "triage", "test", "report", "reproduce"} <= names


def test_analyze_forwards_action_fields(tmp_path: Path) -> None:
    """action/action_prompt form fields are passed through to the orchestrator."""
    app = create_app(_make_settings(tmp_path))
    captured: dict[str, object] = {}

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        captured.update(kwargs)
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        resp = client.post(
            "/v1/analyze",
            files=_upload("bug.mp4", b"vid"),
            data={"action": "triage"},
        )
        assert resp.status_code == 202
        assert captured["action"] == "triage"
        assert captured["action_prompt"] is None


def test_analyze_forwards_skill_and_system_prompt(tmp_path: Path) -> None:
    """skill/system_prompt form fields are passed through to the orchestrator."""
    app = create_app(_make_settings(tmp_path))
    captured: dict[str, object] = {}

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        captured.update(kwargs)
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        resp = client.post(
            "/v1/analyze",
            files=_upload("bug.mp4", b"vid"),
            data={"skill": "tutorial", "intent": "explain it"},
        )
        assert resp.status_code == 202
        _wait_terminal(client, resp.json()["job_id"])
        assert captured["skill"] == "tutorial"
        assert captured["user_intent"] == "explain it"
        assert captured["system_prompt"] is None


def test_cors_allows_only_configured_extension_origins(tmp_path: Path) -> None:
    """Only the configured extension IDs are allowed — never a wildcard.

    Every Chrome extension ID is 32 chars of [a-p], so any pattern-based
    allowlist would admit *every* installed extension. The allowlist must be
    exact.
    """
    app = create_app(_make_settings(tmp_path))
    ext_origin = f"chrome-extension://{_EXT_ID}"
    with TestClient(app) as client:
        # Preflight from the configured extension origin.
        preflight = client.options(
            "/v1/analyze",
            headers={
                "Origin": ext_origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.headers.get("access-control-allow-origin") == ext_origin

        # Actual request echoes the allowed origin too.
        health = client.get("/v1/healthz", headers={"Origin": ext_origin})
        assert health.headers.get("access-control-allow-origin") == ext_origin

    with TestClient(app) as client:
        # A different (unconfigured) extension is NOT allowed.
        other_ext = client.get(
            "/v1/healthz",
            headers={"Origin": "chrome-extension://ponmlkjihgfedcbaponmlkjihgfedcba"},
        )
        assert other_ext.headers.get("access-control-allow-origin") is None

        # A normal web origin is NOT allowed either.
        denied = client.get("/v1/healthz", headers={"Origin": "https://evil.example.com"})
        assert denied.headers.get("access-control-allow-origin") is None


def test_api_token_required_when_configured(tmp_path: Path) -> None:
    """With API_TOKEN set, /v1 endpoints 401 without the bearer token."""
    settings = Settings(
        BUNDLE_DIR=tmp_path / "bundles",
        DATABASE_PATH=tmp_path / "jobs.db",
        API_TOKEN="local-secret",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        # Health stays open so setup checks and doctors keep working.
        assert client.get("/v1/healthz").status_code == 200

        denied = client.get("/v1/jobs/nope")
        assert denied.status_code == 401
        assert denied.json()["code"] == "unauthorized"

        wrong = client.get("/v1/jobs/nope", headers={"Authorization": "Bearer wrong"})
        assert wrong.status_code == 401

        # The right token reaches the handler (404: no such job, but authorized).
        ok = client.get("/v1/jobs/nope", headers={"Authorization": "Bearer local-secret"})
        assert ok.status_code == 404


def test_upload_limit_enforced(tmp_path: Path) -> None:
    """Oversized uploads should be rejected with 413."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        # Real container magic, so the size cap is what rejects this rather than
        # the new ingest content sniff.
        large = _video(b"a" * (11 * 1024 * 1024))
        response = client.post(
            "/v1/analyze",
            files={"video": ("big.mp4", large, "video/mp4")},
        )
        assert response.status_code == 413


def test_render_html_rejects_missing_html(tmp_path: Path) -> None:
    """No/blank HTML is a 400 with a stable error code, before any render work."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        for body in ({}, {"html": ""}, {"html": "   "}):
            resp = client.post("/v1/render-html", json=body)
            assert resp.status_code == 400
            assert resp.json()["code"] == "missing_html"


def test_render_html_rejects_bad_format(tmp_path: Path) -> None:
    """An unsupported output format is rejected up front (400 bad_options)."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.post("/v1/render-html", json={"html": "<h1>hi</h1>", "format": "tiff"})
        assert resp.status_code == 400
        assert resp.json()["code"] == "bad_options"


def test_render_html_success_streams_file(tmp_path: Path, monkeypatch) -> None:
    """A successful render returns the encoded file with the right media type.

    The Chromium/ffmpeg work is stubbed so the endpoint wiring (validation,
    media-type mapping, download filename, response body) is tested without the
    optional render dependencies.
    """
    import framesleuth.pipeline.html_render as hr

    async def fake_render(html: str, options, out_dir: Path, **_limits) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"render.{options.fmt}"
        out.write_bytes(b"ENCODED-BYTES")
        return out

    monkeypatch.setattr(hr, "render_html", fake_render)

    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/render-html",
            json={"html": "<h1>hi</h1>", "format": "gif", "duration_s": 2, "fps": 12},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/gif")
        assert "animation.gif" in resp.headers.get("content-disposition", "")
        assert resp.content == b"ENCODED-BYTES"


def test_render_html_rejects_oversized_html(tmp_path: Path) -> None:
    """HTML beyond RENDER_MAX_HTML_BYTES is rejected before any browser work."""
    settings = _make_settings(tmp_path)
    settings = settings.model_copy(update={"RENDER_MAX_HTML_BYTES": 64})
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.post("/v1/render-html", json={"html": "<h1>" + "x" * 100 + "</h1>"})
        assert resp.status_code == 413
        assert resp.json()["code"] == "html_too_large"


def test_render_html_cleans_temp_dir_on_unexpected_error(tmp_path: Path, monkeypatch) -> None:
    """A non-HtmlRenderError failure must not leak the fs-render- temp dir."""
    import tempfile as _tempfile

    import framesleuth.pipeline.html_render as hr

    async def boom(html: str, options, out_dir: Path, **_limits) -> Path:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(hr, "render_html", boom)
    monkeypatch.setattr(_tempfile, "tempdir", str(tmp_path))

    app = create_app(_make_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/v1/render-html", json={"html": "<h1>hi</h1>"})
        assert resp.status_code == 500
    assert not list(tmp_path.glob("fs-render-*"))


def test_malformed_sidecars_rejected_with_400(tmp_path: Path) -> None:
    """Present-but-invalid sidecars JSON is a 400 — not a silently empty bundle."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/analyze",
            files=_upload("v.mp4", b"vid"),
            data={"sidecars": "{not json"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "bad_sidecars"
        # And no stray temp upload was left behind.
        assert not list((tmp_path / "bundles").glob("upload-*"))


def test_list_jobs_paginates(tmp_path: Path) -> None:
    """GET /v1/jobs lists jobs newest-first with a total and pagination."""
    settings = _make_settings(tmp_path)
    settings.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        _seed_job_sync(settings, f"job-{i}")
    app = create_app(settings)
    with TestClient(app) as client:
        page = client.get("/v1/jobs", params={"limit": 2, "offset": 0}).json()
        assert page["total"] == 3
        assert len(page["jobs"]) == 2
        rest = client.get("/v1/jobs", params={"limit": 2, "offset": 2}).json()
        assert len(rest["jobs"]) == 1


def test_delete_report_purges_bundle_and_row(tmp_path: Path) -> None:
    """DELETE /v1/report removes the bundle directory and the job row."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        job_id = client.post("/v1/analyze", files=_upload("v.mp4", b"vid")).json()["job_id"]
        bundle_dir = app.state.app_state.settings.BUNDLE_DIR / job_id
        assert bundle_dir.exists()

        resp = client.delete(f"/v1/report/{job_id}")
        assert resp.status_code == 200 and resp.json()["deleted"] is True
        assert not bundle_dir.exists()
        assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_delete_report_refuses_active_job(tmp_path: Path) -> None:
    """An active job must be cancelled before its report can be deleted."""
    settings = _make_settings(tmp_path)
    settings.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    with TestClient(app) as client:
        _seed_job_sync(settings, "active-1")  # created QUEUED
        resp = client.delete("/v1/report/active-1")
        assert resp.status_code == 409
        assert resp.json()["code"] == "job_active"


def test_render_html_unavailable_returns_503(tmp_path: Path, monkeypatch) -> None:
    """When Playwright/ffmpeg are missing the endpoint reports 503, not 500."""
    import framesleuth.pipeline.html_render as hr

    async def boom(html: str, options, out_dir: Path, **_limits) -> Path:
        raise hr.HtmlRenderError("Playwright is not installed.")

    monkeypatch.setattr(hr, "render_html", boom)

    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.post("/v1/render-html", json={"html": "<h1>hi</h1>", "format": "mp4"})
        assert resp.status_code == 503
        assert resp.json()["code"] == "render_unavailable"


def _seed_job_sync(settings: Settings, job_id: str, state: JobState | None = None) -> None:
    """Insert a job row directly (shared SQLite file) before the app starts."""
    import asyncio

    from framesleuth.jobs.store import JobStore

    async def _seed() -> None:
        store = JobStore(settings.DATABASE_PATH)
        await store.initialize()
        await store.create_job(job_id, f"h-{job_id}", "v.mp4")
        if state is not None:
            await store.update_job(job_id, state=state)

    asyncio.run(_seed())


def test_cancel_queued_job_sets_flag(tmp_path: Path) -> None:
    """DELETE on an active job requests cancellation."""
    settings = _make_settings(tmp_path)
    settings.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    with TestClient(app) as client:
        # Seed the job while the app is live — a pre-startup queued row would be
        # correctly failed by orphaned-job recovery.
        _seed_job_sync(settings, "q-cancel")
        resp = client.delete("/v1/jobs/q-cancel")
        assert resp.status_code == 200
        assert resp.json()["cancel_requested"] is True

        job = client.get("/v1/jobs/q-cancel").json()
        # Still queued (a real run would transition it to cancelled at a checkpoint).
        assert job["state"] == "queued"


def test_idempotency_skips_failed_jobs_and_respects_options(tmp_path: Path) -> None:
    """A failed attempt never blocks re-analysis; different options = new job."""
    app = create_app(_make_settings(tmp_path))

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(json.dumps({"id": job_id}), encoding="utf-8")
        await state.store.update_job(
            job_id, state=JobState.DONE, progress_pct=100, bundle_path=str(bundle_path)
        )
        return bundle_path

    with TestClient(app) as client:
        state = app.state.app_state
        state.orchestrator.run = fake_run

        files = _upload("v.mp4", b"same-bytes")
        first = client.post("/v1/analyze", files=files).json()
        _wait_terminal(client, first["job_id"])

        # Same bytes + same options → reused.
        again = client.post("/v1/analyze", files=files).json()
        assert again["idempotent"] is True and again["job_id"] == first["job_id"]

        # Same bytes, different skill → a NEW job (options are part of the key).
        other = client.post("/v1/analyze", files=files, data={"skill": "bug_report"}).json()
        assert other["idempotent"] is False and other["job_id"] != first["job_id"]

        assert client.get(f"/v1/jobs/{first['job_id']}").json()["state"] == "done"

    # A *failed* attempt must never block re-analysis of the same bytes. The job
    # is failed by letting the pipeline actually raise, rather than force-writing
    # the row: `update_job` now refuses to mutate a terminal job, so a
    # done → failed poke would (correctly) be a no-op and the test would prove
    # nothing.
    async def explode(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        raise RuntimeError("pipeline blew up")

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = explode
        failing = client.post("/v1/analyze", files=_upload("v.mp4", b"other-bytes")).json()
        assert _wait_terminal(client, failing["job_id"])["state"] == "failed"

        app.state.app_state.orchestrator.run = fake_run
        retry = client.post("/v1/analyze", files=_upload("v.mp4", b"other-bytes")).json()
        assert retry["idempotent"] is False
        assert retry["job_id"] != failing["job_id"]


def test_cancel_terminal_job_is_409(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    settings.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    _seed_job_sync(settings, "q-done", state=JobState.DONE)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.delete("/v1/jobs/q-done")
        assert resp.status_code == 409
        assert resp.json()["code"] == "not_cancellable"


def test_cancel_missing_job_is_404(tmp_path: Path) -> None:
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.delete("/v1/jobs/nope").status_code == 404


def test_job_events_stream_emits_terminal_state(tmp_path: Path) -> None:
    """The SSE stream yields a snapshot and closes once the job is terminal."""
    settings = _make_settings(tmp_path)
    settings.BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    _seed_job_sync(settings, "q-sse", state=JobState.DONE)
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/v1/jobs/q-sse/events")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "data:" in resp.text
        assert '"state": "done"' in resp.text


def test_webhook_fires_on_completion(tmp_path: Path, monkeypatch) -> None:
    """A completed job POSTs a compact payload to WEBHOOK_URL."""
    posts: list[tuple[str, dict]] = []

    class _FakeSession:
        def __init__(self, *a, **k) -> None:
            pass

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *a) -> bool:
            return False

        async def post(self, url: str, json: dict | None = None) -> None:
            posts.append((url, json or {}))

    import framesleuth.service.api as api_module

    monkeypatch.setattr(api_module.aiohttp, "ClientSession", _FakeSession)

    settings = _make_settings(tmp_path)
    settings.WEBHOOK_URL = "http://example.test/hook"
    app = create_app(settings)

    async def fake_run(job_id: str, video_path: Path, source_video: str, **kwargs: object) -> Path:
        state = app.state.app_state
        bundle_dir = state.settings.BUNDLE_DIR / job_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.json"
        bundle_path.write_text(
            json.dumps({"id": job_id, "title": "demo", "action": "summarize"}), encoding="utf-8"
        )
        await state.store.update_job(job_id, state=JobState.DONE, bundle_path=str(bundle_path))
        return bundle_path

    with TestClient(app) as client:
        app.state.app_state.orchestrator.run = fake_run
        resp = client.post("/v1/analyze", files=_upload("s.mp4", b"vid"))
        assert resp.status_code == 202

    assert posts, "expected a webhook POST"
    url, payload = posts[0]
    assert url == "http://example.test/hook"
    assert payload["state"] == "done"
    assert payload["title"] == "demo"


def test_healthz_includes_render_availability(tmp_path: Path) -> None:
    """/v1/healthz surfaces the optional HTML→video readiness block."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/v1/healthz")
        assert resp.status_code == 200
        render = resp.json()["render"]
        for key in ("playwright", "chromium", "ffmpeg", "python", "ready"):
            assert key in render
        assert isinstance(render["ready"], bool)


def test_healthz_reports_degraded_when_the_vision_model_is_not_installed(tmp_path: Path) -> None:
    """`vlm: ready` is the README's promise of real classification — it must be earned.

    A model server with VLM_MODEL unpulled still answers 200 on /v1/models, so
    the endpoint used to report `healthy` / `vlm: ready` for an install where
    every analysis would fall back to sidecars.
    """
    from unittest.mock import AsyncMock, patch

    settings = _make_settings(tmp_path)
    settings.VLM_MODEL = "qwen2.5vl"
    with patch("aiohttp.ClientSession") as mock_session:
        resp_mock = AsyncMock()
        resp_mock.status = 200
        resp_mock.json = AsyncMock(return_value={"data": [{"id": "llama3.2:1b"}]})
        session = mock_session.return_value.__aenter__.return_value
        session.get.return_value.__aenter__.return_value = resp_mock

        with TestClient(create_app(settings)) as client:
            body = client.get("/v1/healthz").json()

    assert body["services"]["vlm"]["status"] == "degraded"
    assert body["services"]["vlm"]["error"] == "model 'qwen2.5vl' not loaded"
    assert body["status"] == "degraded"


def test_cors_does_not_trust_the_hosted_site_by_default(tmp_path: Path) -> None:
    """A stock install trusts no remote origin — not even the vendor's own site.

    The default allowlist used to include https://framesleuth.com, so a page on
    that site could read bundles off any operator's machine without anyone ever
    opting in. In a product whose promise is that nothing leaves your machine,
    that has to be a deliberate paste, not a shipped default.
    """
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.options(
            "/v1/healthz",
            headers={
                "Origin": "https://framesleuth.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") is None


def test_cors_allows_hosted_site_when_operator_opts_in(tmp_path: Path) -> None:
    """Opted in, the hosted site can preflight the loopback API (CORS + Private Network).

    This is what lets framesleuth.com talk to a locally-running agent: the browser
    sends a Private Network Access preflight, and the API must echo the origin plus
    Access-Control-Allow-Private-Network so Chrome doesn't block the real request.
    """
    settings = _make_settings(tmp_path)
    settings.WEB_ORIGINS = "https://framesleuth.com"
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.options(
            "/v1/healthz",
            headers={
                "Origin": "https://framesleuth.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://framesleuth.com"
        assert resp.headers.get("access-control-allow-private-network") == "true"


def test_local_dev_origin_works_without_configuration(tmp_path: Path) -> None:
    """Emptying the default must not break `npm run dev` against a local agent."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.options(
            "/v1/healthz",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_unknown_web_origin(tmp_path: Path) -> None:
    """An origin not on the allowlist is never granted access."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.options(
            "/v1/healthz",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        assert resp.headers.get("access-control-allow-origin") is None


def test_the_cors_allowlist_can_be_emptied(tmp_path: Path) -> None:
    """A hardened deployment must be able to trust NO browser origin.

    The loopback entries were made unconditional, which left no configuration at
    all that produces an empty allowlist — a headless deployment serving no dev
    server still trusted two origins, and the security checklist told operators
    it "starts empty". ALLOW_LOCAL_DEV_ORIGINS restores the reachable state.
    """
    settings = _make_settings(tmp_path)
    settings.WEB_ORIGINS = ""
    settings.ALLOW_LOCAL_DEV_ORIGINS = False
    assert settings.web_origins_list == []

    # ...while the default still makes `npm run dev` work with no config.
    settings.ALLOW_LOCAL_DEV_ORIGINS = True
    assert settings.web_origins_list == ["http://localhost:3000", "http://127.0.0.1:3000"]


def test_web_origins_are_deduplicated(tmp_path: Path) -> None:
    """Re-listing a loopback origin must not put it in the allowlist twice.

    The docstring promised de-duplication and nothing tested it: removing the
    `dict.fromkeys` left the whole suite green.
    """
    settings = _make_settings(tmp_path)
    settings.WEB_ORIGINS = "http://localhost:3000, https://example.dev, https://example.dev"
    assert settings.web_origins_list == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://example.dev",
    ]


def test_web_origins_setting_is_honored(tmp_path: Path) -> None:
    """A custom WEB_ORIGINS allowlist is respected (and parsed from the CSV string)."""
    settings = _make_settings(tmp_path)
    settings.WEB_ORIGINS = "https://example.dev, http://localhost:4000"
    # Loopback dev origins are present by default; the setting adds to them.
    assert settings.web_origins_list == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://example.dev",
        "http://localhost:4000",
    ]
    app = create_app(settings)
    with TestClient(app) as client:
        ok = client.options(
            "/v1/healthz",
            headers={"Origin": "https://example.dev", "Access-Control-Request-Method": "GET"},
        )
        assert ok.headers.get("access-control-allow-origin") == "https://example.dev"
        # framesleuth.com is no longer in the (overridden) allowlist.
        no = client.options(
            "/v1/healthz",
            headers={"Origin": "https://framesleuth.com", "Access-Control-Request-Method": "GET"},
        )
        assert no.headers.get("access-control-allow-origin") is None


def test_healthz_never_leaks_the_interpreter_path(tmp_path: Path) -> None:
    """`/v1/healthz` is public, so it must never carry deployment diagnostics.

    The guard was inverted: it stripped `python`/`hint` only when NO token was
    configured, so setting API_TOKEN — asking for MORE security — was what
    exposed the absolute interpreter path to an unauthenticated caller, because
    healthz is in `_PUBLIC_PATHS` and is never gated.
    """
    for token in ("", "s3cret"):
        settings = _make_settings(tmp_path / f"t{len(token)}")
        settings.API_TOKEN = token
        with TestClient(create_app(settings)) as client:
            body = client.get("/v1/healthz").json()
            assert body["render"].get("python") is None, f"leaked with API_TOKEN={token!r}"
            assert body["render"].get("hint") is None
            # The useful readiness booleans are still there.
            assert "ready" in body["render"]


def test_version_carries_the_render_diagnostics(tmp_path: Path) -> None:
    """The verbose diagnostics live on the token-GATED endpoint instead.

    Asserts the FIELDS, not just that a `render` key exists. The weaker version
    of this test was cited as already pinning this behaviour, and it did not:
    deleting `python` and `hint` from the handler left the suite green. Those
    two are the whole reason the endpoint exists — README and runbook now send
    operators here to diagnose a failed HTML→video render, so an empty `render`
    object would be a documented dead end.

    They live on /v1/version rather than /v1/healthz precisely because the
    interpreter path is sensitive: healthz is the one public endpoint, and this
    one is behind API_TOKEN when a token is configured.
    """
    settings = _make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        body = client.get("/v1/version").json()
        assert "render" in body
        render = body["render"]
        assert render.get("python"), "the interpreter path is the point of this endpoint"
        assert render.get("hint"), "operators are sent here for an actionable hint"
        assert set(body["capabilities"]) >= {"render_html", "grounding_over_http"}

        # And the sensitive half must NOT be on the public endpoint.
        health = client.get("/v1/healthz").json()
        assert health.get("render", {}).get("python") is None


def test_error_responses_are_flat_and_match_the_published_schema(tmp_path: Path) -> None:
    """The OpenAPI error schema must describe what the server actually returns.

    The published schema referenced a wrapper class modelling FastAPI's default
    `{"detail": {...}}` envelope, which the exception handlers flatten — so the
    documented shape disagreed with every real error, and the schema-freshness
    CI check certified it as correct.
    """
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/v1/jobs/does-not-exist")
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" not in body
        assert body["code"] == "job_not_found"
        assert body["request_id"]
        assert resp.headers["x-request-id"] == body["request_id"]

    schema = app.openapi()
    ref = schema["paths"]["/v1/jobs/{job_id}"]["get"]["responses"]["404"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    documented = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    # Whatever the schema names it, its properties must be the flat body.
    assert set(documented["properties"]) >= {"error", "code", "hint", "request_id"}
    assert "detail" not in documented["properties"]
