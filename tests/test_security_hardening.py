"""Regressions for the untrusted-input and silent-evidence-loss fixes.

Each test here pins a specific defect that shipped and was not caught by the
existing suite. They are grouped by the property they protect rather than by
module, because that property is what must not regress.
"""

from __future__ import annotations

from pathlib import Path

from framesleuth.config import Settings
from framesleuth.orchestrator.graph import AnalysisOrchestrator
from framesleuth.pipeline.classify import classify_video
from framesleuth.pipeline.grounding import locate_in_code
from framesleuth.pipeline.preprocess import ExtractedFrame
from framesleuth.pipeline.redact import redact_obj, redact_text
from framesleuth.pipeline.sidecars import parse_sidecars
from framesleuth.schemas import ClassificationLabel, SceneRecord, Transcript

# ===== Redaction reaches every string, not just top-level ones =====


def test_nested_sidecar_secrets_are_redacted() -> None:
    """A network event nests its secrets; a top-level-only scrub missed them all."""
    event = {
        "t": 1.0,
        "url": "https://api.example.com/v1/pay",
        "request": {
            "headers": {
                "authorization": "Bearer abcdef1234567890ABCDEF",
                "cookie": "session_id=supersecretvalue123",
            },
            "body": {"card": "4111 1111 1111 1111"},
        },
        "tags": ["retry", "password=hunter2"],
    }
    scrubbed, redactions = redact_obj(event, timestamp=1.0)

    assert "abcdef1234567890ABCDEF" not in str(scrubbed)
    assert "supersecretvalue123" not in str(scrubbed)
    assert "4111" not in str(scrubbed)
    assert "hunter2" not in str(scrubbed)
    assert redactions
    # Non-string values survive untouched — redaction must not mangle the shape.
    assert scrubbed["t"] == 1.0
    assert isinstance(scrubbed["tags"], list)


def test_redact_obj_is_bounded_on_hostile_nesting() -> None:
    """Deeply nested input cannot turn redaction into an unbounded walk."""
    payload: dict[str, object] = {"leaf": "password=deep"}
    for _ in range(50):
        payload = {"next": payload}
    scrubbed, _ = redact_obj(payload)
    assert scrubbed is not None  # returned rather than recursing to exhaustion


def test_orchestrator_redacts_nested_sidecar_streams() -> None:
    """The persisted sidecar artifacts must never carry a nested secret."""
    settings = Settings(BUNDLE_DIR=Path("/tmp/fs-test"), DATABASE_PATH=Path("/tmp/fs-test/j.db"))
    orchestrator = AnalysisOrchestrator(settings, store=None, vlm_client=None)  # type: ignore[arg-type]
    parsed = parse_sidecars(
        [
            {
                "source": "network",
                "t": 2.0,
                "url": "/api/pay",
                "status": 500,
                "request": {"headers": {"authorization": "Bearer zzzzzzzzzzzzzzzzzz"}},
            }
        ]
    )
    redactions = orchestrator._redact_sidecar_streams(parsed)
    assert "zzzzzzzzzzzzzzzzzz" not in str(parsed.network)
    assert redactions


# ===== The caller's own text is untrusted too =====


def test_user_intent_is_redacted_before_use() -> None:
    """`user_intent` reaches a model and the bundle, so it must be scrubbed."""
    settings = Settings(BUNDLE_DIR=Path("/tmp/fs-test"), DATABASE_PATH=Path("/tmp/fs-test/j.db"))
    orchestrator = AnalysisOrchestrator(settings, store=None, vlm_client=None)  # type: ignore[arg-type]
    scrubbed, redactions = orchestrator._redact_input(
        "fix the 401, my key is sk-ant-api03-abcdefghijklmnop"
    )
    assert scrubbed is not None
    assert "sk-ant-api03" not in scrubbed
    assert redactions


def test_private_key_material_is_removed_not_just_its_header() -> None:
    """Stripping only the BEGIN line left the key body while claiming success."""
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAxyz123\n-----END RSA PRIVATE KEY-----"
    out, redactions = redact_text(pem)
    assert "MIIEowIBAAKCAQEA" not in out
    assert any(r.region == "private_key_block" for r in redactions)


def test_modern_provider_tokens_are_redacted() -> None:
    """The old `(sk|rk|pk)_` rule missed every non-Stripe provider format."""
    for secret in (
        "ghp_16C7e42F292c6912E7710c838347Ae178B4a",
        "sk-ant-api03-abcdefghijklmnopqrst",
        "glpat-ABCdefGHIjklMNOpqrST",
        "xoxb-2334-4567-abcdefghij",
        "AIzaSyD-1234567890abcdefghijklmnop",
    ):
        out, _ = redact_text(f"token is {secret}")
        assert secret not in out, f"{secret} survived redaction"


def test_ordinary_text_is_not_over_redacted() -> None:
    """Widening the detectors must not start scrubbing normal recordings."""
    for benign in (
        "clicked the Save button",
        "GET /api/users returned 500",
        "the session expired after 30 minutes",
        "def toggle_theme():",
    ):
        out, redactions = redact_text(benign)
        assert out == benign and not redactions


# ===== Evidence must not be silently lost =====


def _frames(n: int) -> list[ExtractedFrame]:
    # Rising change_score, so the "most salient" frames are the late ones — which
    # is exactly what the old early-break selector could never reach.
    return [
        ExtractedFrame(t=float(i), file=f"frames/{i}.png", change_score=i / n) for i in range(n)
    ]


def test_keyframe_selection_covers_the_end_of_the_recording() -> None:
    """The tail bins were never visited, so late frames could not be selected.

    A bug surfaces *after* the user acts, so systematically under-sampling the
    end of a recording loses precisely the evidence that matters.
    """
    settings = Settings(BUNDLE_DIR=Path("/tmp/fs-test"), DATABASE_PATH=Path("/tmp/fs-test/j.db"))
    orchestrator = AnalysisOrchestrator(settings, store=None, vlm_client=None)  # type: ignore[arg-type]

    for n, cap in ((16, 8), (28, 12), (40, 8)):
        chosen = orchestrator._coverage_keyframes(_frames(n), max_keyframes=cap)
        indices = sorted(kf.index for kf in chosen)
        assert len(indices) <= cap
        assert indices[0] == 0 and indices[-1] == n - 1, "endpoints must always be kept"
        # Something from the final quarter other than the closing frame itself.
        tail = [i for i in indices if i >= n * 0.75 and i != n - 1]
        assert tail, f"n={n} cap={cap}: nothing sampled in the final quarter ({indices})"


def test_grounding_works_under_a_directory_named_build(tmp_path: Path) -> None:
    """An ancestor named build/dist/target/vendor silently disabled grounding.

    The skip-list was applied to the ABSOLUTE path, so every ancestor of the
    workspace root was tested too and a repo at ``~/build/myapp`` returned zero
    candidates — which then flipped `build_context.is_greenfield` to True and
    told the agent a brownfield feature was net-new.
    """
    repo = tmp_path / "build" / "myapp"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "cart.py").write_text("def save_cart(item):\n    return item\n")

    candidates = locate_in_code(repo, ["save_cart"])
    assert candidates, "grounding returned nothing for a repo under a 'build' ancestor"
    assert any("cart.py" in c.file for c in candidates)


def test_grounding_still_skips_vendored_dirs_inside_the_repo(tmp_path: Path) -> None:
    """Relative-path filtering must not weaken the real vendored-dir exclusion."""
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "node_modules" / "dep").mkdir(parents=True)
    (repo / "app" / "theme.py").write_text("def toggle_theme():\n    ...\n")
    (repo / "node_modules" / "dep" / "index.js").write_text("export function toggle_theme() {}\n")

    files = [c.file.replace("\\", "/") for c in locate_in_code(repo, ["toggle_theme"])]
    assert any("app/theme.py" in f for f in files)
    assert not any("node_modules" in f for f in files)


# ===== Classification must be reachable and calibrated =====


def _t(text: str) -> Transcript:
    from framesleuth.schemas import Transcript as T

    seg = T.Segment(t0=0.0, t1=1.0, text=text, conf=0.9)
    return T(segments=[seg] if text else [], words=None)


def _sc(ocr: str = "", error: bool = False) -> SceneRecord:
    return SceneRecord(
        t0=0.0,
        t1=0.0,
        caption="",
        ocr_text=ocr,
        ui_action=None,
        is_error_state=error,
        reason=None,
        ui_elements=[],
        layout=None,
        screen_name=None,
        design_notes=None,
        data_shown=None,
    )


def test_every_classification_label_is_reachable() -> None:
    """Only BUG/FEATURE/OTHER used to be selectable by the deterministic scorer."""
    cases = [
        (
            "let me show you how to export a report step by step",
            "",
            [],
            None,
            ClassificationLabel.TUTORIAL,
        ),
        ("here is a quick look at the new dashboard", "", [], None, ClassificationLabel.DEMO),
        (
            "i think the spacing looks cramped and is hard to read",
            "",
            [],
            None,
            ClassificationLabel.FEEDBACK,
        ),
        ("", "", ["POST /save 500 Internal Server Error"], None, ClassificationLabel.BUG),
        (
            "here is the screen",
            "Settings",
            [],
            "add a dark mode toggle",
            ClassificationLabel.FEATURE,
        ),
    ]
    for text, ocr, signals, intent, expected in cases:
        result = classify_video(
            _t(text), [_sc(ocr)], error_signals=list(signals), user_intent=intent
        )
        assert result.label is expected, f"{text or ocr!r} -> {result.label.value}"


def test_an_alternative_never_outranks_the_chosen_label() -> None:
    """`alt_labels` used to be `1 - confidence`, so a 0.0 label carried a 1.0 alt."""
    for text in ("how to set up the project", "here is the dashboard", "i think it looks cramped"):
        result = classify_video(_t(text), [_sc("")])
        for _, score in result.alt_labels:
            assert score <= result.confidence


def test_a_lone_4xx_does_not_pin_bug_at_maximum_confidence() -> None:
    """A favicon 404 scored ('bug', 1.0), which also skipped every refinement path.

    Above CLASSIFY_CONFIDENCE_THRESHOLD the result is treated as certain, so
    neither the bounded resample nor the model tie-breaker ever ran on it.
    """
    weak = classify_video(
        _t(""), [_sc("Dashboard")], error_signals=["GET /favicon.ico failed with 404"]
    )
    assert weak.confidence < 0.3

    # ...while a genuine server failure still clears the threshold on its own.
    strong = classify_video(
        _t(""), [_sc("Checkout")], error_signals=["POST /pay 500 Internal Server Error"]
    )
    assert strong.label is ClassificationLabel.BUG
    assert strong.confidence >= 0.7


def test_deterministic_confidence_leaves_headroom_for_the_model() -> None:
    """A keyword heuristic must never claim certainty."""
    saturated = classify_video(
        _t("error exception failed traceback"),
        [_sc("Uncaught TypeError: cannot read x", error=True)],
        error_signals=["500 Internal Server Error", "uncaught exception"],
    )
    assert saturated.confidence <= 0.85


def test_model_tiebreaker_prompt_sees_narrated_build_intent() -> None:
    """The signal block dropped the transcript, so it reported feature_intent=False.

    That block goes into the very prompt asking the model to break the tie, so a
    narrated build request actively misinformed the tie-breaker.
    """
    from framesleuth.pipeline.classify import _classification_signals

    signals = _classification_signals(
        [_sc("Settings")], [], None, "let's build a new settings page"
    )
    assert signals["feature_intent"] is True
