"""Classification routing for video analysis outputs."""

from __future__ import annotations

import json
from typing import Any, Protocol

from framesleuth.config import Settings, get_settings
from framesleuth.logging_config import get_logger
from framesleuth.schemas import Classification, ClassificationLabel, SceneRecord, Transcript

logger = get_logger("pipeline.classify")

_BUG_MARKERS = ("error", "exception", "failed", "traceback", "typeerror")

# Evidence that a captured failure is a real defect rather than page noise.
# Mirrors ``bug_extract._HIGH_SEVERITY_HINTS`` so severity and classification
# agree on what "strong" means.
_STRONG_FAILURE_HINTS = (
    "exception",
    "traceback",
    "stack trace",
    "uncaught",
    "crash",
    "fatal",
    " 500",
    " 502",
    " 503",
    " 504",
    "internal server error",
    "failed to fetch",
    "cannot read",
    "is not a function",
    "is not defined",
    "unhandled",
    # Named runtime error types. A thrown TypeError is every bit as much a real
    # defect as a 5xx, but matching only on the generic words above missed a
    # console line like "TypeError in save_cart at cart.py:1" entirely.
    "typeerror",
    "referenceerror",
    "rangeerror",
    "syntaxerror",
    "valueerror",
    "keyerror",
    "attributeerror",
    "nullpointer",
    "segmentation fault",
    "panic:",
)

# A deterministic keyword heuristic must never claim certainty; the headroom is
# what keeps a strong case eligible for the model tie-breaker.
_MAX_DETERMINISTIC_CONFIDENCE = 0.85

# Positive evidence for the labels that were previously unreachable.
_TUTORIAL_PHRASES = (
    "how to",
    "step by step",
    "first you",
    "then you",
    "let me show you",
    "i'll show you",
    "in this tutorial",
    "walk you through",
    "you need to",
    "start by",
)
_DEMO_PHRASES = (
    "here is",
    "here's the",
    "this is the",
    "a quick look",
    "walkthrough",
    "overview of",
    "demo of",
    "showing off",
    "take a look at",
)
_FEEDBACK_PHRASES = (
    "i think",
    "feels ",
    "looks cramped",
    "too small",
    "too big",
    "hard to read",
    "confusing",
    "would prefer",
    "not a fan",
    "my suggestion",
    "nitpick",
)

# Verbs/phrases that signal a build/feature request ("add a dark mode", "build this
# screen", "let's create a settings page"). Strong verbs weigh more than nouns.
_FEATURE_VERBS = (
    "add",
    "build",
    "create",
    "implement",
    "design",
    "redesign",
    "prototype",
    "scaffold",
    "generate",
    "make a",
    "make the",
)
_FEATURE_PHRASES = (
    "i want",
    "i'd like",
    "i would like",
    "could we",
    "can we add",
    "let's add",
    "lets add",
    "let's build",
    "lets build",
    "let's create",
    "lets create",
    "going to build",
    "going to add",
    "want to add",
    "want to build",
    "need to add",
    "new feature",
    "feature request",
    "would be nice",
    "should have",
    "build this",
    "make this",
    "turn this into",
)
# When the request is really a fix, don't mistake it for a feature.
_FIX_MARKERS = ("fix", "bug", "broken", "not working", "doesn't work", "regression", "crash")


def feature_intent_signal(user_intent: str | None, transcript_text: str = "") -> float:
    """Score 0-1 that this is a build/feature request.

    The user's explicit intent is the strongest signal; narration is secondary.
    A clearly fix-flavored intent suppresses the feature score so a "fix the save
    button" request is never misread as a feature build.
    """
    intent = (user_intent or "").lower()
    transcript = (transcript_text or "").lower()

    score = 0.0
    if any(v in intent for v in _FEATURE_VERBS) or any(p in intent for p in _FEATURE_PHRASES):
        score += 0.6
    # A narrated build *phrase* ("let's build a settings page") is a strong signal
    # on its own; a bare verb in narration is weaker (it may be incidental).
    if any(p in transcript for p in _FEATURE_PHRASES):
        score += 0.5
    elif any(v in transcript for v in _FEATURE_VERBS):
        score += 0.3
    if any(m in intent for m in _FIX_MARKERS):
        score -= 0.5  # it's a fix, not a feature
    return max(0.0, min(1.0, score))


def looks_like_build_intent(user_intent: str | None, transcript_text: str = "") -> bool:
    """Whether to run build-aware visual extraction (structured UI capture).

    Decided up front from intent/narration because it must be known *before* the
    understanding stage (classification only happens afterwards).
    """
    return feature_intent_signal(user_intent, transcript_text) >= 0.5


def _bug_score(
    scenes: list[SceneRecord], error_signals: list[str], text_blob: str, ocr_blob: str
) -> float:
    """Score 0-1 that this recording depicts a defect, weighted by strength.

    Weighted by how *strong* the evidence is, not merely that some exists. The
    previous form added flat 0.45/0.6/0.45 terms and clamped to 1.0, so a single
    captured 404 on a favicon produced ``('bug', 1.0)`` — maximum confidence,
    which put it above the threshold and therefore skipped both the bounded
    resample and the model tie-breaker. That number is then consumed downstream
    as if it were a real probability.
    """
    signal_blob = " ".join(s.lower() for s in error_signals)
    score = 0.0

    if any(scene.is_error_state for scene in scenes):
        score += 0.4

    if error_signals:
        # A 5xx or an exception is a real failure and should clear the threshold
        # on its own; a lone 4xx (a missing favicon, a CORS-blocked analytics
        # beacon) is background noise on most pages and must not.
        strong = any(hint in signal_blob for hint in _STRONG_FAILURE_HINTS)
        score += 0.7 if strong else 0.15

    # Only human-facing text counts here — narration and what was on screen. The
    # sidecar text is already scored above, and counting it twice is how a
    # single 404 (whose text contains "failed") reached maximum confidence.
    if any(marker in text_blob or marker in ocr_blob for marker in _BUG_MARKERS):
        score += 0.25

    if "how to" in text_blob:
        score -= 0.35

    # Capped below 1.0 on purpose: a deterministic keyword heuristic should never
    # claim certainty, and leaving headroom keeps the strongest cases eligible
    # for the model tie-breaker rather than short-circuiting it.
    return max(0.0, min(_MAX_DETERMINISTIC_CONFIDENCE, score))


def _intent_scores(text_blob: str, ocr_blob: str) -> dict[ClassificationLabel, float]:
    """Positive evidence for the non-bug, non-feature labels.

    Without this, ``classify_video`` had exactly three reachable outcomes —
    BUG, FEATURE, OTHER — so TUTORIAL, DEMO and FEEDBACK were first-class enum
    members with dedicated skills and actions that nothing could ever select.
    """
    blob = f"{text_blob} {ocr_blob}"
    scores: dict[ClassificationLabel, float] = {}

    def score_for(phrases: tuple[str, ...]) -> float:
        hits = sum(1 for phrase in phrases if phrase in blob)
        return min(0.8, 0.45 + 0.15 * (hits - 1)) if hits else 0.0

    for label, phrases in (
        (ClassificationLabel.TUTORIAL, _TUTORIAL_PHRASES),
        (ClassificationLabel.DEMO, _DEMO_PHRASES),
        (ClassificationLabel.FEEDBACK, _FEEDBACK_PHRASES),
    ):
        value = score_for(phrases)
        if value:
            scores[label] = value
    return scores


def classify_video(
    transcript: Transcript,
    scenes: list[SceneRecord],
    *,
    settings: Settings | None = None,
    error_signals: list[str] | None = None,
    user_intent: str | None = None,
) -> Classification:
    """Classify session as bug/feature/tutorial/demo/feedback/other (deterministic).

    Args:
        transcript: Narration transcript (may be empty).
        scenes: Visual scene records (empty when the VLM is unavailable).
        settings: Configuration overrides.
        error_signals: Extra error texts from sidecars (console/network) that
            count as strong bug evidence even without any visual analysis.
        user_intent: The user's natural-language request — the strongest signal for
            a build/feature task ("add a dark-mode toggle", "build this screen").
    """
    settings = settings or get_settings()

    text_blob = " ".join(segment.text.lower() for segment in transcript.segments)
    ocr_blob = " ".join(scene.ocr_text.lower() for scene in scenes)

    # Every label is scored on its own evidence, then they compete. Scoring one
    # label and deriving the rest from ``1 - confidence`` is what produced
    # ``('other', 0.0, [('tutorial', 1.0)])`` — an alternative asserted far more
    # confidently than the chosen label.
    candidates: dict[ClassificationLabel, float] = _intent_scores(text_blob, ocr_blob)
    candidates[ClassificationLabel.BUG] = _bug_score(
        scenes, error_signals or [], text_blob, ocr_blob
    )
    feature_score = feature_intent_signal(user_intent, text_blob)
    if feature_score:
        candidates[ClassificationLabel.FEATURE] = feature_score

    ranked = sorted(
        ((label, score) for label, score in candidates.items() if score > 0),
        key=lambda pair: (-pair[1], pair[0].value),
    )

    # A confident bug wins outright: an observed failure outranks a stylistic
    # read of the narration, however chatty that narration is.
    bug_score = candidates[ClassificationLabel.BUG]
    if bug_score >= settings.CLASSIFY_CONFIDENCE_THRESHOLD:
        label, confidence = ClassificationLabel.BUG, bug_score
    elif ranked:
        label, confidence = ranked[0]
    else:
        label, confidence = ClassificationLabel.OTHER, 0.0

    # Alternatives are the *other scored* labels, so each carries its own
    # evidence rather than an invented complement of the winner's.
    alt_labels = [
        (alt_label, round(score, 2)) for alt_label, score in ranked if alt_label is not label
    ][:2]

    return Classification(label=label, confidence=round(confidence, 2), alt_labels=alt_labels)


class SupportsComplete(Protocol):
    """Minimal text-completion interface for model-based refinement."""

    async def complete(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_tokens: int = ...,
        temperature: float = ...,
    ) -> str:
        """Run a system+user chat turn and return the text."""


def is_ambiguous(classification: Classification, settings: Settings) -> bool:
    """Whether the deterministic result sits in the uncertain band.

    Below ``CLASSIFY_AMBIGUOUS_FLOOR`` there is no signal worth a model call;
    at/above ``CLASSIFY_CONFIDENCE_THRESHOLD`` the heuristic is already confident.
    """
    return (
        settings.CLASSIFY_AMBIGUOUS_FLOOR
        <= classification.confidence
        < settings.CLASSIFY_CONFIDENCE_THRESHOLD
    )


def _classification_signals(
    scenes: list[SceneRecord],
    error_signals: list[str] | None,
    user_intent: str | None = None,
    transcript_text: str = "",
) -> dict[str, Any]:
    """Build the diagnostic signal block for the model classifier prompt.

    ``transcript_text`` matters: without it this reported ``feature_intent:
    False`` for a narrated build request ("let's build a settings page") in the
    very prompt asking the model to break that tie — actively misleading it,
    while the deterministic scorer had used both signals all along.
    """
    errors = error_signals or []
    blob = " ".join(s.lower() for s in errors)
    return {
        "error_count": len(errors),
        "has_stack_trace": "traceback" in blob or "stack" in blob,
        "error_frames": sum(1 for scene in scenes if scene.is_error_state),
        "feature_intent": feature_intent_signal(user_intent, transcript_text) >= 0.5,
    }


def _parse_model_classification(raw: str) -> Classification | None:
    """Parse a model JSON classification, tolerating markdown fences.

    Returns ``None`` when the text cannot be mapped to a valid label so the
    caller can keep the deterministic result instead of guessing.
    """
    content = raw.strip()
    if "```" in content:
        # Pull the body of the first fenced block (``` or ```json).
        fenced = content.split("```", 2)
        if len(fenced) >= 2:
            content = fenced[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
    try:
        data = json.loads(content)
        label = ClassificationLabel(str(data["label"]).lower())
        confidence = max(0.0, min(1.0, float(data["confidence"])))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("Model classification unparseable, keeping heuristic: %s", exc)
        return None

    alt_labels: list[tuple[ClassificationLabel, float]] = []
    for item in data.get("alt_labels", []) or []:
        try:
            alt_labels.append(
                (ClassificationLabel(str(item[0]).lower()), max(0.0, min(1.0, float(item[1]))))
            )
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return Classification(label=label, confidence=confidence, alt_labels=alt_labels)


async def refine_classification_with_model(
    base: Classification,
    *,
    summary_text: str,
    scenes: list[SceneRecord],
    error_signals: list[str] | None,
    client: SupportsComplete,
    settings: Settings | None = None,
    user_intent: str | None = None,
    transcript_text: str = "",
) -> Classification:
    """Break an ambiguous-band tie with a model classification.

    Falls back to ``base`` whenever the classification is already confident, the
    band gate is off, there is nothing to summarize, or the model is unavailable
    or returns something unparseable — so this never weakens the deterministic
    result, it only sharpens genuinely uncertain ones.
    """
    settings = settings or get_settings()
    if not settings.CLASSIFY_USE_MODEL or not is_ambiguous(base, settings):
        return base
    if not summary_text.strip():
        return base

    from framesleuth.prompts import ClassificationPrompts

    prompt = ClassificationPrompts.classify_video(
        summary_text,
        _classification_signals(scenes, error_signals, user_intent, transcript_text),
    )
    try:
        raw = await client.complete(
            "You are a precise classifier. Return ONLY the requested JSON.",
            prompt,
            max_tokens=256,
            temperature=0.0,
        )
    except Exception as exc:  # model/network failure must not weaken the result
        logger.warning("Model classification degraded, keeping heuristic: %s", exc)
        return base

    refined = _parse_model_classification(raw)
    if refined is None:
        return base
    logger.info(
        "Classification refined by model: %s (%.2f) <- heuristic %s (%.2f)",
        refined.label.value,
        refined.confidence,
        base.label.value,
        base.confidence,
    )
    return refined
