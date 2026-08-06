"""ASR pipeline with graceful no-audio behavior."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from framesleuth.logging_config import get_logger
from framesleuth.schemas import Transcript

logger = get_logger("pipeline.asr")


@lru_cache(maxsize=2)
def _load_model(model_name: str, compute_type: str) -> Any:
    """Load (and cache) a Whisper model for the process lifetime.

    The model was constructed *inside* ``transcribe()``, so every job paid the
    full load — and with MAX_CONCURRENT_JOBS > 1 two jobs each held their own
    copy of the weights. Caching by (name, compute_type) means the second job
    reuses the first's, which is both faster and materially less memory.
    """
    from faster_whisper import WhisperModel

    logger.info("Loading Whisper model %s (%s)", model_name, compute_type)
    return WhisperModel(model_name, compute_type=compute_type)


class ASRPipeline:
    """Typed ASR service wrapper around faster-whisper."""

    def __init__(
        self,
        model_name: str = "small",
        compute_type: str = "int8",
        *,
        min_confidence: float = 0.0,
        vad_filter: bool = False,
        language: str | None = None,
    ) -> None:
        """Initialize the ASR pipeline with a whisper model and compute type.

        ``min_confidence`` drops segments whose confidence (``1 - no_speech_prob``)
        falls below the threshold, filtering out Whisper's silence hallucinations.
        Defaults to ``0.0`` (keep everything) so direct callers are unaffected.
        ``vad_filter`` enables faster-whisper's built-in Silero voice-activity
        filter so silence is never decoded. ``language`` forces a transcription
        language (ISO code) instead of auto-detecting.
        """
        self.model_name = model_name
        self.compute_type = compute_type
        self.min_confidence = min_confidence
        self.vad_filter = vad_filter
        self.language = language or None

    def transcribe(
        self,
        audio_path: Path | None,
        *,
        has_audio: bool,
        model_override: Any | None = None,
    ) -> Transcript:
        """Transcribe audio into typed transcript segments.

        If there is no audio stream, return an empty transcript by design.
        """
        if not has_audio or audio_path is None:
            return Transcript(segments=[], words=[])

        model = model_override
        if model is None:
            try:
                model = _load_model(self.model_name, self.compute_type)
            except Exception as exc:
                logger.warning("faster-whisper unavailable, returning empty transcript: %s", exc)
                return Transcript(segments=[], words=[])

        raw_segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=self.vad_filter,
            language=self.language,
            # Screencast narration is stop-start, and conditioning on the
            # previous text is what makes Whisper spiral into repeating a phrase
            # across a long silence. The thresholds are the documented guards
            # against that same failure mode.
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )
        segments: list[Transcript.Segment] = []
        words: list[dict[str, Any]] = []

        dropped = 0
        for segment in raw_segments:
            confidence = 1.0 - float(getattr(segment, "no_speech_prob", 0.0))
            conf = max(0.0, min(1.0, confidence))
            # Skip low-confidence segments (and their words): these are typically
            # Whisper hallucinating filler over silence and only pollute the timeline.
            if conf < self.min_confidence:
                dropped += 1
                continue

            segments.append(
                Transcript.Segment(
                    t0=float(segment.start),
                    t1=float(segment.end),
                    text=str(segment.text).strip(),
                    conf=conf,
                )
            )

            segment_words = getattr(segment, "words", None) or []
            for word in segment_words:
                words.append(
                    {
                        "word": getattr(word, "word", ""),
                        "start": float(getattr(word, "start", 0.0)),
                        "end": float(getattr(word, "end", 0.0)),
                        "probability": float(getattr(word, "probability", 0.0)),
                    }
                )

        detected_language = self.language or getattr(info, "language", None)
        logger.info(
            "ASR complete: %s segments (%s dropped below conf %.2f), vad=%s, language=%s",
            len(segments),
            dropped,
            self.min_confidence,
            self.vad_filter,
            detected_language or "unknown",
        )
        return Transcript(segments=segments, words=words, language=detected_language)
