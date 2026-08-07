"""Tests for error taxonomy and API error envelopes."""

import pytest

from framesleuth.errors import (
    DurationExceededError,
    ErrorCode,
    ModelNotLoadedError,
    ModelUnavailableError,
    UnsupportedMediaError,
    UploadTooLargeError,
)


class TestErrorTaxonomy:
    """Test error types and error envelopes."""

    def test_unsupported_media_error(self) -> None:
        """Test UnsupportedMediaError properties."""
        error = UnsupportedMediaError("Video is AV1 codec")
        assert error.code == ErrorCode.UNSUPPORTED_MEDIA
        assert error.status_code == 422
        assert "H.264" in error.hint

    def test_model_unavailable_error(self) -> None:
        """Test ModelUnavailableError properties."""
        error = ModelUnavailableError("VLM", "http://127.0.0.1:8080")
        assert error.status_code == 503
        assert "127.0.0.1:8080" in error.message

    def test_model_not_loaded_error_does_not_claim_unreachable(self) -> None:
        """A missing model must not be reported as an unreachable server."""
        error = ModelNotLoadedError("VLM", "qwen2.5vl", "http://127.0.0.1:11434")
        assert error.code == ErrorCode.MODEL_NOT_LOADED
        assert error.status_code == 503
        assert "qwen2.5vl" in error.message
        assert "not reachable" not in error.message
        assert error.hint is not None
        assert "ollama pull qwen2.5vl" in error.hint

    def test_model_not_loaded_is_caught_as_model_unavailable(self) -> None:
        """Existing `except ModelUnavailableError` paths must keep catching it."""
        assert issubclass(ModelNotLoadedError, ModelUnavailableError)

    def test_upload_too_large_error(self) -> None:
        """Test UploadTooLargeError properties."""
        error = UploadTooLargeError(600.5, 512)
        assert error.code == ErrorCode.UPLOAD_TOO_LARGE
        assert error.status_code == 413
        assert "600.5" in error.message
        assert "512" in error.message

    def test_duration_exceeded_error(self) -> None:
        """Test DurationExceededError properties."""
        error = DurationExceededError(1200.0, 600)
        assert error.code == ErrorCode.DURATION_EXCEEDED
        assert error.status_code == 422

    def test_error_to_dict(self) -> None:
        """Test error serialization to dict."""
        error = UnsupportedMediaError("Test")
        error_dict = error.to_dict()
        assert "error" in error_dict
        assert "code" in error_dict
        assert "hint" in error_dict
        assert error_dict["code"] == "unsupported_media"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
