"""Tests for configuration loading and validation."""

import pytest

from framesleuth.config import Settings


class TestConfig:
    """Test configuration management."""

    def test_settings_default(self) -> None:
        """Test default settings load."""
        # Create a temporary Settings instance without reading .env
        settings = Settings(_env_file=None)
        assert settings.BACKEND_HOST == "127.0.0.1"
        assert settings.BACKEND_PORT == 8010
        # Defaults match the documented quick-start stack (native Ollama).
        assert settings.VLM_URL == "http://127.0.0.1:11434"
        assert settings.VLM_MODEL == "qwen2.5vl"

    def test_settings_validation_paths(self) -> None:
        """Test path creation on validation."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                BUNDLE_DIR=Path(tmpdir) / "bundles",
                DATABASE_PATH=Path(tmpdir) / "db" / "jobs.db",
                _env_file=None,
            )
            settings.validate_paths()
            assert settings.BUNDLE_DIR.exists()
            assert settings.DATABASE_PATH.parent.exists()

    def test_settings_ignores_unknown_vars(self) -> None:
        """A stale .env key from an older release must never block boot."""
        settings = Settings(
            UNKNOWN_VAR="test",  # type: ignore[call-arg]
            _env_file=None,
        )
        assert not hasattr(settings, "UNKNOWN_VAR")

    def test_chrome_extension_origins(self) -> None:
        """Extension IDs become exact chrome-extension:// origins; blank = none."""
        settings = Settings(_env_file=None)
        assert settings.chrome_extension_origins == []

        ids = "abcdefghijklmnopabcdefghijklmnop, ponmlkjihgfedcbaponmlkjihgfedcba"
        configured = Settings(CHROME_EXTENSION_IDS=ids, _env_file=None)
        assert configured.chrome_extension_origins == [
            "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
            "chrome-extension://ponmlkjihgfedcbaponmlkjihgfedcba",
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
