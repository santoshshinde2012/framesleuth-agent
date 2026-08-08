# Multi-stage: the C toolchain needed to build any source-only wheel stays in the
# builder and never ships in the runtime image.
FROM python:3.11-slim AS builder

WORKDIR /app

# build-essential is only needed to compile wheels that lack a binary build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv, for `--frozen` installs. Pinned by digest-bearing tag rather than :latest,
# because "reproducible builds" that fetch an unpinned installer are not.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

# Dependency metadata first, so editing source does NOT invalidate the (slow)
# dependency layer. Previously `COPY . .` preceded the install, so every code
# change re-resolved every dependency and re-downloaded ~150 MB of Chromium.
#
# uv.lock is part of that metadata now. This stage used to `pip install "."`,
# which re-resolved every dependency at build time — so two builds of the SAME
# commit could ship different versions, and the image was the one artifact where
# that mattered most, being the thing users actually run.
COPY pyproject.toml uv.lock README.md ./
COPY framesleuth/__init__.py framesleuth/__init__.py

# Render support (HTML→video via Playwright + headless Chromium) is ON by default
# so the feature works out of the box; build with --build-arg INSTALL_RENDER=false
# for a slimmer image without it.
ARG INSTALL_RENDER=true
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN if [ "$INSTALL_RENDER" = "true" ]; then \
        uv sync --frozen --no-dev --no-install-project --extra render; \
    else \
        uv sync --frozen --no-dev --no-install-project; \
    fi

# Now the real source. Installed non-editable and without deps (already resolved
# above) — an editable install in an image leaves a .pth pointing at a build path.
COPY . .
RUN uv pip install --python /opt/venv/bin/python --no-cache-dir --no-deps .


FROM python:3.11-slim AS runtime

# Runtime system dependencies only:
#   ffmpeg            — encodes the HTML→video output (mp4/gif) and helps media decode
#   libgl1/libglib2.0 — required by opencv-python at import time (cv2 needs libGL)
#   libnss3 …         — the shared libraries headless Chromium links against
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Run as a non-root user. This is not only hygiene: html_render.py disables the
# Chromium sandbox when it detects euid 0 (`no_sandbox = os.geteuid() == 0`), so
# running as root silently turned the sandbox OFF while rendering caller-supplied
# HTML. A non-root uid keeps the user-namespace sandbox on.
RUN useradd --create-home --uid 10001 framesleuth \
    && mkdir -p /app/bug-reports \
    && chown -R framesleuth:framesleuth /app
USER framesleuth

# Chromium is fetched as the same user that will later launch it, so it lands in
# a cache that user can actually read.
ARG INSTALL_RENDER=true
RUN if [ "$INSTALL_RENDER" = "true" ]; then playwright install chromium; fi

# The API listens on BACKEND_HOST:BACKEND_PORT inside the container and binds
# 0.0.0.0 to be reachable from the host. docker-compose publishes it on the
# host's LOOPBACK only, so "reachable from the host" never means "reachable from
# the LAN". The healthcheck and entrypoint both read these, so they cannot drift
# from the actual bind.
ENV BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000

# Liveness: the API returns 200 even in degraded/sidecar-only mode.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/v1/healthz' % os.environ.get('BACKEND_PORT','8000'))"

EXPOSE 8000
# Bind to BACKEND_PORT via the framesleuth-api entrypoint (reads Settings), so
# the container honors the same env var as the direct/dev path.
CMD ["framesleuth-api"]
