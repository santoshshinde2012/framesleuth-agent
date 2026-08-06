"""Bundle retention: delete persisted jobs and their artifacts past a TTL.

Without this, every analyzed video accumulates a bundle directory (frames, the
source recording, sidecars) forever. The sweep runs at startup and removes jobs
older than ``ttl_days`` — both the SQLite row and the on-disk bundle — so disk use
stays bounded. A TTL of ``0`` disables it (keep everything).
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from framesleuth.jobs.store import JobStore
from framesleuth.logging_config import get_logger

logger = get_logger("jobs.retention")


def _parse_iso(value: str) -> datetime | None:
    """Parse a stored ISO timestamp, returning ``None`` if it is unparseable."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def purge_expired_bundles(
    store: JobStore, bundle_dir: Path, ttl_days: int, *, now: datetime | None = None
) -> int:
    """Delete jobs (and their bundle directories) older than ``ttl_days``.

    Args:
        store: The job store to read/delete rows from.
        bundle_dir: Root directory holding ``{job_id}/`` bundles.
        ttl_days: Age threshold in days; ``<= 0`` disables the sweep.
        now: Override for the current time (testing); defaults to ``datetime.now``.

    Returns:
        The number of jobs purged.
    """
    if ttl_days <= 0:
        return 0
    current = now or datetime.now(UTC)
    cutoff = current - timedelta(days=ttl_days)

    purged = 0
    for job in await store.list_jobs():
        created = _parse_iso(job.created_at)
        if created is None or created >= cutoff:
            continue
        # Remove the on-disk bundle first; a stale row with no bundle is harmless,
        # but an orphaned bundle with no row is invisible to cleanup.
        shutil.rmtree(bundle_dir / job.id, ignore_errors=True)
        await store.delete_job(job.id)
        purged += 1

    # MCP render_html_video writes throwaway clips under renders/<hex>/; they have
    # no job row, so the loop above never touches them. Purge those older than the
    # TTL too, or they accumulate forever.
    renders_root = bundle_dir / "renders"
    if renders_root.is_dir():
        for render_dir in renders_root.iterdir():
            if not render_dir.is_dir():
                continue
            try:
                mtime = datetime.fromtimestamp(render_dir.stat().st_mtime, tz=UTC)
            except OSError:
                continue
            if mtime < cutoff:
                shutil.rmtree(render_dir, ignore_errors=True)

    if purged:
        logger.info("Retention sweep purged %d job(s) older than %d day(s)", purged, ttl_days)
    return purged
