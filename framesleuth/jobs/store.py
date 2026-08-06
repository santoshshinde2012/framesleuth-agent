"""SQLite-backed job store with idempotency key support."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from framesleuth.schemas import JobState


@dataclass
class StoredJob:
    """In-memory view of persisted job."""

    id: str
    state: JobState
    progress_pct: int
    content_hash: str
    source_video: str
    bundle_path: str | None
    error_json: dict[str, str] | None
    created_at: str
    updated_at: str
    cancel_requested: bool = False


class JobStore:
    """Transactional job store with state updates and idempotency lookups."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the store backed by the SQLite database at ``db_path``."""
        self.db_path = db_path

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Open a connection with a busy timeout for concurrent access.

        Two analysis jobs writing progress, SSE readers polling every 0.5 s, and
        cancel writes all contend on this file. The busy timeout makes a brief
        lock wait instead of raising ``database is locked`` and failing a whole
        job. (WAL journalling is enabled once in :meth:`initialize` and persists
        at the database-file level, so every connection inherits it.)
        """
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA busy_timeout=5000")
            yield conn

    async def initialize(self) -> None:
        """Initialize schema if needed (and migrate older databases forward)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as conn:
            # WAL lets readers and one writer proceed together; it is a
            # persistent per-database setting, so setting it once is enough.
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    progress_pct INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    source_video TEXT NOT NULL,
                    bundle_path TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_content_hash ON jobs(content_hash)"
            )
            # The listing (`ORDER BY created_at DESC`), the retention scan, and the
            # queue-depth count on every /healthz hit were all full table scans.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")
            # Forward-migrate databases created before the cancel_requested column.
            async with conn.execute("PRAGMA table_info(jobs)") as cursor:
                columns = {row[1] for row in await cursor.fetchall()}
            if "cancel_requested" not in columns:
                await conn.execute(
                    "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
                )
            await conn.commit()

    async def create_job(self, job_id: str, content_hash: str, source_video: str) -> None:
        """Create a queued job record."""
        async with self._connect() as conn:
            await self._insert(conn, job_id, content_hash, source_video)
            await conn.commit()

    async def claim_job(
        self, job_id: str, content_hash: str, source_video: str
    ) -> tuple[StoredJob, bool]:
        """Reuse a live/finished job with the same content hash, or create one.

        Returns ``(job, created)``. The lookup and the insert run inside a single
        ``BEGIN IMMEDIATE`` transaction on one connection: previously they were
        two separate connections with nothing between them, so two identical
        uploads arriving together both missed the lookup and both inserted — each
        then running the full VLM pipeline. The documented idempotency guarantee
        failed exactly under the concurrency it exists to handle.

        A ``failed``/``cancelled`` predecessor is never reused, so a bad run can
        never block re-analysis of the same bytes.
        """
        async with self._connect() as conn:
            conn.row_factory = aiosqlite.Row
            # IMMEDIATE takes the write lock up front, so a concurrent claim of
            # the same hash blocks here rather than racing us to the insert.
            await conn.execute("BEGIN IMMEDIATE")
            try:
                placeholders = ",".join("?" for _ in _REUSABLE_STATES)
                async with conn.execute(
                    f"SELECT * FROM jobs WHERE content_hash = ? AND state IN ({placeholders}) "
                    "ORDER BY created_at DESC LIMIT 1",
                    (content_hash, *(s.value for s in _REUSABLE_STATES)),
                ) as cursor:
                    row = await cursor.fetchone()
                existing = _row_to_job(row)
                if existing is not None:
                    await conn.commit()
                    return existing, False
                await self._insert(conn, job_id, content_hash, source_video)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        created = await self.get_job(job_id)
        if created is None:  # pragma: no cover - the row was just committed
            raise RuntimeError(f"job {job_id} vanished immediately after creation")
        return created, True

    async def _insert(
        self, conn: aiosqlite.Connection, job_id: str, content_hash: str, source_video: str
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await conn.execute(
            """
            INSERT INTO jobs(
                id, state, progress_pct, content_hash, source_video,
                bundle_path, error_json, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, JobState.QUEUED.value, 0, content_hash, source_video, None, None, now, now),
        )

    async def find_by_content_hash(self, content_hash: str) -> StoredJob | None:
        """Find existing job by content hash for idempotency."""
        async with self._connect() as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(
                "SELECT * FROM jobs WHERE content_hash = ? ORDER BY created_at DESC LIMIT 1",
                (content_hash,),
            ) as cursor:
                row = await cursor.fetchone()
        return _row_to_job(row)

    async def get_job(self, job_id: str) -> StoredJob | None:
        """Fetch job by id."""
        async with self._connect() as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cursor:
                row = await cursor.fetchone()
        return _row_to_job(row)

    async def update_job(
        self,
        job_id: str,
        *,
        state: JobState | None = None,
        progress_pct: int | None = None,
        bundle_path: str | None = None,
        error_json: dict[str, str] | None = None,
    ) -> bool:
        """Update selected mutable fields for a job that is not yet terminal.

        The terminal guard is the point: without it, a late write could resurrect
        a finished job. The real case was the watchdog — ``asyncio.timeout``
        firing while the orchestrator was inside its final "mark DONE" await
        overwrote a *successful* job with ``job_timeout``, leaving a job reported
        FAILED whose ``bundle.json`` sits on disk perfectly intact.

        Returns whether a row was actually updated, so a caller can tell "applied"
        from "the job had already finished".
        """
        return await self._update(job_id, state, progress_pct, bundle_path, error_json, guard=True)

    async def finalize(
        self,
        job_id: str,
        *,
        state: JobState,
        error_json: dict[str, str] | None = None,
    ) -> bool:
        """Record a terminal state, but never overwrite one already recorded.

        Used by the job runner's failure paths. Same guard as :meth:`update_job`
        — expressed separately so the intent ("this run is over") is explicit at
        the call site.
        """
        return await self._update(job_id, state, None, None, error_json, guard=True)

    async def _update(
        self,
        job_id: str,
        state: JobState | None,
        progress_pct: int | None,
        bundle_path: str | None,
        error_json: dict[str, str] | None,
        *,
        guard: bool,
    ) -> bool:
        updates: list[str] = []
        values: list[Any] = []

        if state is not None:
            updates.append("state = ?")
            values.append(state.value)
        if progress_pct is not None:
            updates.append("progress_pct = ?")
            values.append(progress_pct)
        if bundle_path is not None:
            updates.append("bundle_path = ?")
            values.append(bundle_path)
        if error_json is not None:
            updates.append("error_json = ?")
            values.append(json.dumps(error_json))

        updates.append("updated_at = ?")
        values.append(datetime.now(UTC).isoformat())
        values.append(job_id)

        where = "id = ?"
        if guard:
            placeholders = ",".join("?" for _ in _TERMINAL_STATES)
            where += f" AND state NOT IN ({placeholders})"
            values.extend(s.value for s in _TERMINAL_STATES)

        async with self._connect() as conn:
            cursor = await conn.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE {where}",
                tuple(values),
            )
            await conn.commit()
            return bool(cursor.rowcount and cursor.rowcount > 0)

    async def request_cancel(self, job_id: str) -> bool:
        """Flag a job for cooperative cancellation; return whether a row was updated."""
        now = datetime.now(UTC).isoformat()
        async with self._connect() as conn:
            cursor = await conn.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                (now, job_id),
            )
            await conn.commit()
            return bool(cursor.rowcount and cursor.rowcount > 0)

    async def is_cancel_requested(self, job_id: str) -> bool:
        """Whether a cancellation has been requested for ``job_id``."""
        async with self._connect() as conn, conn.execute(
            "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return bool(row[0]) if row else False

    async def fail_orphaned_jobs(self) -> int:
        """Mark every non-terminal job FAILED (crash/restart recovery).

        Called once at startup: any job still queued/running in the database
        necessarily belonged to a previous process, so it can never finish.
        Without this, zombie rows inflate the queue depth forever and — worse —
        the content-hash idempotency would keep returning a job that will never
        complete. Returns the number of jobs failed.
        """
        active = ",".join(f"'{s.value}'" for s in _ACTIVE_STATES)
        now = datetime.now(UTC).isoformat()
        error = json.dumps(
            {"error": "Server restarted while the job was in progress", "code": "orphaned"}
        )
        async with self._connect() as conn:
            cursor = await conn.execute(
                f"UPDATE jobs SET state = ?, error_json = ?, updated_at = ? "
                f"WHERE state IN ({active})",
                (JobState.FAILED.value, error, now),
            )
            await conn.commit()
            return int(cursor.rowcount or 0)

    async def count_active(self) -> int:
        """Number of jobs currently queued or running (the live queue depth)."""
        active = ",".join(f"'{s.value}'" for s in _ACTIVE_STATES)
        async with self._connect() as conn, conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE state IN ({active})"
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def list_jobs(self) -> list[StoredJob]:
        """Return every persisted job (used by retention cleanup)."""
        async with self._connect() as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT * FROM jobs ORDER BY created_at") as cursor:
                rows = await cursor.fetchall()
        return [job for job in (_row_to_job(row) for row in rows) if job is not None]

    async def list_jobs_page(
        self, *, limit: int, offset: int, states: Sequence[JobState] | None = None
    ) -> tuple[list[StoredJob], int]:
        """Return a page of jobs (newest first) plus the matching total.

        Powers the ``GET /v1/jobs`` listing so a client can page through history
        instead of only fetching one job it already knows the id of. ``states``
        filters in SQL rather than making the caller page the whole history to
        find, say, the failed jobs. ``total`` reflects the same filter, so
        pagination stays consistent.
        """
        where = ""
        params: tuple[Any, ...] = ()
        if states:
            placeholders = ",".join("?" for _ in states)
            where = f" WHERE state IN ({placeholders})"
            params = tuple(s.value for s in states)
        async with self._connect() as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(f"SELECT COUNT(*) FROM jobs{where}", params) as cursor:
                row = await cursor.fetchone()
                total = int(row[0]) if row else 0
            async with conn.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, max(1, limit), max(0, offset)),
            ) as cursor:
                rows = await cursor.fetchall()
        jobs = [job for job in (_row_to_job(row) for row in rows) if job is not None]
        return jobs, total

    async def delete_job(self, job_id: str) -> None:
        """Delete a job row (its bundle directory is removed separately)."""
        async with self._connect() as conn:
            await conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            await conn.commit()


# Job states that count toward the live queue depth (not yet terminal).
_ACTIVE_STATES = (
    JobState.QUEUED,
    JobState.PREPROCESSING,
    JobState.UNDERSTANDING,
    JobState.CLASSIFYING,
    JobState.EXTRACTING,
    JobState.GROUNDING,
)

# States a job can never leave. Writes are refused once a job reaches one of
# these, so a finished job's outcome is immutable.
_TERMINAL_STATES = (JobState.DONE, JobState.FAILED, JobState.CANCELLED)

# States whose job may be returned for an identical upload. A failed or cancelled
# attempt must NOT be reused — otherwise one bad run would permanently block
# re-analysis of those exact bytes.
_REUSABLE_STATES = (*_ACTIVE_STATES, JobState.DONE)


def _row_to_job(row: Any) -> StoredJob | None:
    if row is None:
        return None
    keys = row.keys() if hasattr(row, "keys") else []
    return StoredJob(
        id=row["id"],
        state=JobState(row["state"]),
        progress_pct=int(row["progress_pct"]),
        content_hash=row["content_hash"],
        source_video=row["source_video"],
        bundle_path=row["bundle_path"],
        error_json=json.loads(row["error_json"]) if row["error_json"] else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        cancel_requested=bool(row["cancel_requested"]) if "cancel_requested" in keys else False,
    )
