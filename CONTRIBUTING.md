# Contributing

Thanks for looking. This is an Apache-2.0 project and patches are welcome.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra dev
```

`--frozen` installs exactly what `uv.lock` records — the versions CI tested.
Without it, `uv sync` will happily re-resolve and rewrite the lock, which is how
you end up debugging a failure nobody else can reproduce.

ffmpeg is **not** required — PyAV bundles its own. You do need a vision model
endpoint to run the full pipeline; see [README.md](README.md) for the options.

### Changing a dependency

`uv.lock` is committed and CI runs `uv lock --check`, so a `pyproject.toml` edit
that was never re-locked fails the build rather than being quietly ignored:

```bash
uv lock                       # after editing pyproject.toml
uv sync --frozen --extra dev
```

To take newer versions deliberately — which is how updates happen here, since
this repo opens no automatic version-bump PRs:

```bash
uv lock --upgrade
uv sync --frozen --extra dev
# then run the full gate below; an upgrade that breaks something fails locally
```

Commit `uv.lock` alongside the change. A dependency bump with no lock diff means
the bump did not happen.

## The checks CI runs

Run these before opening a PR — they are exactly what CI enforces, so a green
local run means a green PR:

```bash
uv run ruff check framesleuth tests && uv run black --check framesleuth tests && uv run mypy --strict framesleuth && uv run pytest tests/ -q --cov=framesleuth --cov-fail-under=75
```

Two more gates that are easy to forget:

```bash
uv run python scripts/eval_harness.py --behavioral
uv run python scripts/export_openapi.py --check --out openapi.json
```

The first runs the behavioral eval suites against committed per-metric
thresholds. The second fails if you changed the HTTP API without regenerating
`openapi.json` — that file is a cross-repo contract, and the website generates
its TypeScript types from it, so a stale schema breaks a different repository.
Regenerate with the same command minus `--check`.

`mypy --strict` is not aspirational here; the package ships `py.typed` and
downstream consumers see these types. New code needs annotations.

## What makes a change easy to merge

- **Say why in the commit message, not what.** The diff shows what changed.
  What is worth writing down is the reasoning that is not recoverable from the
  code — the alternative you rejected, the constraint that forced the shape.
- **Comment the surprising, not the obvious.** Most comments in this codebase
  exist because something bit someone. Match that bar.
- **Add a test that fails without your fix.** For pipeline changes, prefer a
  case in the eval harness over a unit test asserting on model output.
- **Keep degradation graceful.** Recordings arrive with no audio, weak models,
  and unreadable frames. A stage that cannot do its job should degrade and say
  so in `analysis_quality`, not raise.

## Things to be careful with

- **Anything touching prompts.** Screen-captured text reaches an agent that can
  edit code. The nonce fence in [`framesleuth/prompts.py`](framesleuth/prompts.py)
  is a security control — read
  [`tests/test_mcp_server.py`](tests/test_mcp_server.py) before changing it.
- **MCP tool annotations.** `readOnlyHint` / `destructiveHint` are promises made
  to a client that will act on them. Keep them true.
- **The redactor.** Widening a pattern is cheap; narrowing one can leak secrets
  into a bundle that gets pasted into an issue.

## Reporting bugs

Open an issue with the version, the command you ran, and what happened. If the
pipeline produced a bad bundle, the `provenance` block in `bundle.json` records
the model, prompt version, and commit — please include it.

Security issues go to [SECURITY.md](SECURITY.md), not the issue tracker.
