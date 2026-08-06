# Security Policy

## Reporting a vulnerability

Report security issues privately via
[GitHub Security Advisories](https://github.com/thestackhub1/framesleuth-agent/security/advisories/new).
Please do **not** open a public issue for anything exploitable.

Include what you have: affected version or commit, what an attacker gains, and
a reproduction if you have one. A rough report beats a delayed one.

You should get an acknowledgement within 3 working days and an assessment
within 10. This is a small project — if you hear nothing, ping the advisory
thread rather than assuming it was received.

## What is in scope

Framesleuth runs on your machine, opens a local HTTP port, and hands the output
to a coding agent. The interesting boundaries are:

- **The local HTTP API.** It binds loopback and requires `API_TOKEN` when not
  loopback-only. Anything that reads or writes outside the configured workspace
  root, escapes a job directory, or reaches a token-gated endpoint without a
  token is in scope. `/v1/healthz` is intentionally public and intentionally
  free of diagnostics.
- **CORS and DNS rebinding.** Origins are allowlisted. A page that gets a
  browser to drive the local API is in scope.
- **Prompt injection.** Screen-captured text (OCR, console output, page content)
  is untrusted input that ends up near an agent that can edit code. It is fenced
  with a per-render nonce and marked untrusted. A way to break out of that fence,
  forge the close tag, or otherwise get captured text treated as instructions is
  very much in scope — see
  [`tests/test_mcp_server.py`](tests/test_mcp_server.py) for the existing cases.
- **Path traversal** in report ids, keyframe paths, or artifact names.
- **Secret leakage** into bundles, logs, or rendered prompts. The redactor is in
  [`framesleuth/pipeline/redact.py`](framesleuth/pipeline/redact.py); a secret
  format it misses is a valid report.
- **MCP tool surface.** A tool that does more than its annotations claim — a
  `readOnlyHint` tool with a side effect, for instance.

## What is out of scope

- Attacks that need the operator to deliberately misconfigure the agent, such as
  binding `0.0.0.0` on a hostile network with `API_TOKEN` unset. That is a
  documented footgun, not a vulnerability. Tell us if it is under-documented.
- Vulnerabilities in a model you point the agent at, or in your own model server.
- Anything requiring an attacker who already has local code execution as your
  user. At that point they do not need Framesleuth.
- Denial of service through very large or malformed video. Inputs are bounded,
  but the bounds are a resource guard, not a security boundary.

## Supported versions

`main` is the supported branch. Fixes land there and are released from there.
