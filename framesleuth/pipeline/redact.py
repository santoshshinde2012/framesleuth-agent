"""Redaction pipeline for OCR text and sidecar-like sensitive fields.

Two tiers run before any text reaches a model or is persisted: always-on
*secret* detectors (API keys, bearer tokens, JWTs, password/token assignments)
and opt-in *PII* detectors (emails, payment-card numbers, US SSNs/phones, cloud
access keys, private-key blocks). Card numbers are validated with the Luhn
checksum so an arbitrary 16-digit string (an order id, a build number) is not
redacted — only a genuine card is.
"""

from __future__ import annotations

import re
from typing import Any

from framesleuth.schemas import Redaction

# Bounds for the recursive walk over untrusted sidecar payloads: deep enough for
# any realistic browser event, shallow enough that a hostile payload cannot turn
# redaction into a stack overflow or an unbounded traversal.
_MAX_DEPTH = 8
_MAX_NODES = 5000

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # A PEM block must be matched WHOLE. Anchoring on the BEGIN line alone
    # stripped the header and left the base64 key material in place — while still
    # recording ``Redaction(applied=True)``, so the bundle asserted the secret had
    # been scrubbed when the secret was in fact right there. The second
    # alternative catches a block whose END line was cut off (a truncated OCR
    # read, a clipped console line) by consuming the base64 run that follows.
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?"
            r"-----END (?:[A-Z ]+ )?PRIVATE KEY-----"
            r"|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?(?=\n\s*\n|\Z)"
        ),
    ),
    # Provider-issued tokens, matched by their published prefixes. The previous
    # single `(sk|rk|pk)_` rule required an underscore, so it caught Stripe and
    # missed OpenAI/Anthropic `sk-`, GitHub `ghp_`, GitLab `glpat-`, Slack `xox*`,
    # and Google `AIza` — the formats most likely to be on a developer's screen.
    (
        "provider_token",
        re.compile(
            r"\b(?:"
            r"gh[pousr]_[A-Za-z0-9]{16,}"  # GitHub PAT / OAuth / server / refresh
            r"|github_pat_[A-Za-z0-9_]{20,}"  # GitHub fine-grained PAT
            r"|glpat-[A-Za-z0-9_\-]{16,}"  # GitLab PAT
            r"|xox[abprs]-[A-Za-z0-9\-]{10,}"  # Slack
            r"|AIza[A-Za-z0-9_\-]{16,}"  # Google API key
            r"|sk-(?:ant-)?[A-Za-z0-9_\-]{16,}"  # OpenAI / Anthropic
            r"|npm_[A-Za-z0-9]{16,}"  # npm
            r"|dop_v1_[A-Za-z0-9]{32,}"  # DigitalOcean
            r"|shpat_[A-Za-z0-9]{16,}"  # Shopify
            r"|(?:sk|rk|pk)_(?:live|test)?_?[A-Za-z0-9]{8,}"  # Stripe-style
            r")\b"
        ),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    # Any HTTP auth scheme, not just Bearer — Basic carries base64 user:password.
    (
        "auth_header",
        re.compile(
            r"\b(?:Authorization|Proxy-Authorization)\s*:\s*\S+\s+\S+"
            r"|\b(?:Bearer|Basic|Token|Digest)\s+[A-Za-z0-9._\-+/=]{12,}",
            re.IGNORECASE,
        ),
    ),
    (
        "cookie",
        re.compile(
            r"\b(?:Set-)?Cookie\s*:\s*.+"
            # No mandatory leading character class here: requiring one meant a
            # name that *starts* with the keyword ("session_id=...") consumed its
            # own first letter and could never match.
            r"|\b[\w.\-]*(?:session|sessid|sid|csrf|xsrf|auth)[\w.\-]*\s*=\s*[^;\s]{8,}",
            re.IGNORECASE,
        ),
    ),
    (
        "password_assignment",
        re.compile(r"\b(password|passwd|pwd|passphrase)\s*[:=]\s*\S+", re.IGNORECASE),
    ),
    # Generalized so `client_secret`, `refresh_token`, `AWS_SECRET_ACCESS_KEY`,
    # `private_key` etc. are covered rather than only the four literal names.
    (
        "token_assignment",
        re.compile(
            r"\b[\w.\-]*(?:secret|token|api[_-]?key|apikey|private[_-]?key|credential"
            r"|auth[_-]?key|access[_-]?key)[\w.\-]*\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
    ),
]

# PII detectors applied only when ``redact_pii`` is set. Kept separate from the
# secret tier so they can be disabled independently.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("us_ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Require separators so a bare 10-digit identifier is not mistaken for a phone.
    ("us_phone", re.compile(r"\b(?:\+?1[ .\-])?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}\b")),
]

# A run of 13-19 digits, optionally grouped with spaces/dashes — a *candidate*
# card number that is only redacted once it passes the Luhn checksum.
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    """Return whether ``digits`` satisfies the Luhn (mod-10) checksum."""
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _redact_cards(text: str, timestamp: float) -> tuple[str, list[Redaction]]:
    """Redact only Luhn-valid card numbers, leaving ordinary long numbers intact."""
    redactions: list[Redaction] = []

    def _replace(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            redactions.append(Redaction(t=timestamp, region="credit_card", applied=True))
            return "[REDACTED]"
        return match.group()

    return _CARD_CANDIDATE.sub(_replace, text), redactions


def redact_obj(
    value: Any,
    timestamp: float = 0.0,
    *,
    redact_pii: bool = True,
    _depth: int = 0,
    _budget: list[int] | None = None,
) -> tuple[Any, list[Redaction]]:
    """Recursively scrub every string reachable inside a nested structure.

    Browser sidecars are not flat: a network event carries
    ``request.headers.authorization``, ``request.cookies``, and a response body,
    all nested one or more levels down. Scrubbing only the top-level string
    values left every one of those in the ``sidecars.json`` / ``timeline.json``
    written into the bundle directory the API serves — while the bundle recorded
    that redaction had run.

    Bounded by depth and total node count: the input is untrusted and must not be
    able to turn redaction into a stack overflow or an unbounded walk.
    """
    budget = _budget if _budget is not None else [_MAX_NODES]
    redactions: list[Redaction] = []
    if _depth > _MAX_DEPTH or budget[0] <= 0:
        return value, redactions
    budget[0] -= 1

    if isinstance(value, str):
        return redact_text(value, timestamp, redact_pii=redact_pii)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            scrubbed, applied = redact_obj(
                item, timestamp, redact_pii=redact_pii, _depth=_depth + 1, _budget=budget
            )
            out[key] = scrubbed
            redactions.extend(applied)
        return out, redactions
    if isinstance(value, list | tuple):
        items = []
        for item in value:
            scrubbed, applied = redact_obj(
                item, timestamp, redact_pii=redact_pii, _depth=_depth + 1, _budget=budget
            )
            items.append(scrubbed)
            redactions.extend(applied)
        return (type(value)(items) if isinstance(value, tuple) else items), redactions
    return value, redactions


def redact_text(
    text: str, timestamp: float = 0.0, *, redact_pii: bool = True
) -> tuple[str, list[Redaction]]:
    """Redact likely secrets (and optionally PII) from OCR/text streams.

    Args:
        text: The text to scrub.
        timestamp: Timestamp recorded on each :class:`Redaction`.
        redact_pii: Also apply the PII detectors (emails, cards, SSNs, phones,
            cloud keys). Defaults to on; pass ``False`` to redact secrets only.
    """
    redactions: list[Redaction] = []
    output = text

    for region, pattern in _SECRET_PATTERNS:
        if pattern.search(output):
            output = pattern.sub("[REDACTED]", output)
            redactions.append(Redaction(t=timestamp, region=region, applied=True))

    if redact_pii:
        # Cards first (checksum-validated), then the simple regex detectors.
        output, card_redactions = _redact_cards(output, timestamp)
        redactions.extend(card_redactions)
        for region, pattern in _PII_PATTERNS:
            if pattern.search(output):
                output = pattern.sub("[REDACTED]", output)
                redactions.append(Redaction(t=timestamp, region=region, applied=True))

    return output, redactions
