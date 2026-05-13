# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tier 2 semantic-similarity helper — local embedding via Ollama HTTP.

The Stop hook calls into this module to compute a cosine similarity
between (envelope goal text) and (Claude's final assistant message)
when the Tier 1 Jaccard score falls in the ambiguous zone. Tier 2 is
strictly optional and fails open — if Ollama is unreachable, the
embedding model is missing, or the request times out, the helper
returns ``None`` and the Stop hook falls back to Tier 1 + Tier 3 only.

Default endpoint and model match Varun's local Ollama install
(verified ``nomic-embed-text`` pulled, daemon on
``127.0.0.1:11434``). All configuration is env-var driven so users
can swap models or point at a remote Ollama without code changes:

    AGENT_AMP_OLLAMA_URL=http://127.0.0.1:11434
    AGENT_AMP_EMBED_MODEL=nomic-embed-text
    AGENT_AMP_EMBED_ENABLED=1        # set to 0 to short-circuit Tier 2
    AGENT_AMP_EMBED_TIMEOUT_S=2.0

No third-party dependencies — we use ``urllib.request`` from the
stdlib so installing AA never pulls in an HTTP client.
"""
from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.request
from typing import Final

LOG = logging.getLogger("agent_amplifier._internal.embedding")

_DEFAULT_OLLAMA_URL: Final[str] = "http://127.0.0.1:11434"
_DEFAULT_EMBED_MODEL: Final[str] = "nomic-embed-text"
_DEFAULT_TIMEOUT_S: Final[float] = 2.0
_MAX_INPUT_CHARS: Final[int] = 8000  # nomic-embed-text 8k token window


def is_tier2_enabled() -> bool:
    """Return False when ``AGENT_AMP_EMBED_ENABLED`` is explicitly ``"0"``.

    Any other value (unset, ``"1"``, ``"true"``, ``"yes"``) is treated as
    enabled — Tier 2 is on by default once the embedding model is
    available, but the env var lets CI / debugging force a Tier-1-only
    code path.
    """
    return os.environ.get("AGENT_AMP_EMBED_ENABLED", "1") != "0"


def _ollama_url() -> str:
    return os.environ.get("AGENT_AMP_OLLAMA_URL", _DEFAULT_OLLAMA_URL).rstrip("/")


def _embed_model() -> str:
    return os.environ.get("AGENT_AMP_EMBED_MODEL", _DEFAULT_EMBED_MODEL)


def _timeout_s() -> float:
    raw = os.environ.get("AGENT_AMP_EMBED_TIMEOUT_S")
    if not raw:
        return _DEFAULT_TIMEOUT_S
    try:
        return max(0.1, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT_S


def embed(text: str) -> list[float] | None:
    """Compute an embedding for ``text`` via the local Ollama API.

    Returns the vector on success, ``None`` on any failure path
    (disabled, empty input, HTTP error, timeout, malformed response).
    Never raises — every error is logged at DEBUG.

    Truncates the input at ``_MAX_INPUT_CHARS`` so a 1MB transcript
    doesn't trigger model context overflow.
    """
    if not is_tier2_enabled():
        return None
    if not text or not text.strip():
        return None
    payload = json.dumps(
        {
            "model": _embed_model(),
            "prompt": text[:_MAX_INPUT_CHARS],
        }
    ).encode("utf-8")
    url = f"{_ollama_url()}/api/embeddings"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=_timeout_s()) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        LOG.debug("Ollama embed failed: %s", exc)
        return None
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, TypeError) as exc:
        LOG.debug("Ollama embed JSON decode failed: %s", exc)
        return None
    if not isinstance(obj, dict):
        return None
    vec = obj.get("embedding")
    if not isinstance(vec, list) or not vec:
        return None
    try:
        return [float(x) for x in vec]
    except (TypeError, ValueError) as exc:
        LOG.debug("Ollama embed non-numeric value: %s", exc)
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float | None:
    """Cosine similarity between two equal-length vectors, in ``[-1, 1]``.

    Returns ``None`` on length mismatch, empty input, or zero vector
    (cosine is undefined when one operand has zero norm).
    """
    if not a or not b or len(a) != len(b):
        return None
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def similarity(text_a: str, text_b: str) -> float | None:
    """Convenience wrapper — embed both texts then cosine-compare.

    Returns ``None`` if either embedding fails or if cosine is undefined.
    Clamps the result to ``[0, 1]`` (negative cosines are mapped to 0)
    so it composes cleanly with the Tier 1 score on the ``[0, 1]`` axis.
    """
    va = embed(text_a)
    if va is None:
        return None
    vb = embed(text_b)
    if vb is None:
        return None
    cos = cosine_similarity(va, vb)
    if cos is None:
        return None
    return max(0.0, min(1.0, cos))


__all__ = [
    "cosine_similarity",
    "embed",
    "is_tier2_enabled",
    "similarity",
]
