# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier._internal.embedding`` (F1D Tier 2).

The embedding helper is the only Tier 2 surface — a fail-open wrapper
around Ollama's HTTP API. All HTTP is mocked so tests run offline.
"""
from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from agent_amplifier._internal import embedding as _e

# ---------------------------------------------------------------------------
# Helpers — fake urlopen so we never hit real network
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def _fake_ok(body: dict) -> Any:
    payload = json.dumps(body).encode("utf-8")

    def _impl(req: Any, timeout: float = 0.0) -> _FakeResp:
        return _FakeResp(payload)

    return _impl


def _fake_url_error(exc: BaseException) -> Any:
    def _impl(req: Any, timeout: float = 0.0) -> None:
        raise exc

    return _impl


# ---------------------------------------------------------------------------
# is_tier2_enabled / config defaults
# ---------------------------------------------------------------------------


def test_is_tier2_enabled_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_AMP_EMBED_ENABLED", raising=False)
    assert _e.is_tier2_enabled() is True


def test_is_tier2_enabled_off_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_AMP_EMBED_ENABLED", "0")
    assert _e.is_tier2_enabled() is False


def test_is_tier2_enabled_treats_other_values_as_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in ("1", "true", "yes", "anything"):
        monkeypatch.setenv("AGENT_AMP_EMBED_ENABLED", value)
        assert _e.is_tier2_enabled() is True


def test_url_and_model_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_AMP_OLLAMA_URL", raising=False)
    monkeypatch.delenv("AGENT_AMP_EMBED_MODEL", raising=False)
    assert _e._ollama_url() == "http://127.0.0.1:11434"
    assert _e._embed_model() == "nomic-embed-text"


def test_url_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_AMP_OLLAMA_URL", "http://x:11434/")
    assert _e._ollama_url() == "http://x:11434"


def test_timeout_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_AMP_EMBED_TIMEOUT_S", raising=False)
    assert _e._timeout_s() == 2.0


def test_timeout_parses_valid_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_AMP_EMBED_TIMEOUT_S", "0.5")
    assert _e._timeout_s() == 0.5


def test_timeout_floors_at_0p1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_AMP_EMBED_TIMEOUT_S", "0.01")
    assert _e._timeout_s() == 0.1


def test_timeout_falls_back_on_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_AMP_EMBED_TIMEOUT_S", "not-a-number")
    assert _e._timeout_s() == 2.0


# ---------------------------------------------------------------------------
# embed() — fail-open contract
# ---------------------------------------------------------------------------


def test_embed_returns_none_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_AMP_EMBED_ENABLED", "0")
    assert _e.embed("hello") is None


def test_embed_returns_none_on_empty_text() -> None:
    assert _e.embed("") is None
    assert _e.embed("   \n  ") is None


def test_embed_returns_vector_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_AMP_EMBED_ENABLED", raising=False)
    monkeypatch.setattr(
        _e.urllib.request,
        "urlopen",
        _fake_ok({"embedding": [0.1, 0.2, 0.3]}),
    )
    assert _e.embed("hello") == [0.1, 0.2, 0.3]


def test_embed_truncates_long_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm we cap at MAX_INPUT_CHARS before POSTing."""
    seen: list[bytes] = []

    def _capture(req: Any, timeout: float = 0.0) -> _FakeResp:
        seen.append(req.data)
        return _FakeResp(json.dumps({"embedding": [1.0]}).encode("utf-8"))

    monkeypatch.setattr(_e.urllib.request, "urlopen", _capture)
    big = "x" * 20_000
    _e.embed(big)
    sent = json.loads(seen[0].decode("utf-8"))
    assert len(sent["prompt"]) == _e._MAX_INPUT_CHARS


def test_embed_returns_none_on_url_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _e.urllib.request,
        "urlopen",
        _fake_url_error(urllib.error.URLError("connection refused")),
    )
    assert _e.embed("hi") is None


def test_embed_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _e.urllib.request,
        "urlopen",
        _fake_url_error(TimeoutError()),
    )
    assert _e.embed("hi") is None


def test_embed_returns_none_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _e.urllib.request, "urlopen", _fake_url_error(OSError("eof"))
    )
    assert _e.embed("hi") is None


def test_embed_returns_none_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bad(req: Any, timeout: float = 0.0) -> _FakeResp:
        return _FakeResp(b"<<<not json>>>")

    monkeypatch.setattr(_e.urllib.request, "urlopen", _bad)
    assert _e.embed("hi") is None


def test_embed_returns_none_on_non_dict_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _e.urllib.request,
        "urlopen",
        _fake_ok(["array", "not", "dict"]),  # type: ignore[arg-type]
    )
    assert _e.embed("hi") is None


def test_embed_returns_none_when_embedding_field_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _e.urllib.request, "urlopen", _fake_ok({"other": "field"})
    )
    assert _e.embed("hi") is None


def test_embed_returns_none_when_embedding_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _e.urllib.request, "urlopen", _fake_ok({"embedding": []})
    )
    assert _e.embed("hi") is None


def test_embed_returns_none_on_non_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _e.urllib.request,
        "urlopen",
        _fake_ok({"embedding": [0.1, "not-a-float", 0.3]}),
    )
    assert _e.embed("hi") is None


# ---------------------------------------------------------------------------
# cosine_similarity — pure math
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors_returns_one() -> None:
    v = [1.0, 0.0, 0.0]
    assert _e.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_returns_zero() -> None:
    assert _e.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_returns_negative_one() -> None:
    assert _e.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_returns_none_on_empty() -> None:
    assert _e.cosine_similarity([], []) is None
    assert _e.cosine_similarity([], [1.0]) is None


def test_cosine_returns_none_on_length_mismatch() -> None:
    assert _e.cosine_similarity([1.0, 2.0], [1.0]) is None


def test_cosine_returns_none_on_zero_vector() -> None:
    assert _e.cosine_similarity([0.0, 0.0], [1.0, 1.0]) is None
    assert _e.cosine_similarity([1.0, 1.0], [0.0, 0.0]) is None


# ---------------------------------------------------------------------------
# similarity — wrapper that embeds both sides
# ---------------------------------------------------------------------------


def test_similarity_identical_text_returns_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _e.urllib.request, "urlopen", _fake_ok({"embedding": [1.0, 0.0, 0.0]})
    )
    assert _e.similarity("alpha", "alpha") == pytest.approx(1.0)


def test_similarity_returns_none_when_first_embed_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_AMP_EMBED_ENABLED", "0")
    assert _e.similarity("a", "b") is None


def test_similarity_returns_none_when_second_embed_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call succeeds, second fails — exercise the second-leg None path."""
    calls = {"n": 0}

    def _alternating(req: Any, timeout: float = 0.0) -> _FakeResp:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResp(json.dumps({"embedding": [1.0]}).encode("utf-8"))
        raise urllib.error.URLError("down")

    monkeypatch.setattr(_e.urllib.request, "urlopen", _alternating)
    assert _e.similarity("first", "second") is None


def test_similarity_returns_none_on_zero_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _e.urllib.request, "urlopen", _fake_ok({"embedding": [0.0, 0.0]})
    )
    assert _e.similarity("a", "b") is None


def test_similarity_clamps_negative_cosine_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two anti-parallel embeddings give cosine -1; the wrapper clamps to 0
    so the result composes on the [0,1] axis with Tier 1.
    """
    calls = {"n": 0}

    def _alternating(req: Any, timeout: float = 0.0) -> _FakeResp:
        calls["n"] += 1
        vec = [1.0, 0.0] if calls["n"] == 1 else [-1.0, 0.0]
        return _FakeResp(json.dumps({"embedding": vec}).encode("utf-8"))

    monkeypatch.setattr(_e.urllib.request, "urlopen", _alternating)
    assert _e.similarity("a", "b") == 0.0
