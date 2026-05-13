# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for F1D — Tier 2 embedding blend inside the Stop hook.

These tests exercise ``_maybe_blend_tier2`` directly so the branches in
``stop_hook.py`` are covered without depending on a live Ollama daemon.
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_amplifier._internal import embedding as _embedding
from agent_amplifier.adapters.claude_code import stop_hook as _sh


def _no_blend_envelope() -> dict[str, Any]:
    return {
        "user_prompt_redacted": "fix the auth bug",
        "envelope_text": "anchor goal here",
    }


# ---------------------------------------------------------------------------
# Short-circuit branches
# ---------------------------------------------------------------------------


def test_tier2_skipped_below_ambig_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier1 = 0.20 is decisive (low) — no embedding call."""
    monkeypatch.setattr(
        _sh._embedding,
        "similarity",
        lambda *a, **k: pytest.fail("should not be called"),
    )
    result = _sh._maybe_blend_tier2(
        0.20,
        envelope=_no_blend_envelope(),
        session_id="sid",
        project_cwd="/cwd",
    )
    assert result == 0.20


def test_tier2_skipped_above_ambig_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier1 = 0.80 is decisive (high) — no embedding call."""
    monkeypatch.setattr(
        _sh._embedding,
        "similarity",
        lambda *a, **k: pytest.fail("should not be called"),
    )
    result = _sh._maybe_blend_tier2(
        0.80,
        envelope=_no_blend_envelope(),
        session_id="sid",
        project_cwd="/cwd",
    )
    assert result == 0.80


def test_tier2_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_AMP_EMBED_ENABLED", "0")
    monkeypatch.setattr(
        _sh._embedding,
        "similarity",
        lambda *a, **k: pytest.fail("should not be called when disabled"),
    )
    result = _sh._maybe_blend_tier2(
        0.50,
        envelope=_no_blend_envelope(),
        session_id="sid",
        project_cwd="/cwd",
    )
    assert result == 0.50


def test_tier2_skipped_when_no_goal_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_AMP_EMBED_ENABLED", raising=False)
    monkeypatch.setattr(
        _sh._embedding,
        "similarity",
        lambda *a, **k: pytest.fail("should not reach similarity"),
    )
    result = _sh._maybe_blend_tier2(
        0.50,
        envelope={"user_prompt_redacted": None, "envelope_text": None},
        session_id="sid",
        project_cwd="/cwd",
    )
    assert result == 0.50


def test_tier2_skipped_when_transcript_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv("AGENT_AMP_EMBED_ENABLED", raising=False)
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "nothing"))
    monkeypatch.setattr(
        _sh._embedding,
        "similarity",
        lambda *a, **k: pytest.fail("should not call similarity when no transcript"),
    )
    result = _sh._maybe_blend_tier2(
        0.50,
        envelope=_no_blend_envelope(),
        session_id="absent",
        project_cwd="/cwd",
    )
    assert result == 0.50


# ---------------------------------------------------------------------------
# Blend branches
# ---------------------------------------------------------------------------


def _seed_assistant_transcript(
    tmp_path: Any, session_id: str, project_cwd: str, text: str
) -> None:
    from pathlib import Path

    from agent_amplifier.adapters.claude_code.transcript import (
        encoded_project_dir,
    )

    base = tmp_path / "transcripts"
    target = base / encoded_project_dir(Path(project_cwd)) / f"{session_id}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    import json

    target.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": text}]},
            }
        ),
        encoding="utf-8",
    )


def test_tier2_falls_through_when_similarity_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv("AGENT_AMP_EMBED_ENABLED", raising=False)
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    _seed_assistant_transcript(tmp_path, "sid", "/cwd", "agent output")
    monkeypatch.setattr(_sh._embedding, "similarity", lambda a, b: None)
    result = _sh._maybe_blend_tier2(
        0.50,
        envelope=_no_blend_envelope(),
        session_id="sid",
        project_cwd="/cwd",
    )
    assert result == 0.50


def test_tier2_blends_when_similarity_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Blend formula: 0.3*tier1 + 0.7*cos. Tier1=0.50, cos=1.0 → 0.85."""
    monkeypatch.delenv("AGENT_AMP_EMBED_ENABLED", raising=False)
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    _seed_assistant_transcript(tmp_path, "sid", "/cwd", "agent output")
    monkeypatch.setattr(_sh._embedding, "similarity", lambda a, b: 1.0)
    result = _sh._maybe_blend_tier2(
        0.50,
        envelope=_no_blend_envelope(),
        session_id="sid",
        project_cwd="/cwd",
    )
    assert result == pytest.approx(0.85)


def test_tier2_blend_clamps_to_unit_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Even with cosine=1.0 the blend must stay in [0,1]."""
    monkeypatch.delenv("AGENT_AMP_EMBED_ENABLED", raising=False)
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    _seed_assistant_transcript(tmp_path, "sid", "/cwd", "agent output")
    monkeypatch.setattr(_sh._embedding, "similarity", lambda a, b: 1.0)
    result = _sh._maybe_blend_tier2(
        0.70,
        envelope=_no_blend_envelope(),
        session_id="sid",
        project_cwd="/cwd",
    )
    assert 0.0 <= result <= 1.0


def test_tier2_blend_uses_only_envelope_text_when_prompt_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv("AGENT_AMP_EMBED_ENABLED", raising=False)
    monkeypatch.setenv("AGENT_AMP_TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    _seed_assistant_transcript(tmp_path, "sid", "/cwd", "x")
    captured: list[str] = []
    monkeypatch.setattr(
        _sh._embedding,
        "similarity",
        lambda a, b: captured.append(a) or 0.6,
    )
    result = _sh._maybe_blend_tier2(
        0.50,
        envelope={"user_prompt_redacted": None, "envelope_text": "just env"},
        session_id="sid",
        project_cwd="/cwd",
    )
    assert captured == ["just env"]
    assert result == pytest.approx(0.30 * 0.50 + 0.70 * 0.6)


# Ensure module-level vars are reachable for any monkeypatch.delenv coverage.
def test_embedding_module_constants_consistent() -> None:
    assert _embedding._MAX_INPUT_CHARS > 0
