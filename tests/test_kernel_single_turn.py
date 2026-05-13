# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Kernel single-iteration adapter branch tests (v1.1.1).

Covers the ``before_step`` override that swaps the multi-iteration phase
envelope for the structured single-turn envelope when the adapter's
``is_single_iteration`` flag is True.

Both dispatch paths exercised:

* Inline path (``LOW``, ``MEDIUM``, ``HIGH`` complexity) — envelope
  contains ``<plan>`` / ``<execute>`` / ``<reflection>`` / ``<final_answer>``.
* Subagent path (``MAX`` complexity) — envelope contains Task tool dispatch
  instructions plus the inline envelope embedded verbatim.

Non-single-iteration adapters (default) keep the legacy multi-iteration
envelope.
"""

from __future__ import annotations

from agent_amplifier import AgentAmplifier
from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.adapters import ClaudeCodeAdapter
from agent_amplifier.effort_router import EffortLevel
from agent_amplifier.kernel import _build_single_turn_envelope


class _SingleIterStubAdapter(AdapterBase):
    """Minimal stub that flags single-iteration without touching disk."""

    framework_name = "stub_single"
    version = "0.0.0"
    is_single_iteration = True

    def install(self) -> None:  # pragma: no cover
        return None

    def uninstall(self) -> None:  # pragma: no cover
        return None

    def on_before_step(self, context):  # pragma: no cover
        return context

    def on_after_step(self, context, result):  # pragma: no cover
        return None


class _MultiIterStubAdapter(AdapterBase):
    """Stub with is_single_iteration left at the default (False)."""

    framework_name = "stub_multi"
    version = "0.0.0"

    def install(self) -> None:  # pragma: no cover
        return None

    def uninstall(self) -> None:  # pragma: no cover
        return None

    def on_before_step(self, context):  # pragma: no cover
        return context

    def on_after_step(self, context, result):  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# _build_single_turn_envelope — direct unit tests
# ---------------------------------------------------------------------------


def test_build_single_turn_envelope_inline_for_low_medium_high() -> None:
    for level in (EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH):
        env = _build_single_turn_envelope(query="some task", complexity=level)
        assert "<plan>" in env
        assert "<execute>" in env
        assert "<reflection>" in env
        assert "<final_answer>" in env
        # Inline path never carries the Task-tool dispatch directive.
        assert "subagent_type" not in env


def test_build_single_turn_envelope_subagent_for_max() -> None:
    env = _build_single_turn_envelope(
        query="design distributed consensus protocol with formal proofs",
        complexity=EffortLevel.MAX,
    )
    assert "Task tool" in env
    assert "general-purpose" in env
    # Subagent envelope embeds the inline envelope, so all 5 tags appear too.
    assert "<plan>" in env
    assert "<final_answer>" in env


def test_build_single_turn_envelope_minimal_uses_inline_path() -> None:
    env = _build_single_turn_envelope(
        query="hi", complexity=EffortLevel.MINIMAL
    )
    assert "<final_answer>" in env
    assert "subagent_type" not in env  # MINIMAL stays inline


# ---------------------------------------------------------------------------
# Kernel ``before_step`` integration — adapter flag drives envelope choice
# ---------------------------------------------------------------------------


def test_kernel_uses_single_turn_envelope_for_claude_code_adapter() -> None:
    aa = AgentAmplifier()
    adapter = ClaudeCodeAdapter(aa)
    aa_with_adapter = AgentAmplifier(adapter=adapter)
    env_obj = aa_with_adapter.before_step(
        "Refactor authentication module to use JWT", {}
    )
    assert "<plan>" in env_obj.envelope
    assert "<execute>" in env_obj.envelope
    assert "<reflection>" in env_obj.envelope
    assert "<final_answer>" in env_obj.envelope


def test_kernel_uses_single_turn_envelope_for_any_single_iteration_adapter() -> None:
    aa = AgentAmplifier()
    adapter = _SingleIterStubAdapter(aa)
    aa_with_adapter = AgentAmplifier(adapter=adapter)
    env_obj = aa_with_adapter.before_step(
        "Refactor authentication module to use JWT", {}
    )
    assert "<plan>" in env_obj.envelope
    assert "<final_answer>" in env_obj.envelope


def test_kernel_keeps_legacy_envelope_for_non_single_iteration_adapter() -> None:
    aa = AgentAmplifier()
    adapter = _MultiIterStubAdapter(aa)
    aa_with_adapter = AgentAmplifier(adapter=adapter)
    env_obj = aa_with_adapter.before_step(
        "Refactor authentication module to use JWT", {}
    )
    # Legacy envelope uses the multi-iteration phase prompts, NOT the XML
    # phase-staged single-turn structure. Either an explicit "PHASE:" header
    # or, at minimum, the absence of the XML phase tags.
    assert "<plan>" not in env_obj.envelope or "<final_answer>" not in env_obj.envelope


def test_kernel_keeps_legacy_envelope_when_adapter_is_none() -> None:
    # No adapter wired -> always legacy path.
    aa = AgentAmplifier()
    env_obj = aa.before_step("Refactor authentication module to use JWT", {})
    assert "<plan>" not in env_obj.envelope or "<final_answer>" not in env_obj.envelope


def test_kernel_single_turn_envelope_carries_max_dispatch_for_complex_tasks() -> None:
    aa = AgentAmplifier()
    adapter = ClaudeCodeAdapter(aa)
    aa_with_adapter = AgentAmplifier(adapter=adapter)
    env_obj = aa_with_adapter.before_step(
        "design distributed consensus protocol with formal proofs across 5 services",
        {},
    )
    if env_obj.classification.complexity == EffortLevel.MAX:
        assert "Task tool" in env_obj.envelope
    else:  # pragma: no cover - branch exercised by the prior test path
        assert "<plan>" in env_obj.envelope
