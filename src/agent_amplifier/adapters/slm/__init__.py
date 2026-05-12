# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""SuperLocalMemory (SLM) adapter — Mode 2 of the SLM x amp complementarity model.

Per H-5 (DECISIONS-LOCKED.md), this adapter activates the **composed pipeline**
mode: when SLM is detected on the host, the amplifier kernel uses SLM's
recall API as its ``memory_recall`` source. The result: SLM data shapes
amp's classification, goal-anchor, and convergence comparison — not just
sits adjacent to it (Mode 1).

Mode 3 (closed loop) is implemented separately in
``agent_amplifier.adapters.claude_code.slm_writeback`` — the Stop hook
writes per-turn outcome summaries back to SLM so future amp turns can
recall them.

Public surface:
    * :class:`SLMAdapter` — thin shim over the ``slm`` CLI.
    * :func:`detect_slm` — boolean detection helper used by hosts to
      decide whether to wire SLMAdapter as the memory plane.
"""
from __future__ import annotations

from agent_amplifier.adapters.slm.memory import SLMAdapter, detect_slm

__all__ = [
    "SLMAdapter",
    "detect_slm",
]
