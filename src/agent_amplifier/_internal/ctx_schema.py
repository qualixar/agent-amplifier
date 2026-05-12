# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
""".15 — adapter-context boundary validator.

Bad shape → ``ValueError`` caught by the outer try/except in
``_AmplifierCore.before_step``, which then returns a degenerate envelope
(graceful degradation per locked decision E-7).

Public API:
    validate_context(ctx) -> dict
"""

from __future__ import annotations

from typing import Any

# Keys recognized by the kernel. Unknown keys pass through (forward compat).
_ALLOWED_KEYS: set[str] = {
    "available_tools",
    "prev_output",
    "chosen",
    "issues",
    "amp_tokens_used",
    "amp_recalled_patterns",
}

# Keys whose value-type is enforced. Subset of ``_ALLOWED_KEYS``.
_TYPED: dict[str, type] = {
    "available_tools": list,
    "prev_output": str,
    "chosen": str,
    "issues": str,
    "amp_tokens_used": int,
}


def validate_context(ctx: dict[str, Any] | None) -> dict[str, Any]:
    """Validate adapter-supplied context dict.

    Returns a NEW dict (defensive copy) with validated values. Unknown keys
    are PASSED THROUGH (forward-compat) — adapter authors may carry
    extra fields without breaking the kernel.

    Raises:
        ValueError: ctx is neither None nor a dict, OR a known key has the
            wrong type.

    Note: ``isinstance(x, int)`` accepts ``bool`` (since ``bool`` is a
    subclass of ``int``). The kernel never relies on this; if a caller
    passes ``amp_tokens_used=True`` it is accepted as ``1``. This matches
    .15's ``_TYPED`` mapping.
    """
    if ctx is None:
        return {}
    if not isinstance(ctx, dict):
        raise ValueError(
            f"context must be dict or None, got {type(ctx).__name__}"
        )
    out = dict(ctx)
    for k, t in _TYPED.items():
        if k in out and not isinstance(out[k], t):
            raise ValueError(
                f"context[{k!r}] must be {t.__name__}, "
                f"got {type(out[k]).__name__}"
            )
    return out


__all__ = ["validate_context"]
