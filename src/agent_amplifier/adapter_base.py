# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Abstract base for Agent Amplifier adapters (IP-6), V2.0 —  §2.


     (DistSys F-01) — async sibling methods (aon_before_step/aon_after_step)
                            with default impls bridging to sync via
                            ``anyio.to_thread.run_sync``
     (Sec F-08) — ``framework_name`` regex validation in __init__
     (PM F-08)  — ``detect()`` classmethod for ``agent-amp install --auto``

Design rationale:
    * **ABC over Protocol** (locked E-1). We OWN every adapter; loud failure
      on missing methods is the right tradeoff.
    * **Lifecycle**: ``install()`` ↔ ``uninstall()`` are inverses. Tested per
      adapter.
    * **Hot-path methods** are pure functions of ``(context, result)``. The
      base class does no I/O. Subclasses MAY do I/O but MUST document it.
    * **Async siblings** (``aon_*``) default to a sync-bridge impl so adapter
      authors can opt out. Async-host adapters (LangGraph, async-LangChain
      callbacks) override ``aon_*`` directly.
    * **STAGE-5C-003 hard warning**: adapters MUST NOT route raw tool output
      / SLM-recalled text into prompt slots without ``_neutralize_xml``;
      see  V2.0 §1.5.1 + the docstring on ``on_before_step``.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # pragma: no cover
    from agent_amplifier.kernel import AgentAmplifier, AsyncAgentAmplifier
    from agent_amplifier.types import Outcome, RecalledPattern

LOG = logging.getLogger("agent_amplifier.adapter")

#  . Enforced in ``__init__`` so a bad
# class fails to construct BEFORE any I/O.
_NAME_RE: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]{2,31}$")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class AdapterError(Exception):
    """Base for all adapter-side failures."""


class AdapterNotInstalledError(AdapterError):
    """Raised when ``uninstall()`` is called on a non-installed adapter."""


class AdapterAlreadyInstalledError(AdapterError):
    """Raised when ``install()`` is called twice without ``uninstall()`` in between."""


# ---------------------------------------------------------------------------
# AdapterBase
# ---------------------------------------------------------------------------


class AdapterBase(ABC):
    """Contract every framework adapter MUST implement.

    Subclass invariants (V2.0):

    1. ``framework_name`` class attribute matches ``^[a-z][a-z0-9_]{2,31}$``.
    2. ``install()`` is idempotent OR raises
       :class:`AdapterAlreadyInstalledError`.
    3. ``uninstall()`` removes ONLY this adapter's hooks, never others'.
    4. ``on_before_step`` / ``on_after_step`` are pure on the kernel state —
       they READ the kernel and TRANSFORM context, but do NOT mutate kernel
       internals (the kernel exposes its own state mutators).
    5. ``aon_*`` default impls call the sync version via
       ``anyio.to_thread.run_sync``. Override only when the host is
       genuinely async.
    6. **()** User-supplied callbacks (``observability_callback``,
       post-decision adapter code) are invoked OUTSIDE the kernel's lock.
    7. **(STAGE-5C-003 hard warning)** Adapters MUST NOT route raw tool
       output / SLM-recalled text into prompt slots without first applying
       ``agent_amplifier.semantic_modifiers._neutralize_xml`` AND must trust
       the kernel's nonce-envelope smuggling-detector for cross-iteration
       protection. The kernel applies layer-1 ``neutralize`` at the OUTER
       boundary; this docstring is the contract for adapter authors.
    """

    framework_name: ClassVar[str] = ""
    version: ClassVar[str] = "0.0.0"

    INSTALL_PERSISTENT: ClassVar[bool] = False
    """ — declare whether ``install()`` writes any persistent
    state.  File-based adapters (Claude Code, Cursor, GitHub Copilot) leave
    this False because installation is a process-local marker; the host's
    own files are read on demand and writes happen on ``finalize()``.
    Framework adapters with no persistent state also leave it False.

    The CLI's ``agent-amp install`` uses this to print an honest message —
    ``installed`` for persistent adapters, ``ready`` (with explanation)
    for marker-only adapters.  Authors of adapters that DO write to disk
    or register a host-level hook should set this to True.
    """

    def __init__(
        self, kernel: AgentAmplifier | AsyncAgentAmplifier | None
    ) -> None:
        # enforce regex at construction. Raises BEFORE any I/O.
        if not _NAME_RE.match(self.framework_name or ""):
            raise TypeError(
                f"{type(self).__name__} must set class attribute "
                f"`framework_name` matching ^[a-z][a-z0-9_]{{2,31}}$, "
                f"got {self.framework_name!r}"
            )
        self.kernel = kernel
        self._installed: bool = False

    # ------------------------------------------------------------------
    # discovery ()
    # ------------------------------------------------------------------

    @classmethod
    def detect(cls) -> bool:
        """Return ``True`` if this adapter's host framework is installed.

        Default: check ``framework_name`` against
        ``importlib.util.find_spec``. Adapters MAY override (e.g. Claude
        Code checks ``~/.claude/settings.json``; MCP adapter checks
        ``$PATH`` for the host).
        """
        if not cls.framework_name:
            return False
        try:
            return importlib.util.find_spec(cls.framework_name) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def install(self) -> None:
        """Attach amplifier hooks to the host framework. Idempotent or raise."""

    @abstractmethod
    def uninstall(self) -> None:
        """Detach amplifier hooks. Surgical: only OUR hooks, never others'."""

    def is_installed(self) -> bool:
        """Default: track via the in-process flag. Subclasses MAY override."""
        return self._installed

    # ------------------------------------------------------------------
    # hot path (sync)
    # ------------------------------------------------------------------

    @abstractmethod
    def on_before_step(self, context: dict[str, Any]) -> dict[str, Any]:
        """Translate framework event → kernel call → modified context.

        STAGE-5C-003 contract: do NOT splice raw tool output or
        SLM-recalled text into prompt slots without ``_neutralize_xml``.
        """

    @abstractmethod
    def on_after_step(
        self,
        context: dict[str, Any],
        result: dict[str, Any] | str,
    ) -> dict[str, Any]:
        """Feed framework result back to kernel; return decision dict.

        Decision dict MUST carry ``"action"`` ∈ ``{"continue","stop","re_anchor"}``.
        """

    # ------------------------------------------------------------------
    # hot path (async, default = bridge to sync via anyio)
    # ------------------------------------------------------------------

    async def aon_before_step(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Async sibling of ``on_before_step``. Default impl bridges to sync.

        Override when the host framework is genuinely async (LangGraph,
        async LangChain callbacks). The default uses
        ``anyio.to_thread.run_sync`` so adapter authors who only implement
        the sync method still work in async harnesses.
        """
        import anyio

        return await anyio.to_thread.run_sync(self.on_before_step, context)

    async def aon_after_step(
        self,
        context: dict[str, Any],
        result: dict[str, Any] | str,
    ) -> dict[str, Any]:
        """Async sibling of ``on_after_step``. Default impl bridges to sync."""
        import anyio

        return await anyio.to_thread.run_sync(
            self.on_after_step, context, result
        )

    # ------------------------------------------------------------------
    # memory plane (V2.1,  V2.1 §2.5.1) — universal hooks.
    # Default no-op. Concrete adapters override to bind to host memory:
    #   - ClaudeCodeAdapter reads CLAUDE.md / MEMORY.md
    #   - CursorAdapter reads .cursor/rules/*.mdc
    #   - LangChainAdapter calls memory.load_memory_variables({"input": query})
    #   - SemanticKernelAdapter calls VectorStoreCollection.search(...)
    # MUST NOT raise. On error, log at WARNING and return.
    # ------------------------------------------------------------------

    def default_memory_recall(
        self, query: str, limit: int = 3
    ) -> list[RecalledPattern]:
        """Read host-native memory and return up to ``limit`` patterns.

        Default: empty list. Subclasses override per host. The kernel
        applies ``recall_safety.apply_recall_safety()`` to every chunk's
        ``text`` regardless. MUST NOT raise; on error, log at WARNING and
        return ``[]``.
        """
        return []

    def default_memory_remember(self, outcome: Outcome) -> None:
        """Persist the amplifier outcome to host-native memory.

        Default: no-op. Fire-and-forget. MUST NOT raise; on error, log at
        WARNING and return. The kernel does not verify a write.
        """
        return None

    # ------------------------------------------------------------------
    # helpers (subclass plumbing)
    # ------------------------------------------------------------------

    def _mark_installed(self) -> None:
        if self._installed:
            raise AdapterAlreadyInstalledError(
                f"{self.framework_name} adapter already installed"
            )
        self._installed = True
        LOG.info("Adapter installed: %s", self.framework_name)

    def _mark_uninstalled(self) -> None:
        if not self._installed:
            raise AdapterNotInstalledError(
                f"{self.framework_name} adapter not installed"
            )
        self._installed = False
        LOG.info("Adapter uninstalled: %s", self.framework_name)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"framework={self.framework_name!r}, "
            f"version={self.version}, "
            f"installed={self._installed})"
        )


__all__ = [
    "AdapterAlreadyInstalledError",
    "AdapterBase",
    "AdapterError",
    "AdapterNotInstalledError",
]
