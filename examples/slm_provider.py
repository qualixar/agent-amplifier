# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Qualixar
"""Example memory provider for SuperLocalMemory (SLM).

This file is NOT part of the agent_amplifier core package. It is a
working drop-in reference that any SLM user can copy / import to wire
SLM into the universal memory plane introduced in V2.1.

Wiring pattern::

    from agent_amplifier import AgentAmplifier, RecalledPattern, Outcome
    # from agent_amplifier import ClaudeCodeAdapter   # in Wave 7.6.B
    from examples.slm_provider import SLMProvider

    slm = SLMProvider()
    amp = AgentAmplifier(
        # adapter=ClaudeCodeAdapter(...),
        memory_recall=slm.recall,
        memory_remember=slm.remember,
    )

The provider exposes two callables matching the V2.1 contract:

    recall(query: str, limit: int = 3) -> list[RecalledPattern]
    remember(outcome: Outcome) -> None

It preserves the V2.0 SLM-specific defenses (HMAC signing, tag-allowlist,
sentinel-prefixed argv). Universal injection-defense (cap + neutralize +
smuggling-detect) is now performed by the kernel via
``agent_amplifier._internal.recall_safety`` regardless of which provider
produced the text.
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import stat
import subprocess
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Protocol

from agent_amplifier.types import EffortLevel, Outcome, RecalledPattern

LOG = logging.getLogger("agent_amplifier.examples.slm_provider")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SLM_TIMEOUT_S: float = 0.2
_NAMESPACE: str = "agent-amplifier.v1"
_TAG_OUTCOME: str = f"{_NAMESPACE}.outcome"
_TAG_PATTERN: str = f"{_NAMESPACE}.pattern"
_TAG_RE: re.Pattern[str] = re.compile(r"\A[A-Za-z0-9._-]{1,64}\Z")
_TEXT_SENTINEL: str = "amplifier:"
_KEY_PATH: Path = Path.home() / ".agent-amplifier" / "session.key"
# Stage 10 CODEX-044: bound dedup memory at 4096 outcomes (~256 KB worst-case).
_DEDUP_MAX: int = 4096


# ---------------------------------------------------------------------------
# Subprocess injection seam (test-friendly)
# ---------------------------------------------------------------------------


class SubprocessRunner(Protocol):
    def __call__(  # pragma: no cover  (Protocol stub)
        self,
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]: ...


# ---------------------------------------------------------------------------
# Helpers (pure)
# ---------------------------------------------------------------------------


def _safe_tag(t: Any) -> str:
    if not isinstance(t, str) or not _TAG_RE.match(t) or t.startswith("-"):
        raise ValueError(f"refusing unsafe SLM tag: {t!r}")
    return t


def _read_or_create_session_key() -> bytes:
    """Read or generate the HMAC key (0600 perms). 32-byte secret.

    Stage 10 CODEX-045: errors do NOT include the absolute key path because
    the path embeds the user's home directory.  Caller logs error class only.
    """
    _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _KEY_PATH.exists():
        if os.name == "posix":
            mode = _KEY_PATH.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise PermissionError(
                    f"refusing to use session key with permissive mode "
                    f"{oct(mode)} (set 0600 to enable)"
                )
        return _KEY_PATH.read_bytes()
    key = secrets.token_bytes(32)
    _KEY_PATH.write_bytes(key)
    if os.name == "posix":
        os.chmod(_KEY_PATH, 0o600)
    return key


def _hmac_payload(key: bytes, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()[:16]


def _idempotency_key(
    query: str, effort: EffortLevel, iterations: int, quality: float
) -> str:
    raw = f"{query[:200]}|{effort.value}|{iterations}|{round(quality, 3)}"
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SLMProvider — drop-in callbacks for the agent-amplifier memory plane.
# ---------------------------------------------------------------------------


class SLMProvider:
    """Optional bridge to SuperLocalMemory. Always-instantiable, never raises.

    Public callable surface (match V2.1 memory plane contract):

        recall(query: str, limit: int = 3) -> list[RecalledPattern]
        remember(outcome: Outcome) -> None

    Pass ``provider.recall`` / ``provider.remember`` as ``memory_recall`` /
    ``memory_remember`` callbacks on ``AgentAmplifier``.
    """

    _warned_once: ClassVar[set[tuple[str, str | None]]] = set()

    def __init__(
        self,
        *,
        project_path: str | None = None,
        enabled: bool | None = None,
        runner: SubprocessRunner | None = None,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._project_path: str | None = project_path
        self._runner: SubprocessRunner = runner or self._default_runner
        self._which: Callable[[str], str | None] = which
        self._executable: str | None = None
        self._reason: str = "uninitialized"
        self._enabled: bool = False
        # Stage 10 CODEX-044: bounded FIFO instead of unbounded set so a
        # long-running session cannot leak memory.  Mirror the kernel's
        # dedup cap (~256 KB worst-case at 4096 keys * 64 hex chars).
        self._dedup: OrderedDict[str, None] = OrderedDict()
        self._dedup_lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._key: bytes | None = None

        if enabled is False:
            self._reason = "disabled by caller"
        else:
            exe = self._which("slm")
            if exe is None:
                self._reason = "slm CLI not found on PATH"
            elif Path(exe).suffix.lower() in {".bat", ".cmd"}:
                self._reason = (
                    f"refusing slm shim with extension {Path(exe).suffix!r}"
                )
                LOG.warning("SLMProvider disabled: %s", self._reason)
            else:
                self._executable = exe
                if self._probe_slm():
                    self._enabled = True
                    self._reason = "ok"
                else:
                    self._reason = "slm probe failed"

        if self._enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="slm"
            )
            try:
                self._key = _read_or_create_session_key()
            except Exception as e:
                # Stage 10 CODEX-045: log only the error class so we never
                # leak the absolute key path (which embeds the user's home).
                LOG.warning(
                    "SLMProvider: HMAC key unavailable (%s); disabling.",
                    type(e).__name__,
                )
                self._enabled = False
                self._reason = f"hmac key error: {type(e).__name__}"
                if self._executor is not None:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                    self._executor = None

        warn_key = (self._reason, self._project_path)
        if not self._enabled and self._reason != "disabled by caller":
            if warn_key not in type(self)._warned_once:
                LOG.warning(
                    "SLM cross-session learning is disabled (%s). "
                    "To enable: pip install superlocalmemory && slm init.",
                    self._reason,
                )
                type(self)._warned_once.add(warn_key)
        else:
            LOG.info(
                "SLMProvider enabled=%s reason=%s",
                self._enabled,
                self._reason,
            )

    # ------------------------------------------------------------------
    # Default subprocess runner (Popen with explicit kill+wait)
    # ------------------------------------------------------------------

    def _default_runner(
        self,
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=text,
            shell=False,
            executable=self._executable,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                LOG.error("SLMProvider: subprocess refused to die after kill")
            raise
        rc = proc.returncode
        if check and rc != 0:
            raise subprocess.CalledProcessError(rc, args, stdout, stderr)
        return subprocess.CompletedProcess(args, rc, stdout, stderr)

    # ------------------------------------------------------------------
    # Detection probe
    # ------------------------------------------------------------------

    def _probe_slm(self) -> bool:
        try:
            r = self._runner(
                [self._executable or "slm", "--version"],
                capture_output=True,
                text=True,
                timeout=_SLM_TIMEOUT_S,
                check=False,
            )
            return r.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def is_available(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # V2.1 memory-plane callable surface — recall + remember
    # ------------------------------------------------------------------

    def recall(self, query: str, limit: int = 3) -> list[RecalledPattern]:
        """Synchronous recall. Returns ``[]`` when disabled."""
        return self._recall_blocking(query, limit) if self._enabled else []

    def start_recall(
        self, query: str, *, limit: int = 3
    ) -> Future[list[RecalledPattern]]:
        """Submit a recall query as a background future (still useful for callers
        that want to overlap recall with other work)."""
        if not self._enabled or not query.strip() or self._executor is None:
            f: Future[list[RecalledPattern]] = Future()
            f.set_result([])
            return f
        return self._executor.submit(self._recall_blocking, query, limit)

    def _recall_blocking(
        self, query: str, limit: int
    ) -> list[RecalledPattern]:
        if not self._enabled or not query.strip():
            return []
        try:
            user_tag = _safe_tag(f"{_NAMESPACE}.user.{getpass.getuser()}")
        except ValueError:
            user_tag = _TAG_OUTCOME

        try:
            r = self._runner(
                [
                    self._executable or "slm",
                    "recall",
                    query,
                    "--tag",
                    _TAG_OUTCOME,
                    "--tag",
                    user_tag,
                    "--limit",
                    str(int(limit)),
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=_SLM_TIMEOUT_S,
                check=False,
            )
            if r.returncode != 0:
                return []
            payload = json.loads(r.stdout or "{}")
            results = payload.get("data", {}).get("results", []) or []
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            return []

        out: list[RecalledPattern] = []
        for item in results:
            content = item.get("content", "") or ""
            sig = item.get("hmac") or item.get("metadata", {}).get("hmac")
            payload_inner = (
                item.get("payload")
                or item.get("metadata", {}).get("payload")
            )
            if not (self._key and sig and payload_inner):
                continue
            try:
                if not hmac.compare_digest(
                    _hmac_payload(self._key, payload_inner), str(sig)
                ):
                    continue
            except (TypeError, ValueError):
                continue

            text_raw = str(payload_inner.get("query_summary", content))
            out.append(
                RecalledPattern(
                    text=text_raw,
                    score=float(item.get("score", 0.0)),
                    tags=tuple(item.get("tags", []) or ()),
                    source=str(
                        item.get("session_id") or item.get("id") or ""
                    ),
                )
            )
        return out

    def remember(self, outcome: Outcome) -> None:
        """V2.1 callable: persist an Outcome. Best-effort. HMAC-bound. Idempotent."""
        self.remember_outcome(
            query=outcome.query,
            effort=outcome.effort,
            iterations=outcome.iterations,
            quality=outcome.quality,
            converged=outcome.converged,
            tokens_used=outcome.tokens_used,
        )

    def remember_outcome(
        self,
        *,
        query: str,
        effort: EffortLevel,
        iterations: int,
        quality: float,
        converged: bool,
        tokens_used: int,
        extra_tags: Iterable[str] = (),
    ) -> None:
        """Persist an amplification outcome. Best-effort. HMAC-bound. Idempotent."""
        if not self._enabled:
            return
        safe_query = (query or "")[:200]
        if not safe_query.strip():
            return

        idem = _idempotency_key(safe_query, effort, iterations, quality)
        with self._dedup_lock:
            if idem in self._dedup:
                LOG.debug("SLMProvider.remember dedup hit %s", idem)
                return
            self._dedup[idem] = None
            # Stage 10 CODEX-044: evict oldest entries past the cap so the
            # dedup cache cannot grow without bound across long sessions.
            while len(self._dedup) > _DEDUP_MAX:
                self._dedup.popitem(last=False)

        payload: dict[str, Any] = {
            "query_summary": safe_query,
            "effort": effort.value,
            "iterations": int(iterations),
            "quality": round(float(quality), 3),
            "converged": bool(converged),
            "tokens_used": int(tokens_used),
            "user": getpass.getuser(),
            "idem_id": idem,
        }
        sig = _hmac_payload(self._key, payload) if self._key else ""
        text = _TEXT_SENTINEL + json.dumps(
            {"payload": payload, "hmac": sig}, separators=(",", ":")
        )

        try:
            tags = [
                _safe_tag(_TAG_OUTCOME),
                _safe_tag(f"{_NAMESPACE}.effort.{effort.value}"),
                _safe_tag(f"{_NAMESPACE}.user.{getpass.getuser()}"),
                _safe_tag(f"{_NAMESPACE}.idem.{idem}"),
                *[_safe_tag(t) for t in extra_tags],
            ]
        except ValueError as e:
            LOG.warning("SLMProvider.remember rejected tags: %s", str(e))
            with self._dedup_lock:
                self._dedup.pop(idem, None)
            return

        try:
            self._runner(
                [
                    self._executable or "slm",
                    "remember",
                    text,
                    *(arg for t in tags for arg in ("--tag", t)),
                ],
                capture_output=True,
                text=True,
                timeout=_SLM_TIMEOUT_S,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __repr__(self) -> str:
        return (
            f"SLMProvider(enabled={self._enabled}, reason={self._reason!r})"
        )


__all__ = [
    "SLMProvider",
    "SubprocessRunner",
]
