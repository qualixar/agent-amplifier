# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Path-safety helpers for host-adapter file I/O.

Defense layers (SEC-03/04 + Codex /009/010/011):

* **Symlink at target** — SEC-03 ``.cursor/rules/poisoned.mdc``
  resolving to ``/etc/hosts``.  Closed by ``is_symlink()`` rejection +
  ``O_NOFOLLOW`` on the open syscall.
* **TOCTOU on read** — .  A file passing ``is_symlink``
  could be swapped before ``read_text``.  Closed by reading from the open
  ``fd`` (``os.open(O_RDONLY|O_NOFOLLOW)`` + ``os.fstat`` + ``os.fdopen``).
* **TOCTOU on append** — SEC-04.  Closed by ``O_NOFOLLOW`` on the
  open syscall (POSIX) so a swapped symlink raises ``ELOOP``.
* **Symlinked allowed_root** — .  An ``allowed_root`` that
  is itself a symlink would authorise reads in the symlink's target tree.
  Closed by ``is_symlink()`` rejection on the root before any other check.
* **Symlinked parent directories** — .  An attacker can
  swap a parent dir like ``.cursor/rules`` for a symlink even when the
  final target is a regular file, redirecting writes elsewhere.  Closed by
  ``_parent_chain_is_safe`` walking each segment between ``allowed_root``
  and ``path`` and refusing if any component is a symlink.
* **Windows reparse points** — .  Windows lacks portable
  ``O_NOFOLLOW``; we instead reject paths whose ``stat.st_file_attributes``
  has ``FILE_ATTRIBUTE_REPARSE_POINT`` set, which catches NTFS junctions
  and symlinks.  Combined with ``Path.is_symlink()`` this gives best-effort
  parity with the POSIX guarantees.

Both helpers are pure stdlib + ``pathlib`` — zero new dependencies.
"""
from __future__ import annotations

import contextlib
import os
import stat
import sys
from pathlib import Path
from typing import IO

# Windows file-attribute constant (not exported by ``stat`` on POSIX).
# Source: MSDN <https://learn.microsoft.com/windows/win32/fileio/file-attribute-constants>.
_FILE_ATTRIBUTE_REPARSE_POINT: int = 0x400


def _is_windows() -> bool:
    """Indirection so mypy does not narrow the platform Literal at the
    call site (``sys.platform == "win32"`` makes mypy treat the body as
    unreachable on POSIX, which trips ``warn_unreachable``).
    """
    return bool(sys.platform == "win32")


def _resolve_safely(path: Path) -> Path | None:
    """Resolve ``path`` without following the final symlink; return ``None``
    if any step fails. We use ``Path.resolve(strict=False)`` because the file
    is allowed to not exist yet (e.g. write target).
    """
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError):
        # RuntimeError can be raised by resolve() on cycles; OSError on
        # broken paths, permission errors, etc. Treat all as "unsafe".
        return None


def _is_inside(child: Path, parent: Path) -> bool:
    """Return True iff ``child`` is the same as or a descendant of ``parent``.

    Both arguments are expected to be ``resolve()``-d already. Uses
    ``Path.is_relative_to`` (Python 3.9+, project requires 3.11+).
    """
    try:
        return child.is_relative_to(parent)
    except (OSError, ValueError):  # pragma: no cover - defensive
        return False


def _is_symlink_or_reparse(path: Path) -> bool:
    """Return True iff ``path`` is a POSIX symlink OR a Windows reparse point.

    on Windows ``Path.is_symlink`` covers true symlinks
    but not all reparse points (junctions, mount points, OneDrive cloud
    files).  We additionally check ``stat.st_file_attributes`` when
    available so a junction-redirected directory is treated as a symlink.

    Returns ``True`` on any error reading the attribute, on the principle
    of "fail closed for an unparseable path".
    """
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True
    if not _is_windows():
        return False
    try:
        st = path.lstat()
    except OSError:
        return True
    file_attrs = getattr(st, "st_file_attributes", 0)
    return bool(file_attrs & _FILE_ATTRIBUTE_REPARSE_POINT)


def _parent_chain_is_safe(path: Path, allowed_root: Path) -> bool:
    """Verify every directory from ``allowed_root`` down to ``path.parent`` is
    a non-symlink, non-reparse directory.

    even when the FINAL target is opened with
    ``O_NOFOLLOW``, a symlinked parent directory can redirect the path
    traversal before the kernel reaches the final segment.  This walker
    closes that gap.

    Returns ``False`` on any unreadable / non-directory / symlinked component.
    The path itself is NOT walked (it is checked separately by the caller
    so the file may not exist yet for write paths).
    """
    try:
        parent = path.parent
    except (OSError, ValueError):  # pragma: no cover - defensive
        return False
    try:
        rel = parent.relative_to(allowed_root)
    except ValueError:
        return False
    # Belt-and-suspenders: callers (safe_read_text / safe_open_append /
    # safe_open_write) already reject a symlinked allowed_root before
    # invoking us, so this branch is preserved as defense-in-depth for any
    # future direct caller.
    if _is_symlink_or_reparse(allowed_root):  # pragma: no cover - defensive
        return False
    current = allowed_root
    for part in rel.parts:
        if part in ("", "."):  # pragma: no cover - relative_to strips these
            continue
        current = current / part
        if _is_symlink_or_reparse(current):
            return False
        # Tolerate not-yet-existing intermediate dirs for write paths; the
        # actual ``os.open`` will fail loudly if the parent does not exist.
        if not current.exists():
            return True
        if not current.is_dir():
            return False
    return True


def safe_read_text(path: Path, allowed_root: Path) -> str | None:
    """Read ``path`` as UTF-8 only when it is safely inside ``allowed_root``.

    Refuses (returns ``None``) when ANY of:

    * ``allowed_root`` itself is a symlink / reparse point ().
    * ``path`` is itself a symlink / reparse point (SEC-03).
    * ``path.resolve()`` lies outside ``allowed_root.resolve()``.
    * Any directory between ``allowed_root`` and ``path.parent`` is a
      symlink / reparse point ().
    * The file cannot be opened, is not a regular file, or decode fails.

    The actual read is performed from the descriptor returned by
    ``os.open(path, O_RDONLY|O_NOFOLLOW)`` so a symlink swap between the
    pre-flight checks and the read raises ``OSError`` instead of leaking
    attacker content (— TOCTOU).

    Args:
        path: target file. May or may not exist; existence is checked.
        allowed_root: directory the resolved path MUST live inside.

    Returns:
        The file text on success, ``None`` on any safety or I/O failure.
    """
    if _is_symlink_or_reparse(allowed_root):
        return None
    if _is_symlink_or_reparse(path):
        return None

    resolved = _resolve_safely(path)
    if resolved is None:
        return None
    resolved_root = _resolve_safely(allowed_root)
    if resolved_root is None:
        return None
    if not _is_inside(resolved, resolved_root):
        return None

    if not _parent_chain_is_safe(path, allowed_root):
        return None

    return _read_from_fd(path)


def _read_from_fd(path: Path) -> str | None:
    """Open ``path`` with ``O_NOFOLLOW`` and read via ``fdopen``.

    (TOCTOU): the read happens from the descriptor
    obtained at the moment of the safety checks; a later symlink swap
    cannot redirect this read.  ``fstat`` confirms the descriptor is a
    regular file before we expose its contents.

    Windows: ``O_NOFOLLOW`` is not portable, so we fall back to
    ``Path.open()``.  The reparse-point check in ``_is_symlink_or_reparse``
    is the primary defense on that platform.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):  # POSIX  # pragma: no branch
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError:
        return None

    handle: IO[str] | None = None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None
        handle = os.fdopen(fd, "r", encoding="utf-8", errors="replace")
        # fd ownership transferred to ``handle`` once fdopen succeeds.
        return handle.read()
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if handle is not None:
            with contextlib.suppress(OSError):  # pragma: no cover - extremely rare
                handle.close()
        else:
            with contextlib.suppress(OSError):  # pragma: no cover - extremely rare
                os.close(fd)


def safe_open_append(
    path: Path, allowed_root: Path | None = None
) -> IO[str] | None:
    """Open ``path`` for UTF-8 append with symlink + parent-chain defense.

    Refuses (returns ``None``) when ANY of:

    * ``allowed_root`` (when provided) is a symlink / reparse point
      ().
    * ``path`` is a symlink / reparse point (SEC-04).
    * Any parent directory between ``allowed_root`` and ``path.parent``
      is a symlink / reparse point ().
    * ``os.open(O_NOFOLLOW)`` raises (POSIX symlink at the final segment,
      permission, etc.).

    Args:
        path: file to append to.
        allowed_root: optional trusted root; when provided, the parent
            chain is validated.  Pass it whenever the caller knows the
            project root.

    Caller is responsible for closing the returned handle.

    Windows: ``O_NOFOLLOW`` is not portable.  We rely on
    ``_is_symlink_or_reparse`` to reject reparse points and fall back to
    ``Path.open("a")`` for the actual write.  This is best-effort —
    Windows users with privileged attackers should treat file-based
    remember as advisory.
    """
    if allowed_root is not None:
        if _is_symlink_or_reparse(allowed_root):
            return None
        if not _parent_chain_is_safe(path, allowed_root):
            return None

    if _is_symlink_or_reparse(path):
        return None

    if _is_windows():  # pragma: no cover - Windows-only branch
        try:
            return path.open("a", encoding="utf-8")
        except OSError:
            return None

    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o644)
    except OSError:
        # ELOOP (symlink), EACCES, or similar — refuse to open.
        return None
    try:
        return os.fdopen(fd, "a", encoding="utf-8")
    except OSError:  # pragma: no cover - extremely rare after successful os.open
        os.close(fd)
        return None


def safe_open_write(
    path: Path, allowed_root: Path | None = None
) -> IO[str] | None:
    """Open ``path`` for create-or-truncate UTF-8 write with full defense.

    Same defense layers as ``safe_open_append`` (allowed_root symlink
    rejection, parent-chain validation, ``O_NOFOLLOW`` on POSIX).  Use
    when the caller wants ``O_WRONLY|O_CREAT|O_TRUNC`` semantics rather
    than append (e.g. writing a fresh ``.cursor/rules/*.mdc`` file).

    Caller is responsible for closing the returned handle.
    """
    if allowed_root is not None:
        if _is_symlink_or_reparse(allowed_root):
            return None
        if not _parent_chain_is_safe(path, allowed_root):
            return None

    if _is_symlink_or_reparse(path):
        return None

    if _is_windows():  # pragma: no cover - Windows-only branch
        try:
            return path.open("w", encoding="utf-8")
        except OSError:
            return None

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o644)
    except OSError:
        return None
    try:
        return os.fdopen(fd, "w", encoding="utf-8")
    except OSError:  # pragma: no cover - extremely rare after successful os.open
        os.close(fd)
        return None


__all__ = ["safe_open_append", "safe_open_write", "safe_read_text"]
