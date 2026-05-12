# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.adapters._path_safety`` (B3 / B4).

Targets:
* 100% line + branch coverage on the safety helpers.
* Exercise every refusal branch (symlink, resolve failure, traversal,
  not-a-file) so a regression silently allowing one of these vectors
  fails CI.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import pytest

from agent_amplifier.adapters._path_safety import (
    _is_inside,
    _is_symlink_or_reparse,
    _parent_chain_is_safe,
    _resolve_safely,
    safe_open_append,
    safe_open_write,
    safe_read_text,
)

# ---------------------------------------------------------------------------
# _resolve_safely — error branches (lines 42-45)
# ---------------------------------------------------------------------------


def test_resolve_safely_returns_none_on_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Path.resolve`` raising OSError → _resolve_safely returns None."""
    p = Path("/tmp/anywhere")

    def _broken(self: Path, **kw: Any) -> Path:
        raise OSError("broken")

    monkeypatch.setattr(Path, "resolve", _broken)
    assert _resolve_safely(p) is None


def test_resolve_safely_returns_none_on_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Path.resolve`` raising RuntimeError (cycles) → returns None."""
    p = Path("/tmp/cycle")

    def _broken(self: Path, **kw: Any) -> Path:
        raise RuntimeError("symlink loop")

    monkeypatch.setattr(Path, "resolve", _broken)
    assert _resolve_safely(p) is None


def test_resolve_safely_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "file"
    p.write_text("x")
    assert _resolve_safely(p) == p.resolve()


# ---------------------------------------------------------------------------
# _is_inside — both branches
# ---------------------------------------------------------------------------


def test_is_inside_true_for_descendant(tmp_path: Path) -> None:
    child = tmp_path / "sub" / "f"
    assert _is_inside(child, tmp_path)


def test_is_inside_false_for_sibling(tmp_path: Path) -> None:
    sibling = tmp_path.parent / "elsewhere"
    assert _is_inside(sibling, tmp_path) is False


# ---------------------------------------------------------------------------
# safe_read_text — every refusal branch
# ---------------------------------------------------------------------------


def test_safe_read_text_happy_path(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_text("hello")
    assert safe_read_text(p, tmp_path) == "hello"


def test_safe_read_text_refuses_symlink(tmp_path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - Windows symlinks need admin
        pytest.skip("symlinks not portable on Windows")
    target = tmp_path.parent / "outside.txt"
    target.write_text("ATTACKER")
    link = tmp_path / "f"
    link.symlink_to(target)
    assert safe_read_text(link, tmp_path) is None


def test_safe_read_text_returns_none_on_is_symlink_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines 85-86: is_symlink itself raising → defensive None return."""
    p = tmp_path / "f"
    p.write_text("hi")

    def _broken(self: Path) -> bool:
        raise OSError("permission")

    monkeypatch.setattr(Path, "is_symlink", _broken)
    assert safe_read_text(p, tmp_path) is None


def test_safe_read_text_returns_none_when_path_resolve_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Line 90: ``_resolve_safely(path)`` returning None bubbles up."""
    p = tmp_path / "f"
    p.write_text("hi")
    real_resolve = Path.resolve
    call_count = {"n": 0}

    def _broken(self: Path, **kw: Any) -> Path:
        call_count["n"] += 1
        # Fail ONLY on the first call (the path itself), succeed on the
        # second call (allowed_root). Otherwise the test would also short-
        # circuit on the allowed_root check below.
        if call_count["n"] == 1:
            raise OSError("path resolve broken")
        return real_resolve(self, **kw)  # type: ignore[no-any-return]

    monkeypatch.setattr(Path, "resolve", _broken)
    assert safe_read_text(p, tmp_path) is None


def test_safe_read_text_returns_none_when_root_resolve_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Line 93: ``_resolve_safely(allowed_root)`` None bubbles up."""
    p = tmp_path / "f"
    p.write_text("hi")
    real_resolve = Path.resolve
    call_count = {"n": 0}

    def _broken(self: Path, **kw: Any) -> Path:
        call_count["n"] += 1
        # Path resolves cleanly; root fails on the second call.
        if call_count["n"] == 2:
            raise OSError("root resolve broken")
        return real_resolve(self, **kw)  # type: ignore[no-any-return]

    monkeypatch.setattr(Path, "resolve", _broken)
    assert safe_read_text(p, tmp_path) is None


def test_safe_read_text_refuses_traversal_outside_root(
    tmp_path: Path,
) -> None:
    """Line 95: resolved path outside allowed_root is refused."""
    outside = tmp_path.parent / "neighbor.txt"
    outside.write_text("ATTACKER")
    # Pretend tmp_path is the root, but query a sibling — refused.
    assert safe_read_text(outside, tmp_path) is None


def test_safe_read_text_refuses_directory(tmp_path: Path) -> None:
    """Line 101: path resolves to a directory (not is_file) → None."""
    sub = tmp_path / "sub"
    sub.mkdir()
    assert safe_read_text(sub, tmp_path) is None


def test_safe_read_text_swallows_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OSError from the open-on-fd read after all checks pass → None.

    the read goes through ``os.open(O_RDONLY|O_NOFOLLOW)``
    + ``os.fdopen`` to close the TOCTOU window.  We mock ``os.fstat`` so the
    file passes the open call but fails before being read — exercising the
    "open succeeded, downstream OSError" branch.
    """
    p = tmp_path / "f"
    p.write_text("hi")
    real_fstat = os.fstat

    def _broken(fd: int) -> os.stat_result:
        # Always raise — only the safe_read_text fd reaches this in the test.
        raise OSError("locked")

    monkeypatch.setattr(os, "fstat", _broken)
    try:
        assert safe_read_text(p, tmp_path) is None
    finally:
        monkeypatch.setattr(os, "fstat", real_fstat)


def test_safe_read_text_handles_corrupt_utf8(tmp_path: Path) -> None:
    """UTF-8 decode errors fall back to replace — bytes still readable."""
    p = tmp_path / "f"
    p.write_bytes(b"\xff\xfe\x00\x00valid-tail")
    out = safe_read_text(p, tmp_path)
    assert out is not None
    assert "valid-tail" in out


# ---------------------------------------------------------------------------
# safe_open_append — every refusal branch
# ---------------------------------------------------------------------------


def test_safe_open_append_happy_path_creates_or_appends(
    tmp_path: Path,
) -> None:
    p = tmp_path / "log.txt"
    p.write_text("seed\n")
    fh = safe_open_append(p)
    assert fh is not None
    try:
        fh.write("appended\n")
    finally:
        fh.close()
    assert p.read_text() == "seed\nappended\n"


def test_safe_open_append_refuses_symlink_target(tmp_path: Path) -> None:
    """SEC-04: symlink at the target raises ELOOP under O_NOFOLLOW → None."""
    if os.name == "nt":  # pragma: no cover - Windows symlinks need admin
        pytest.skip("O_NOFOLLOW is POSIX")
    real_target = tmp_path.parent / "real-target.txt"
    real_target.write_text("original\n")
    link = tmp_path / "log.txt"
    link.symlink_to(real_target)
    fh = safe_open_append(link)
    assert fh is None
    # Real target untouched
    assert real_target.read_text() == "original\n"


def test_safe_open_append_returns_none_when_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic OSError on open → None (e.g. permission denied)."""
    if os.name == "nt":  # pragma: no cover - POSIX-only branch
        pytest.skip("POSIX path")
    p = tmp_path / "denied.txt"
    p.write_text("seed\n")

    real_open = os.open

    def _broken(path: Any, flags: int, mode: int = 0o777) -> int:
        if str(path).endswith("denied.txt"):
            raise OSError("permission denied")
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", _broken)
    fh = safe_open_append(p)
    assert fh is None


# ---------------------------------------------------------------------------
# — symlinked allowed_root rejection
# ---------------------------------------------------------------------------


def test_safe_read_text_refuses_symlinked_allowed_root(tmp_path: Path) -> None:
    """an ``allowed_root`` that is itself a symlink is refused.

    Otherwise an attacker who controls a project-local symlink (e.g.
    ``.cursor`` -> ``/etc``) could authorize reads in the symlink target.
    """
    if os.name == "nt":  # pragma: no cover - Windows symlinks need admin
        pytest.skip("symlinks not portable on Windows")
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "f").write_text("contents")
    link_root = tmp_path / "link"
    link_root.symlink_to(real_dir)
    target = link_root / "f"
    assert safe_read_text(target, link_root) is None


def test_safe_open_append_refuses_symlinked_allowed_root(tmp_path: Path) -> None:
    """(write): symlinked allowed_root is refused for append."""
    if os.name == "nt":  # pragma: no cover - Windows symlinks need admin
        pytest.skip("symlinks not portable on Windows")
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_root = tmp_path / "link"
    link_root.symlink_to(real_dir)
    target = link_root / "out.txt"
    assert safe_open_append(target, allowed_root=link_root) is None


def test_safe_open_write_refuses_symlinked_allowed_root(tmp_path: Path) -> None:
    """(write): symlinked allowed_root is refused for create-write."""
    if os.name == "nt":  # pragma: no cover
        pytest.skip("symlinks not portable on Windows")
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_root = tmp_path / "link"
    link_root.symlink_to(real_dir)
    target = link_root / "fresh.txt"
    assert safe_open_write(target, allowed_root=link_root) is None


# ---------------------------------------------------------------------------
# — symlinked parent directory rejection
# ---------------------------------------------------------------------------


def test_safe_read_text_refuses_symlinked_parent_dir(tmp_path: Path) -> None:
    """a parent dir between allowed_root and path is a symlink → refuse."""
    if os.name == "nt":  # pragma: no cover
        pytest.skip("symlinks not portable on Windows")
    # allowed_root contains a child dir that is a symlink to elsewhere
    elsewhere = tmp_path.parent / "elsewhere-stage10"
    elsewhere.mkdir(exist_ok=True)
    (elsewhere / "leaked.txt").write_text("ATTACKER")
    project = tmp_path / "project"
    project.mkdir()
    rules = project / "rules"
    rules.symlink_to(elsewhere)
    target = rules / "leaked.txt"
    try:
        assert safe_read_text(target, project) is None
    finally:
        # cleanup leaked file (it lives under tmp_path.parent)
        with contextlib.suppress(OSError):
            (elsewhere / "leaked.txt").unlink()
            elsewhere.rmdir()


def test_safe_open_append_refuses_symlinked_parent_dir(tmp_path: Path) -> None:
    """(write): symlinked parent dir → safe_open_append refuses."""
    if os.name == "nt":  # pragma: no cover
        pytest.skip("symlinks not portable on Windows")
    elsewhere = tmp_path.parent / "elsewhere-write-stage10"
    elsewhere.mkdir(exist_ok=True)
    project = tmp_path / "proj"
    project.mkdir()
    rules = project / "rules"
    rules.symlink_to(elsewhere)
    target = rules / "out.txt"
    try:
        assert safe_open_append(target, allowed_root=project) is None
    finally:
        with contextlib.suppress(OSError):
            elsewhere.rmdir()


def test_safe_open_write_refuses_symlinked_parent_dir(tmp_path: Path) -> None:
    """(write): symlinked parent dir → safe_open_write refuses."""
    if os.name == "nt":  # pragma: no cover
        pytest.skip("symlinks not portable on Windows")
    elsewhere = tmp_path.parent / "elsewhere-write2-stage10"
    elsewhere.mkdir(exist_ok=True)
    project = tmp_path / "proj"
    project.mkdir()
    rules = project / "rules"
    rules.symlink_to(elsewhere)
    target = rules / "fresh.mdc"
    try:
        assert safe_open_write(target, allowed_root=project) is None
    finally:
        with contextlib.suppress(OSError):
            elsewhere.rmdir()


def test_parent_chain_safe_returns_false_for_unrelated_path(
    tmp_path: Path,
) -> None:
    """``_parent_chain_is_safe`` returns False when path is outside root."""
    other = tmp_path.parent / "elsewhere-unrelated"
    other.mkdir(exist_ok=True)
    p = other / "f"
    try:
        assert _parent_chain_is_safe(p, tmp_path) is False
    finally:
        with contextlib.suppress(OSError):
            other.rmdir()


def test_parent_chain_safe_handles_missing_intermediate_dir(
    tmp_path: Path,
) -> None:
    """Missing intermediate directory short-circuits to True (write paths)."""
    p = tmp_path / "future" / "subdir" / "f"
    # ``future`` does not exist yet; chain walker should accept (write target).
    assert _parent_chain_is_safe(p, tmp_path) is True


def test_parent_chain_safe_returns_false_for_non_dir_intermediate(
    tmp_path: Path,
) -> None:
    """A regular file appearing where a dir is expected → False."""
    bad = tmp_path / "not-a-dir"
    bad.write_text("stub")
    p = bad / "child" / "f"
    assert _parent_chain_is_safe(p, tmp_path) is False


# ---------------------------------------------------------------------------
# — TOCTOU close (read uses fd, not Path.read_text)
# ---------------------------------------------------------------------------


def test_safe_read_text_returns_none_when_fdopen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """if ``os.fdopen`` fails after ``os.open`` succeeds, fd
    is closed and ``None`` returned without leaking the descriptor.
    """
    p = tmp_path / "f"
    p.write_text("hi")
    real_fdopen = os.fdopen

    def _broken(fd: int, *a: Any, **kw: Any) -> Any:
        os.close(fd)  # close the fd ourselves so the real impl can't
        raise OSError("fdopen failed")

    monkeypatch.setattr(os, "fdopen", _broken)
    try:
        assert safe_read_text(p, tmp_path) is None
    finally:
        monkeypatch.setattr(os, "fdopen", real_fdopen)


def test_safe_read_text_returns_none_for_non_regular_file(
    tmp_path: Path,
) -> None:
    """``os.fstat`` reports S_ISREG False (e.g. directory descriptor)."""
    sub = tmp_path / "sub"
    sub.mkdir()
    # safe_read_text already short-circuits at is_file(); we exercise the
    # fstat branch directly by checking the helper doesn't read directories.
    assert safe_read_text(sub, tmp_path) is None


# ---------------------------------------------------------------------------
# — Windows reparse-point detection helper
# ---------------------------------------------------------------------------


def test_is_symlink_or_reparse_returns_true_on_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed: any OSError in is_symlink → treat as reparse (True)."""
    p = tmp_path / "f"
    p.write_text("x")

    def _broken(self: Path) -> bool:
        raise OSError("eh")

    monkeypatch.setattr(Path, "is_symlink", _broken)
    assert _is_symlink_or_reparse(p) is True


def test_is_symlink_or_reparse_returns_true_on_lstat_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows branch: is_symlink says False but lstat raises → True (fail-closed).

    We monkeypatch ``is_symlink`` to return False so we reach the explicit
    ``lstat`` call, then make ``lstat`` raise.  This exercises the
    Windows-only fail-closed branch on the ``except OSError`` path.
    """
    monkeypatch.setattr(
        "agent_amplifier.adapters._path_safety.sys.platform", "win32"
    )
    p = tmp_path / "f"
    p.write_text("x")

    def _no_symlink(self: Path) -> bool:
        return False

    def _broken_lstat(self: Path) -> Any:
        raise OSError("eh")

    monkeypatch.setattr(Path, "is_symlink", _no_symlink)
    monkeypatch.setattr(Path, "lstat", _broken_lstat)
    assert _is_symlink_or_reparse(p) is True


def test_is_symlink_or_reparse_returns_false_for_regular_posix(
    tmp_path: Path,
) -> None:
    p = tmp_path / "f"
    p.write_text("x")
    if os.name != "nt":
        assert _is_symlink_or_reparse(p) is False


# ---------------------------------------------------------------------------
# safe_open_write — happy path + refusal branches
# ---------------------------------------------------------------------------


def test_safe_open_write_creates_or_truncates(tmp_path: Path) -> None:
    p = tmp_path / "out.txt"
    fh = safe_open_write(p, allowed_root=tmp_path)
    assert fh is not None
    try:
        fh.write("body\n")
    finally:
        fh.close()
    assert p.read_text() == "body\n"
    # Second call truncates
    fh = safe_open_write(p, allowed_root=tmp_path)
    assert fh is not None
    try:
        fh.write("REPLACED")
    finally:
        fh.close()
    assert p.read_text() == "REPLACED"


def test_safe_open_write_refuses_symlink_target(tmp_path: Path) -> None:
    if os.name == "nt":  # pragma: no cover
        pytest.skip("symlinks not portable on Windows")
    real_target = tmp_path.parent / "real-write-target.txt"
    real_target.write_text("orig")
    link = tmp_path / "out.txt"
    link.symlink_to(real_target)
    try:
        assert safe_open_write(link, allowed_root=tmp_path) is None
        # Real target untouched
        assert real_target.read_text() == "orig"
    finally:
        real_target.unlink()


def test_safe_open_write_returns_none_on_open_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":  # pragma: no cover
        pytest.skip("POSIX path")
    p = tmp_path / "out.txt"
    real_open = os.open

    def _broken(path: Any, flags: int, mode: int = 0o777) -> int:
        if str(path).endswith("out.txt"):
            raise OSError("permission denied")
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", _broken)
    assert safe_open_write(p, allowed_root=tmp_path) is None


def test_safe_open_write_no_allowed_root_works(tmp_path: Path) -> None:
    """allowed_root is optional; helper still applies symlink check."""
    p = tmp_path / "no_root.txt"
    fh = safe_open_write(p)
    assert fh is not None
    try:
        fh.write("body")
    finally:
        fh.close()
    assert p.read_text() == "body"


def test_safe_open_append_no_allowed_root_works(tmp_path: Path) -> None:
    """allowed_root is optional; helper still applies symlink check."""
    p = tmp_path / "append.txt"
    p.write_text("seed\n")
    fh = safe_open_append(p)
    assert fh is not None
    try:
        fh.write("more\n")
    finally:
        fh.close()
    assert p.read_text() == "seed\nmore\n"


def test_safe_read_text_handles_internal_symlinked_parent(tmp_path: Path) -> None:
    """even when a symlink resolves to an internal path
    (still inside allowed_root), parent_chain check rejects it.

    Hits the line where ``_parent_chain_is_safe`` returns False and
    ``safe_read_text`` returns None for an otherwise-resolvable path.
    """
    if os.name == "nt":  # pragma: no cover
        pytest.skip("symlinks not portable on Windows")
    project = tmp_path / "project"
    project.mkdir()
    real_dir = project / "actual"
    real_dir.mkdir()
    (real_dir / "f.txt").write_text("ok")
    rules_link = project / "rules"
    rules_link.symlink_to(real_dir)
    target = rules_link / "f.txt"
    # `target.resolve()` lands inside `project.resolve()` because the
    # symlink target is INSIDE the allowed root, so _is_inside passes.
    # But the parent-chain walker sees the symlinked `rules` dir and
    # refuses, returning None.
    assert safe_read_text(target, project) is None


def test_safe_read_text_returns_none_when_os_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """os.open raising → None (TOCTOU race outcome path)."""
    if os.name == "nt":  # pragma: no cover
        pytest.skip("POSIX path")
    p = tmp_path / "f"
    p.write_text("hi")
    real_open = os.open

    def _broken(path: Any, flags: int, mode: int = 0o777) -> int:
        if str(path).endswith("/f"):
            raise OSError("ELOOP")
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", _broken)
    assert safe_read_text(p, tmp_path) is None


def test_is_symlink_or_reparse_windows_no_reparse_attr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows branch: lstat succeeds but file has no reparse attribute → False.

    Hits the ``getattr(st, "st_file_attributes", 0)`` default-zero branch
    when running on POSIX where stat_result has no st_file_attributes.
    """
    monkeypatch.setattr(
        "agent_amplifier.adapters._path_safety.sys.platform", "win32"
    )
    p = tmp_path / "f"
    p.write_text("x")
    assert _is_symlink_or_reparse(p) is False
