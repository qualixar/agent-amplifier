# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for CursorAdapter ().

Coverage targets: 100 % line + 100 % branch on
``src/agent_amplifier/adapters/cursor.py``.
"""
from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from agent_amplifier.adapters.cursor import _PER_CHUNK_BYTES, CursorAdapter
from agent_amplifier.types import EffortLevel, Outcome, RecalledPattern

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cwd_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Switch CWD to an empty tmp dir."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_outcome(query: str = "test", quality: float = 0.5) -> Outcome:
    return Outcome(
        query=query,
        effort=EffortLevel.LOW,
        iterations=1,
        quality=quality,
    )


def _mdc(
    description: str = "Demo",
    always_apply: str = "false",
    globs: str = "",
    body: str = "rule body",
) -> str:
    fields = [f"description: {description}", f"alwaysApply: {always_apply}"]
    if globs:
        fields.append(f"globs: {globs}")
    return "---\n" + "\n".join(fields) + "\n---\n\n" + body + "\n"


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


def test_detect_true_with_mdc(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "x.mdc").write_text(_mdc())
    assert CursorAdapter.detect() is True


def test_detect_true_with_legacy(cwd_only: Path) -> None:
    (cwd_only / ".cursorrules").write_text("legacy single file rules")
    assert CursorAdapter.detect() is True


def test_detect_false_when_nothing(cwd_only: Path) -> None:
    assert CursorAdapter.detect() is False


def test_detect_false_when_dir_present_but_empty(cwd_only: Path) -> None:
    """``.cursor/rules/`` exists but no MDC files inside."""
    (cwd_only / ".cursor" / "rules").mkdir(parents=True)
    assert CursorAdapter.detect() is False


# ---------------------------------------------------------------------------
# default_memory_recall — MDC parsing
# ---------------------------------------------------------------------------


def test_recall_returns_recalledpattern(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "py.mdc").write_text(
        _mdc(description="Python rules", body="Use uv for python venvs.")
    )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert isinstance(res[0], RecalledPattern)
    assert "uv" in res[0].text
    assert res[0].source.startswith("cursor:")


def test_recall_parses_alwaysapply_true_into_tag(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "always.mdc").write_text(
        _mdc(description="Always-on", always_apply="true", body="invariant body")
    )
    adapter = CursorAdapter(kernel=None)
    # query that does NOT match — alwaysApply should still surface it
    res = adapter.default_memory_recall("zzz_no_match_zzz")
    assert len(res) == 1
    assert "project-rule" in res[0].tags


def test_recall_parses_globs_into_scoped_tag(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "scoped.mdc").write_text(
        _mdc(
            description="Scoped",
            always_apply="false",
            globs="src/**/*.py",
            body="contains keyword python here",
        )
    )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "scoped" in res[0].tags
    # MED-5: globs surfaced as list[str] in metadata even for the
    # single-string form.
    assert res[0].metadata.get("globs") == ["src/**/*.py"]


def test_recall_parses_list_form_globs(cwd_only: Path) -> None:
    """MED-5 — list form is normalized to ``list[str]``."""
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "scoped.mdc").write_text(
        _mdc(
            description="Scoped",
            always_apply="false",
            globs='["src/**/*.py", "tests/**/*.py"]',
            body="contains keyword python here",
        )
    )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "scoped" in res[0].tags
    assert res[0].metadata.get("globs") == [
        "src/**/*.py",
        "tests/**/*.py",
    ]


def test_recall_handles_empty_list_globs(cwd_only: Path) -> None:
    """MED-5 — empty list globs => no scoped tag, no globs metadata."""
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "rule.mdc").write_text(
        _mdc(
            description="Rule",
            always_apply="false",
            globs="[]",
            body="contains keyword python here",
        )
    )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "scoped" not in res[0].tags
    assert "globs" not in res[0].metadata


def test_parse_globs_helper_unit() -> None:
    """MED-5 — ``_parse_globs`` shape coverage (unit-level).

    Covers the four canonical forms documented in the helper docstring:
    empty, single-string, quoted single-string, list-form, empty list,
    and the malformed-tail safe-default branch.
    """
    pg = CursorAdapter._parse_globs
    assert pg("") == []
    assert pg("   ") == []
    assert pg("src/**/*.py") == ["src/**/*.py"]
    assert pg('"src/**/*.py"') == ["src/**/*.py"]
    assert pg("[]") == []
    assert pg('["a", "b"]') == ["a", "b"]
    assert pg("['a', 'b', 'c']") == ["a", "b", "c"]
    # Mixed-quote tolerance — naive splitter strips both kinds.
    assert pg('["a",   "b"  ]') == ["a", "b"]
    # Single-element list still works.
    assert pg('["only"]') == ["only"]


def test_recall_keyword_filter_excludes_non_matching(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "py.mdc").write_text(
        _mdc(description="Python", body="uv stuff")
    )
    (rules / "rust.mdc").write_text(
        _mdc(description="Rust", body="cargo stuff")
    )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("rust")
    assert len(res) == 1
    assert "cargo" in res[0].text.lower()


def test_recall_respects_limit(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    for i in range(5):
        (rules / f"r{i}.mdc").write_text(
            _mdc(description=f"R{i}", body="contains keyword python")
        )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python", limit=2)
    assert len(res) == 2


def test_recall_caps_chunk_size(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    huge = "x" * (_PER_CHUNK_BYTES * 4)
    (rules / "huge.mdc").write_text(_mdc(body=huge))
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("")
    assert res
    for r in res:
        assert len(r.text) <= _PER_CHUNK_BYTES


def test_recall_returns_empty_when_nothing(cwd_only: Path) -> None:
    adapter = CursorAdapter(kernel=None)
    assert adapter.default_memory_recall("anything") == []


def test_recall_falls_back_to_legacy_cursorrules(cwd_only: Path) -> None:
    """When no MDC files match, ``.cursorrules`` legacy is read."""
    (cwd_only / ".cursorrules").write_text("legacy: prefer black formatter")
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("black")
    assert len(res) == 1
    assert "legacy" in res[0].tags
    assert "prefer black" in res[0].text


def test_recall_does_not_use_legacy_when_mdc_hits(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "py.mdc").write_text(
        _mdc(description="P", body="contains python keyword")
    )
    (cwd_only / ".cursorrules").write_text("legacy text python here too")
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "legacy" not in res[0].tags


def test_recall_skips_unreadable_mdc(
    cwd_only: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """B3: read goes through safe_read_text, surface log either way."""
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "bad.mdc").write_text(_mdc())

    # read uses os.fdopen via os.open(O_RDONLY|O_NOFOLLOW).
    # Mock os.fstat to fail downstream of the open call.
    import os as _os
    real_fstat = _os.fstat

    def _broken_fstat(fd: int) -> _os.stat_result:
        raise OSError("boom")

    monkeypatch.setattr(_os, "fstat", _broken_fstat)
    adapter = CursorAdapter(kernel=None)
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("x")
    monkeypatch.setattr(_os, "fstat", real_fstat)
    assert res == []
    assert any(
        "refused unsafe path" in rec.message
        or "cannot read" in rec.message
        for rec in caplog.records
    )


def test_recall_handles_malformed_frontmatter_no_closing_fence(
    cwd_only: Path,
) -> None:
    """Missing closing ``---`` returns content as-is (treated as plain body)."""
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "broken.mdc").write_text(
        "---\ndescription: never closes\n# more text but no fence\nbody python here\n"
    )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "python" in res[0].text


def test_recall_handles_no_frontmatter(cwd_only: Path) -> None:
    """File without ``---`` opening is treated as plain body."""
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "plain.mdc").write_text("just markdown about python and rust")
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "python" in res[0].text


def test_recall_skips_unreadable_legacy(
    cwd_only: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (cwd_only / ".cursorrules").write_text("legacy")

    # read uses os.fdopen via os.open(O_RDONLY|O_NOFOLLOW).
    import os as _os
    real_fstat = _os.fstat

    def _broken_fstat(fd: int) -> _os.stat_result:
        raise OSError("blocked")

    monkeypatch.setattr(_os, "fstat", _broken_fstat)
    adapter = CursorAdapter(kernel=None)
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("legacy")
    monkeypatch.setattr(_os, "fstat", real_fstat)
    assert res == []


def test_recall_legacy_with_no_query_match(cwd_only: Path) -> None:
    (cwd_only / ".cursorrules").write_text("legacy stuff about widgets")
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert res == []


def test_recall_empty_query_returns_all(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "a.mdc").write_text(_mdc(description="A", body="aaa"))
    (rules / "b.mdc").write_text(_mdc(description="B", body="bbb"))
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("")
    assert len(res) == 2


def test_recall_mdc_ignores_yaml_comments(cwd_only: Path) -> None:
    """Frontmatter ``# comment`` lines should be skipped, not parsed."""
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "c.mdc").write_text(
        "---\n# this is a comment\ndescription: Real\nalwaysApply: true\n---\n\nbody python\n"
    )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "project-rule" in res[0].tags


def test_recall_quoted_values_stripped(cwd_only: Path) -> None:
    """Quoted values like ``description: "x"`` should have quotes stripped."""
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "q.mdc").write_text(
        '---\ndescription: "Python tooling"\nalwaysApply: false\n---\n\nbody\n'
    )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("python tooling")
    # description matches via haystack
    assert len(res) == 1


def test_recall_empty_body_falls_back_to_raw(cwd_only: Path) -> None:
    """When body is empty, raw text is used so we still surface SOMETHING."""
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    # Frontmatter only, no body — alwaysApply forces inclusion.
    (rules / "fm.mdc").write_text(
        "---\ndescription: only\nalwaysApply: true\n---\n"
    )
    adapter = CursorAdapter(kernel=None)
    res = adapter.default_memory_recall("zzz_no_match")
    assert len(res) == 1
    # raw text used as fallback chunk → frontmatter shows up
    assert "description" in res[0].text


# ---------------------------------------------------------------------------
# default_memory_remember
# ---------------------------------------------------------------------------


def test_remember_writes_new_mdc_when_dir_exists(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    adapter = CursorAdapter(kernel=None)
    adapter.default_memory_remember(_make_outcome("hello cursor", 0.6))
    written = list(rules.glob("agent-amplifier-*.mdc"))
    assert len(written) == 1
    body = written[0].read_text()
    assert "Amplifier note" in body
    assert "hello cursor" in body
    assert "quality=0.60" in body
    assert "alwaysApply: false" in body


def test_remember_noop_when_no_dir(cwd_only: Path) -> None:
    adapter = CursorAdapter(kernel=None)
    adapter.default_memory_remember(_make_outcome())
    # No ``.cursor/`` should have been created
    assert not (cwd_only / ".cursor").exists()


def test_remember_swallows_oserror(
    cwd_only: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """write goes through safe_open_write which logs
    ``refused unsafe write target`` when ``os.open`` fails on a read-only
    parent dir.  Either log line is acceptable evidence the write was
    swallowed without raising.
    """
    if os.name == "nt":  # pragma: no cover - Windows perms differ
        pytest.skip("chmod not portable on Windows")
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    rules.chmod(stat.S_IRUSR | stat.S_IXUSR)  # read-only dir → write fails
    adapter = CursorAdapter(kernel=None)
    try:
        with caplog.at_level(logging.WARNING):
            adapter.default_memory_remember(_make_outcome())
        assert any(
            "write to" in rec.message
            or "refused unsafe write target" in rec.message
            for rec in caplog.records
        )
    finally:
        rules.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def test_remember_truncates_long_query(cwd_only: Path) -> None:
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    adapter = CursorAdapter(kernel=None)
    adapter.default_memory_remember(_make_outcome("Q" * 500, 0.9))
    written = list(rules.glob("agent-amplifier-*.mdc"))
    body = written[0].read_text()
    assert "Q" * 100 in body
    assert "Q" * 101 not in body


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------


def test_class_metadata() -> None:
    assert CursorAdapter.framework_name == "cursor"
    assert CursorAdapter.HOST_NAME == "cursor"


# ---------------------------------------------------------------------------
# bool parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        ("true", True),
        ("True", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("0", False),
        ("", False),
        ("nope", False),
    ],
)
def test_parse_bool(val: str, expected: bool) -> None:
    assert CursorAdapter._parse_bool(val) is expected


# ---------------------------------------------------------------------------
# B3 / B4 — symlink defense (SEC-03 / SEC-04)
# ---------------------------------------------------------------------------


def test_cursor_refuses_symlink_escape(
    cwd_only: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SEC-03: a .mdc symlink pointing outside .cursor MUST be refused."""
    if os.name == "nt":  # pragma: no cover - symlinks need admin on Windows
        pytest.skip("symlink defense exercised on POSIX runner")
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    attacker_target = tmp_path.parent / "attacker-rules.mdc"
    attacker_target.write_text(_mdc(body="ATTACKER PAYLOAD"))
    (rules / "poisoned.mdc").symlink_to(attacker_target)
    adapter = CursorAdapter(kernel=None)
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("ATTACKER")
    assert res == []
    assert any(
        "refused unsafe path" in rec.message for rec in caplog.records
    )


def test_cursor_legacy_refuses_symlink_escape(
    cwd_only: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SEC-03: legacy .cursorrules symlink to outside CWD must be refused."""
    if os.name == "nt":  # pragma: no cover - symlinks need admin on Windows
        pytest.skip("symlink defense exercised on POSIX runner")
    attacker_target = tmp_path.parent / "attacker-cursorrules.txt"
    attacker_target.write_text("ATTACKER PAYLOAD legacy")
    (cwd_only / ".cursorrules").symlink_to(attacker_target)
    adapter = CursorAdapter(kernel=None)
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("ATTACKER")
    assert res == []
    assert any(
        "refused unsafe path" in rec.message for rec in caplog.records
    )


def test_cursor_remember_refuses_symlink(
    cwd_only: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SEC-04: pre-staged symlink at the write target must NOT be followed."""
    if os.name == "nt":  # pragma: no cover - symlinks need admin on Windows
        pytest.skip("O_NOFOLLOW is POSIX")
    rules = cwd_only / ".cursor" / "rules"
    rules.mkdir(parents=True)
    attacker_target = tmp_path.parent / "ssh-authorized-keys-cursor"
    attacker_target.write_text("# original\n")
    today = __import__("datetime").date.today().isoformat()
    (rules / f"agent-amplifier-{today}.mdc").symlink_to(attacker_target)
    adapter = CursorAdapter(kernel=None)
    with caplog.at_level(logging.WARNING):
        adapter.default_memory_remember(_make_outcome("hello", 0.5))
    # Attacker target untouched
    assert attacker_target.read_text() == "# original\n"
    assert any(
        "refused symlink target" in rec.message
        or "refused unsafe write target" in rec.message
        or "write to" in rec.message
        for rec in caplog.records
    )
