# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for GitHubCopilotAdapter ().

Coverage targets: 100 % line + 100 % branch on
``src/agent_amplifier/adapters/github_copilot.py``.
"""
from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from agent_amplifier.adapters.github_copilot import (
    _PER_CHUNK_BYTES,
    GitHubCopilotAdapter,
)
from agent_amplifier.types import EffortLevel, Outcome, RecalledPattern

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cwd_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_outcome(query: str = "test", quality: float = 0.5) -> Outcome:
    return Outcome(
        query=query,
        effort=EffortLevel.LOW,
        iterations=1,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


def test_detect_true_with_repo_instructions(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text("# Repo instructions")
    assert GitHubCopilotAdapter.detect() is True


def test_detect_true_with_scoped_instructions(cwd_only: Path) -> None:
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    (scoped / "py.instructions.md").write_text("---\napplyTo: '**/*.py'\n---\nbody")
    assert GitHubCopilotAdapter.detect() is True


def test_detect_false_when_nothing(cwd_only: Path) -> None:
    assert GitHubCopilotAdapter.detect() is False


def test_detect_false_when_scoped_dir_empty(cwd_only: Path) -> None:
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    assert GitHubCopilotAdapter.detect() is False


# ---------------------------------------------------------------------------
# default_memory_recall
# ---------------------------------------------------------------------------


def test_recall_returns_recalledpattern_with_source(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text(
        "# Top\n\n## Python\n\nUse uv.\n\n## Rust\n\ncargo here.\n"
    )
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert isinstance(res[0], RecalledPattern)
    assert "uv" in res[0].text.lower()
    assert res[0].source.startswith("github-copilot:")
    assert "copilot-instructions.md" in res[0].source


def test_recall_returns_empty_when_no_files(cwd_only: Path) -> None:
    adapter = GitHubCopilotAdapter(kernel=None)
    assert adapter.default_memory_recall("anything") == []


def test_recall_keyword_ranks(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text(
        "## Python\n\nuv\n\n## Rust\n\ncargo\n"
    )
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("rust")
    assert len(res) == 1
    assert "cargo" in res[0].text.lower()


def test_recall_respects_limit(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    body = "\n\n".join(
        f"## Section {i}\n\nthe word python here\n" for i in range(10)
    )
    (gh_dir / "copilot-instructions.md").write_text("Prologue.\n\n" + body)
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("python", limit=2)
    assert len(res) == 2


def test_recall_caps_chunk_size(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    huge = "## Big\n\n" + ("x" * (_PER_CHUNK_BYTES * 4))
    (gh_dir / "copilot-instructions.md").write_text(huge)
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("")
    assert res
    for r in res:
        assert len(r.text) <= _PER_CHUNK_BYTES


def test_recall_walks_repo_and_scoped(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text(
        "## Repo\n\nrepo level python content\n"
    )
    scoped = gh_dir / "instructions"
    scoped.mkdir()
    (scoped / "py.instructions.md").write_text(
        "## Scoped\n\nscoped python rules\n"
    )
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 2
    sources = {r.source for r in res}
    tags = [r.tags for r in res]
    assert any("copilot-instructions.md" in s for s in sources)
    assert any("py.instructions.md" in s for s in sources)
    assert any("scoped" in t for t in tags)


def test_recall_skips_unreadable(
    cwd_only: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """B3: read goes through safe_read_text, surface log either way."""
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text("## a\n")

    # read uses os.fdopen via os.open(O_RDONLY|O_NOFOLLOW).
    import os as _os
    real_fstat = _os.fstat

    def _broken_fstat(fd: int) -> _os.stat_result:
        raise OSError("nope")

    monkeypatch.setattr(_os, "fstat", _broken_fstat)
    adapter = GitHubCopilotAdapter(kernel=None)
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("a")
    monkeypatch.setattr(_os, "fstat", real_fstat)
    assert res == []
    assert any(
        "refused unsafe path" in rec.message
        or "cannot read" in rec.message
        for rec in caplog.records
    )


def test_recall_skips_unicode_decode_error(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_bytes(b"\xff\xfe\x00\x00x")
    adapter = GitHubCopilotAdapter(kernel=None)
    assert adapter.default_memory_recall("anything") == []


def test_recall_handles_empty_file(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text("")
    adapter = GitHubCopilotAdapter(kernel=None)
    assert adapter.default_memory_recall("x") == []


def test_recall_no_h2_returns_full_body(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text(
        "Plain text mentioning python without headings."
    )
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1


def test_recall_empty_query_returns_all(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text("## A\n\nfoo\n\n## B\n\nbar\n")
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("")
    assert len(res) == 2


def test_recall_limit_cuts_off_at_repo_level(cwd_only: Path) -> None:
    """Sufficient hits in repo file to reach limit → scoped never read."""
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text(
        "## A\n\npython\n\n## B\n\npython\n\n## C\n\npython\n"
    )
    scoped = gh_dir / "instructions"
    scoped.mkdir()
    (scoped / "x.instructions.md").write_text("## D\n\npython\n")
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("python", limit=2)
    assert len(res) == 2
    for r in res:
        assert "instructions/" not in r.source


def test_recall_scoped_only_when_repo_missing(cwd_only: Path) -> None:
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    (scoped / "py.instructions.md").write_text("## Py\n\npython rules\n")
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("python")
    assert len(res) == 1
    assert "scoped" in res[0].tags


def test_recall_scoped_loop_hits_limit_early(cwd_only: Path) -> None:
    """Repo file gives 1 hit; scoped files give many — limit hit inside
    scoped loop, exercising the inner-loop early ``return`` branch."""
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    (gh_dir / "copilot-instructions.md").write_text(
        "## Repo\n\npython repo line\n"
    )
    scoped = gh_dir / "instructions"
    scoped.mkdir()
    (scoped / "a.instructions.md").write_text(
        "## A1\n\npython aaa\n\n## A2\n\npython bbb\n\n## A3\n\npython ccc\n"
    )
    (scoped / "b.instructions.md").write_text(
        "## B1\n\npython ddd\n\n## B2\n\npython eee\n"
    )
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("python", limit=3)
    assert len(res) == 3
    # First hit comes from repo file; subsequent two come from scoped a.*
    assert res[0].source.endswith("copilot-instructions.md")
    assert any("a.instructions.md" in r.source for r in res[1:])


# ---------------------------------------------------------------------------
# default_memory_remember
# ---------------------------------------------------------------------------


def test_remember_appends_when_file_exists(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    target = gh_dir / "copilot-instructions.md"
    target.write_text("# original\n")
    adapter = GitHubCopilotAdapter(kernel=None)
    adapter.default_memory_remember(_make_outcome("hello copilot", 0.7))
    body = target.read_text()
    assert "# original" in body
    assert "Amplifier note" in body
    assert "hello copilot" in body
    assert "quality=0.70" in body


def test_remember_noop_when_no_file(cwd_only: Path) -> None:
    adapter = GitHubCopilotAdapter(kernel=None)
    adapter.default_memory_remember(_make_outcome())
    assert not (cwd_only / ".github").exists()


def test_remember_swallows_oserror(
    cwd_only: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """B4: append goes through safe_open_append; either log line ok."""
    if os.name == "nt":  # pragma: no cover - Windows perms differ
        pytest.skip("chmod not portable on Windows")
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    target = gh_dir / "copilot-instructions.md"
    target.write_text("# locked\n")
    target.chmod(stat.S_IRUSR)
    adapter = GitHubCopilotAdapter(kernel=None)
    try:
        with caplog.at_level(logging.WARNING):
            adapter.default_memory_remember(_make_outcome())
        assert any(
            "append to" in rec.message
            or "refused unsafe append target" in rec.message
            for rec in caplog.records
        )
    finally:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_remember_truncates_long_query(cwd_only: Path) -> None:
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    target = gh_dir / "copilot-instructions.md"
    target.write_text("seed\n")
    adapter = GitHubCopilotAdapter(kernel=None)
    adapter.default_memory_remember(_make_outcome("Q" * 500, 0.9))
    body = target.read_text()
    assert "Q" * 100 in body
    assert "Q" * 101 not in body


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------


def test_class_metadata() -> None:
    assert GitHubCopilotAdapter.framework_name == "github_copilot"
    assert GitHubCopilotAdapter.HOST_NAME == "github-copilot"


# ---------------------------------------------------------------------------
# B3 / B4 — symlink defense (SEC-03 / SEC-04)
# ---------------------------------------------------------------------------


def test_remember_logs_oserror_during_write(
    cwd_only: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """B4: OSError raised by ``fh.write`` after a successful open
    is logged as ``append to ... failed`` and not propagated.
    """
    if os.name == "nt":  # pragma: no cover - POSIX-specific test
        pytest.skip("safe_open_append POSIX path")
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    target = gh_dir / "copilot-instructions.md"
    target.write_text("# original\n")

    real_fdopen = os.fdopen

    class _BadHandle:
        def __init__(self, real: Any) -> None:
            self._real = real

        def write(self, data: str) -> int:
            raise OSError("disk full")

        def close(self) -> None:
            self._real.close()

    def _broken_fdopen(fd: int, *a: Any, **kw: Any) -> Any:
        real = real_fdopen(fd, *a, **kw)
        return _BadHandle(real)

    monkeypatch.setattr(os, "fdopen", _broken_fdopen)
    adapter = GitHubCopilotAdapter(kernel=None)
    with caplog.at_level(logging.WARNING):
        adapter.default_memory_remember(_make_outcome())
    assert any("append to" in rec.message for rec in caplog.records)


def test_github_copilot_refuses_symlink_escape(
    cwd_only: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SEC-03: copilot-instructions symlink outside .github MUST be refused."""
    if os.name == "nt":  # pragma: no cover - symlinks need admin on Windows
        pytest.skip("symlink defense exercised on POSIX runner")
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    attacker_target = tmp_path.parent / "attacker-copilot.md"
    attacker_target.write_text("ATTACKER PAYLOAD")
    (gh_dir / "copilot-instructions.md").symlink_to(attacker_target)
    adapter = GitHubCopilotAdapter(kernel=None)
    with caplog.at_level(logging.WARNING):
        res = adapter.default_memory_recall("ATTACKER")
    assert res == []
    assert any(
        "refused unsafe path" in rec.message for rec in caplog.records
    )


def test_github_copilot_remember_refuses_symlink(
    cwd_only: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SEC-04: append target swapped to symlink must NOT redirect writes."""
    if os.name == "nt":  # pragma: no cover - symlinks need admin on Windows
        pytest.skip("O_NOFOLLOW is POSIX")
    gh_dir = cwd_only / ".github"
    gh_dir.mkdir()
    attacker_target = tmp_path.parent / "ssh-authorized-keys-gh"
    attacker_target.write_text("# original\n")
    target = gh_dir / "copilot-instructions.md"
    target.symlink_to(attacker_target)
    adapter = GitHubCopilotAdapter(kernel=None)
    with caplog.at_level(logging.WARNING):
        adapter.default_memory_remember(_make_outcome("attempt", 0.9))
    assert attacker_target.read_text() == "# original\n"
    assert any(
        "refused unsafe append target" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# MED-6: applyTo: frontmatter parsing
# ---------------------------------------------------------------------------


def test_recall_scoped_applyto_frontmatter_surfaces_in_metadata(
    cwd_only: Path,
) -> None:
    """MED-6.

    A scoped ``*.instructions.md`` with ``applyTo:`` frontmatter MUST
    surface that glob via ``RecalledPattern.metadata["apply_to"]``.
    """
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    (scoped / "typescript.instructions.md").write_text(
        '---\napplyTo: "**/*.ts"\n---\n\n## TypeScript rules\n\n'
        "always use TypeScript strict mode\n"
    )
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("typescript")
    assert len(res) >= 1
    # All scoped recalls must carry the apply_to metadata.
    scoped_hits = [r for r in res if "scoped" in r.tags]
    assert scoped_hits, "expected at least one scoped recall"
    assert all(r.metadata.get("apply_to") == "**/*.ts" for r in scoped_hits)


def test_recall_scoped_no_frontmatter_omits_apply_to(cwd_only: Path) -> None:
    """A scoped file without frontmatter must NOT carry apply_to metadata."""
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    (scoped / "plain.instructions.md").write_text(
        "## Plain\n\nplain typescript content\n"
    )
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("typescript")
    assert len(res) >= 1
    scoped_hits = [r for r in res if "scoped" in r.tags]
    assert scoped_hits
    assert all("apply_to" not in r.metadata for r in scoped_hits)


def test_read_with_frontmatter_no_fence(cwd_only: Path) -> None:
    """Files without ``---`` fence return ``({}, full_text)``."""
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    f = scoped / "plain.instructions.md"
    f.write_text("plain content\n")
    meta, body = GitHubCopilotAdapter._read_with_frontmatter(
        f, cwd_only / ".github"
    )
    assert meta == {}
    assert body == "plain content\n"


def test_read_with_frontmatter_malformed_no_closing_fence(
    cwd_only: Path,
) -> None:
    """Frontmatter without closing fence falls back to plain body."""
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    f = scoped / "broken.instructions.md"
    f.write_text("---\napplyTo: '**/*.py'\nbody never closed\n")
    meta, body = GitHubCopilotAdapter._read_with_frontmatter(
        f, cwd_only / ".github"
    )
    # No closing fence => entire text returned as body.
    assert meta == {}
    assert body is not None
    assert "applyTo" in body  # frontmatter surfaced as raw body content


def test_read_with_frontmatter_handles_unsafe_path(
    cwd_only: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3 + 9.D MED-6: unsafe-path returns ``(meta={}, body=None)``."""
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    f = scoped / "x.instructions.md"
    f.write_text("---\napplyTo: '**/*.py'\n---\nbody\n")

    # Monkey-patch safe_read_text to simulate the unsafe-path return.
    import agent_amplifier.adapters.github_copilot as gh_mod

    def _refuse(*_a: object, **_kw: object) -> Any:
        return None

    monkeypatch.setattr(gh_mod, "safe_read_text", _refuse)
    meta, body = GitHubCopilotAdapter._read_with_frontmatter(
        f, cwd_only / ".github"
    )
    assert meta == {}
    assert body is None


def test_read_with_frontmatter_empty_file(cwd_only: Path) -> None:
    """Empty file => ``({}, '')``."""
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    f = scoped / "empty.instructions.md"
    f.write_text("")
    meta, body = GitHubCopilotAdapter._read_with_frontmatter(
        f, cwd_only / ".github"
    )
    assert meta == {}
    assert body == ""


def test_read_with_frontmatter_handles_comments_and_blank_lines(
    cwd_only: Path,
) -> None:
    """Frontmatter parser tolerates comments + blank lines in the YAML head."""
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    f = scoped / "comments.instructions.md"
    f.write_text(
        "---\n"
        "# this is a comment\n"
        "\n"
        "applyTo: '**/*.go'\n"
        "no_colon_line_skipped\n"
        "---\n"
        "\n"
        "## Body\n"
        "go content\n"
    )
    meta, body = GitHubCopilotAdapter._read_with_frontmatter(
        f, cwd_only / ".github"
    )
    assert meta.get("applyTo") == "**/*.go"
    assert body is not None
    assert "## Body" in body


def test_parse_apply_to_helper_unit() -> None:
    """MED-6 — single-string normalization shape."""
    fn = GitHubCopilotAdapter._parse_apply_to
    assert fn("") == ""
    assert fn("   ") == ""
    assert fn("**/*.ts") == "**/*.ts"
    assert fn('"**/*.ts"') == "**/*.ts"
    assert fn("'**/*.ts'") == "**/*.ts"
    assert fn("  **/*.ts  ") == "**/*.ts"


def test_recall_scoped_skips_unsafe_path(
    cwd_only: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MED-6 — scoped path that returns None body MUST be skipped.

    Covers the ``if body_text is None: continue`` branch in the scoped
    recall loop after MED-6 added frontmatter pre-parsing.
    """
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    (scoped / "x.instructions.md").write_text("## A\n\ntypescript\n")
    (scoped / "y.instructions.md").write_text("## B\n\ntypescript\n")

    import agent_amplifier.adapters.github_copilot as gh_mod

    real_safe_read = gh_mod.safe_read_text

    def _refuse_first(path: Path, allowed_root: Path) -> Any:
        # Simulate first scoped file refused (symlink-out-of-tree); the
        # second one reads normally.
        if "x.instructions.md" in str(path):
            return None
        return real_safe_read(path, allowed_root)

    monkeypatch.setattr(gh_mod, "safe_read_text", _refuse_first)
    adapter = GitHubCopilotAdapter(kernel=None)
    res = adapter.default_memory_recall("typescript")
    # Only the second file's chunk should surface.
    assert len(res) >= 1
    assert all("y.instructions.md" in r.source for r in res)


def test_rank_chunks_returns_empty_for_empty_text() -> None:
    """``_rank_chunks`` MUST return ``[]`` for empty input (defensive)."""
    assert GitHubCopilotAdapter._rank_chunks("", "anything") == []
    assert GitHubCopilotAdapter._rank_chunks("", "") == []


def test_read_with_frontmatter_no_leading_blank_line(cwd_only: Path) -> None:
    """Cover the ``body.startswith('\\n')`` False branch in _read_with_frontmatter.

    When the post-fence body does NOT start with a newline (e.g. another
    fence sits flush against content), the leading-blank-line strip is a
    no-op. This branch was added in MED-6.
    """
    scoped = cwd_only / ".github" / "instructions"
    scoped.mkdir(parents=True)
    f = scoped / "tight.instructions.md"
    # No blank line between closing fence and body.
    f.write_text("---\napplyTo: '**/*.py'\n---\nimmediate body\n")
    meta, body = GitHubCopilotAdapter._read_with_frontmatter(
        f, cwd_only / ".github"
    )
    assert meta.get("applyTo") == "**/*.py"
    assert body is not None
    assert body.startswith("immediate body")
