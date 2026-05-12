# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Bench branch coverage (.11/§4.3 — closes coverage gaps).

Targets:
  * Line 59 — blank-line skip in load_examples.
  * Lines 123-125 — ValueError raised by load_examples surfaces as rc=1.
  * Line 132 — default-mode (no flags) toggles do_with=True.
  * Branch 150→159 — with-amp-only path skips Delta print.
  * Lines 205-222 — matplotlib SUCCESS path renders SVG.
  * Branches 229→231, 231→233 — markdown fallback omits/keeps rows.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# load_examples — blank-line skip + unknown task
# ---------------------------------------------------------------------------


def test_load_examples_skips_blank_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSONL with blank lines must be skipped (line 59 — `continue`)."""
    from agent_amplifier import bench

    class _FakeRes:
        def read_text(self, encoding: str = "utf-8") -> str:
            return (
                '{"id":"a","problem_statement":"x"}\n'
                "\n"
                '   \n'
                '{"id":"b","problem_statement":"y"}\n'
            )

    class _FakeFiles:
        def __truediv__(self, name: str) -> _FakeRes:
            return _FakeRes()

    monkeypatch.setattr(bench, "files", lambda _pkg: _FakeFiles())
    examples = bench.load_examples("swe-bench-lite-mini")
    assert len(examples) == 2
    assert examples[0]["id"] == "a"
    assert examples[1]["id"] == "b"


def test_run_cli_unknown_task_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown task → run_cli prints error to stderr and returns 1 (lines 123-125)."""
    from agent_amplifier import bench

    args = argparse.Namespace(
        task="totally-unknown-task",
        model="sonnet",
        with_amp=False,
        without_amp=False,
        compare=False,
        export_svg=None,
    )
    rc = bench.run_cli(args)
    err = capsys.readouterr().err
    assert rc == 1
    assert "bench:" in err
    assert "unknown task" in err


# ---------------------------------------------------------------------------
# Default mode (no flags) — line 132 falls through to do_with = True
# ---------------------------------------------------------------------------


def test_run_cli_default_mode_runs_with_amp(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier import bench

    args = argparse.Namespace(
        task="swe-bench-lite-mini",
        model="sonnet",
        with_amp=False,
        without_amp=False,
        compare=False,
        export_svg=None,
    )
    rc = bench.run_cli(args)
    out = capsys.readouterr().out
    assert rc == 0
    # Default fallback: do_with becomes True → "With amplifier" line printed.
    assert "With amplifier" in out
    # ...and no Delta line when only one mode runs.
    assert "Delta:" not in out


# ---------------------------------------------------------------------------
# with-amp-only path — branch 150→159 (skip Delta block)
# ---------------------------------------------------------------------------


def test_run_cli_with_amp_only_skips_delta(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier import bench

    args = argparse.Namespace(
        task="swe-bench-lite-mini",
        model="sonnet",
        with_amp=True,
        without_amp=False,
        compare=False,
        export_svg=None,
    )
    rc = bench.run_cli(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "With amplifier" in out
    assert "Without amplifier" not in out
    assert "Delta:" not in out


def test_run_cli_without_amp_only_skips_delta(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier import bench

    args = argparse.Namespace(
        task="swe-bench-lite-mini",
        model="sonnet",
        with_amp=False,
        without_amp=True,
        compare=False,
        export_svg=None,
    )
    rc = bench.run_cli(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Without amplifier" in out
    assert "With amplifier" not in out
    assert "Delta:" not in out


def test_run_cli_compare_produces_delta(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent_amplifier import bench

    args = argparse.Namespace(
        task="swe-bench-lite-mini",
        model="sonnet",
        with_amp=False,
        without_amp=False,
        compare=True,
        export_svg=None,
    )
    rc = bench.run_cli(args)
    out = capsys.readouterr().out
    assert rc == 0
    # Delta line present and contains both signs (or zero).
    # every output line is prefixed with ``[SYNTHETIC HARNESS]``
    # so consumers cannot mistake the stub harness for a real benchmark.
    delta_line = next(
        (line for line in out.splitlines() if "Delta:" in line),
        "",
    )
    assert delta_line, out
    assert "[SYNTHETIC HARNESS]" in delta_line
    assert "%" in delta_line
    assert "pass rate" in delta_line
    assert "tokens" in delta_line


# ---------------------------------------------------------------------------
# _export_chart — matplotlib SUCCESS path (lines 205-222)
# ---------------------------------------------------------------------------


def test_export_chart_uses_matplotlib_when_available(
    tmp_path: Path,
) -> None:
    """If matplotlib is installed, _export_chart writes the requested file."""
    matplotlib = pytest.importorskip("matplotlib")

    from agent_amplifier import bench

    out = tmp_path / "chart.svg"
    bench._export_chart(
        str(out),
        with_pass=7,
        with_tokens=19000,
        without_pass=4,
        without_tokens=28000,
        n=10,
    )
    assert out.exists()
    assert out.stat().st_size > 0
    # Don't pin the exact format — just confirm matplotlib emitted something.
    _ = matplotlib  # silence unused import warning


def test_run_cli_export_svg_with_matplotlib_creates_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    from agent_amplifier import bench

    out_file = tmp_path / "out.svg"
    args = argparse.Namespace(
        task="swe-bench-lite-mini",
        model="sonnet",
        with_amp=False,
        without_amp=False,
        compare=True,
        export_svg=str(out_file),
    )
    rc = bench.run_cli(args)
    assert rc == 0
    assert out_file.exists()
    assert out_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# Markdown fallback branch matrix — 229→231, 231→233
# ---------------------------------------------------------------------------


def _force_no_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __builtins__.__import__ if isinstance(__builtins__, dict) is False else __builtins__["__import__"]  # type: ignore[index]
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Also evict any cached matplotlib so the import inside _export_chart fails.
    for mod in list(sys.modules):
        if mod == "matplotlib" or mod.startswith("matplotlib."):
            sys.modules.pop(mod, None)


def test_export_chart_markdown_fallback_with_amp_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_no_matplotlib(monkeypatch)
    from agent_amplifier import bench

    importlib.reload(bench)  # ensure no cached matplotlib

    out = tmp_path / "amp_only.svg"
    bench._export_chart(
        str(out),
        with_pass=7,
        with_tokens=19000,
        without_pass=None,
        without_tokens=None,
        n=10,
    )
    md = Path(str(out) + ".md")
    assert md.exists()
    content = md.read_text(encoding="utf-8")
    # Branch 229→231 not taken (without_pass is None) — only "with amp" row.
    assert "with amp" in content
    assert "without amp" not in content


def test_export_chart_markdown_fallback_without_amp_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_no_matplotlib(monkeypatch)
    from agent_amplifier import bench

    importlib.reload(bench)

    out = tmp_path / "noamp_only.svg"
    bench._export_chart(
        str(out),
        with_pass=None,
        with_tokens=None,
        without_pass=4,
        without_tokens=28000,
        n=10,
    )
    md = Path(str(out) + ".md")
    assert md.exists()
    content = md.read_text(encoding="utf-8")
    # Branch 231→233 not taken (with_pass is None) — only "without amp" row.
    assert "without amp" in content
    assert "with amp |" not in content  # the table cell


def test_export_chart_markdown_fallback_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_no_matplotlib(monkeypatch)
    from agent_amplifier import bench

    importlib.reload(bench)

    out = tmp_path / "both.svg"
    bench._export_chart(
        str(out),
        with_pass=7,
        with_tokens=19000,
        without_pass=4,
        without_tokens=28000,
        n=10,
    )
    md = Path(str(out) + ".md")
    assert md.exists()
    content = md.read_text(encoding="utf-8")
    # Both branches taken: 229→231 → 231→233.
    assert "with amp" in content
    assert "without amp" in content


def test_export_chart_markdown_fallback_neither_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both None — table has only the header rows."""
    _force_no_matplotlib(monkeypatch)
    from agent_amplifier import bench

    importlib.reload(bench)

    out = tmp_path / "nada.svg"
    bench._export_chart(
        str(out),
        with_pass=None,
        with_tokens=None,
        without_pass=None,
        without_tokens=None,
        n=10,
    )
    md = Path(str(out) + ".md")
    assert md.exists()
    content = md.read_text(encoding="utf-8")
    # Only header lines — no data rows.
    assert "variant" in content
    assert "with amp" not in content
    assert "without amp" not in content


# ---------------------------------------------------------------------------
# — --real fail-closed switch
# ---------------------------------------------------------------------------


def test_run_cli_real_returns_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--real`` is a fail-closed placeholder for V1.1.  Returns non-zero
    with a stderr message so wrapper scripts using ``set -e`` see the
    failure rather than believe they got real benchmark numbers."""
    import argparse

    from agent_amplifier import bench

    args = argparse.Namespace(
        task="swe-bench-lite-mini",
        model="sonnet",
        with_amp=True,
        without_amp=False,
        compare=False,
        export_svg=None,
        real=True,
    )
    rc = bench.run_cli(args)
    err = capsys.readouterr().err
    assert rc != 0
    assert "not implemented" in err
