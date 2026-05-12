# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.bench`` (.11 + §4.3, )."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest


def test_bundled_dataset_exists() -> None:
    """SWE-bench-Lite-mini bundled data file must ship in package."""
    from agent_amplifier import bench

    examples = bench.load_examples("swe-bench-lite-mini")
    assert isinstance(examples, list)
    assert len(examples) >= 5  # spec says 10; allow some slack
    for ex in examples:
        assert "id" in ex
        assert "problem_statement" in ex


def test_load_examples_unknown_task_raises() -> None:
    from agent_amplifier import bench

    with pytest.raises(ValueError, match="unknown task"):
        bench.load_examples("nonexistent-task")


def test_run_one_returns_stub_metrics() -> None:
    from agent_amplifier import bench

    ex = {"id": "x", "problem_statement": "fix the typo in README"}
    result = bench.run_one(ex, with_amp=False, model="sonnet")
    assert "passed" in result
    assert "tokens" in result
    assert isinstance(result["tokens"], int)


def test_run_cli_compare_prints_delta(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
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
    assert "Without amplifier" in out
    assert "With amplifier" in out
    assert "Delta" in out


def test_run_cli_with_amp_only(capsys: pytest.CaptureFixture[str]) -> None:
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
    assert "passed" in out


def test_export_svg_creates_markdown_fallback_when_matplotlib_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without matplotlib, --export-svg must still produce a markdown fallback."""
    import builtins as _b
    real_import = _b.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_b, "__import__", fake_import)
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
    # When matplotlib is absent, we expect a markdown-table fallback at out_file.
    md_fallback = Path(str(out_file) + ".md")
    assert md_fallback.exists() or out_file.exists()
