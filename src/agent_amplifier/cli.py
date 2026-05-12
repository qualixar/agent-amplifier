# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""``agent-amp`` CLI entry point (, , G-4).

Subcommand surface (must mirror PHASE-5-RELEASE.md V1.1 verbatim):

    agent-amp install <target>     attach amplifier to a host framework
    agent-amp install --auto       auto-detect & install all available
    agent-amp uninstall <target>   detach amplifier from a host framework
    agent-amp list                 list supported adapters + detection status
    agent-amp status               installed-only summary
    agent-amp doctor               print environment diagnostics
    agent-amp config show|set|path manage TOML config
    agent-amp bench [...]          run amplifier benchmarks

Exit codes (.1):
    0 success
    1 generic error
    2 unknown command or target
    3 already installed
    4 permission denied
    5 host framework not detected
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from agent_amplifier import __version__
from agent_amplifier.adapter_base import AdapterBase
from agent_amplifier.config import (
    LEGACY_USER_CONFIG_PATH,
    USER_CONFIG_PATH,
    load_config,
)

LOG = logging.getLogger("agent_amplifier.cli")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-amp",
        description=(
            "agent-amp - runtime amplification for AI agents.\n\n"
            "USAGE:\n"
            "  agent-amp <COMMAND> [OPTIONS]\n\n"
            "COMMANDS:\n"
            "  install <target>       Attach amplifier to a host framework "
            "(e.g. claude-code, langchain).\n"
            "  uninstall <target>     Detach amplifier from a host framework.\n"
            "  list                   List supported adapters and their detection status.\n"
            "  status                 Show current installation status across all adapters.\n"
            "  doctor                 Diagnose environment (SLM, Python, OS, anyio).\n"
            "  config                 Inspect or edit ~/.config/agent-amplifier/config.toml.\n"
            "  bench                  Run amplifier benchmarks against a task.\n\n"
            "  dashboard              Launch the local dashboard backend.\n\n"
            "EXIT CODES:\n"
            "  0 success\n"
            "  1 generic error\n"
            "  2 unknown command or target\n"
            "  3 already installed\n"
            "  4 permission denied\n"
            "  5 host framework not detected\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list adapters")
    p_status = sub.add_parser("status", help="installed adapters")
    p_status.add_argument(
        "--watch",
        action="store_true",
        help="live token usage bar (refreshes every 2s)",
    )
    p_status.add_argument(
        "--once",
        action="store_true",
        help="print one snapshot and exit (for scripting/testing)",
    )
    sub.add_parser("doctor", help="environment diagnostics")

    p_install = sub.add_parser("install", help="install adapter")
    p_install.add_argument("target", nargs="?")
    p_install.add_argument("--auto", action="store_true")

    p_uninstall = sub.add_parser("uninstall", help="uninstall adapter")
    p_uninstall.add_argument("target")

    p_config = sub.add_parser("config", help="inspect/edit config")
    p_config.add_argument("op", choices=["show", "set", "path"])
    p_config.add_argument("kv", nargs="?")

    p_report = sub.add_parser(
        "report", help="show amplification dashboard from state.db"
    )
    p_report.add_argument(
        "--last",
        type=int,
        default=10,
        metavar="N",
        help="show last N turns (default 10)",
    )

    sub.add_parser("dashboard", help="launch dashboard backend")

    p_bench = sub.add_parser("bench", help="benchmark")
    p_bench.add_argument("--task", default="swe-bench-lite-mini")
    p_bench.add_argument("--model", default="sonnet")
    p_bench.add_argument("--with-amp", action="store_true")
    p_bench.add_argument("--without-amp", action="store_true")
    p_bench.add_argument("--compare", action="store_true")
    p_bench.add_argument("--export-svg", default=None)
    # H-6: single-prompt mode (launch GIF). Triggered by ``--prompt PROMPT``;
    # other dataset flags are ignored when --prompt is set.
    p_bench.add_argument(
        "--prompt",
        default=None,
        metavar="PROMPT",
        help="single-prompt demo mode; pairs with --baseline / --vs-amplified",
    )
    p_bench.add_argument(
        "--baseline",
        action="store_true",
        help="(single-prompt mode) show the unamplified prompt",
    )
    p_bench.add_argument(
        "--vs-amplified",
        action="store_true",
        help="(single-prompt mode) show the amplified envelope",
    )
    # ``--real`` is a fail-closed placeholder for a
    # V1.1 real-LLM harness.  V1 only supports the synthetic harness.
    p_bench.add_argument(
        "--real",
        action="store_true",
        help="run against a real LLM (not implemented in V1; reserved for V1.1)",
    )

    p_demo = sub.add_parser(
        "demo",
        help="amplification preview for a single prompt (alias for bench --prompt)",
    )
    p_demo.add_argument("prompt", help="the prompt to amplify")

    # Persona subcommands: list / show / add / remove
    p_persona = sub.add_parser(
        "persona",
        help="manage audit personas (built-in + user-defined)",
    )
    p_persona_sub = p_persona.add_subparsers(dest="persona_cmd")
    p_persona_sub.add_parser("list", help="list built-in + custom personas")
    p_persona_show = p_persona_sub.add_parser("show", help="show a persona")
    p_persona_show.add_argument("slug")
    p_persona_add = p_persona_sub.add_parser("add", help="add a custom persona")
    p_persona_add.add_argument("--name", required=True)
    p_persona_add.add_argument("--label", required=True)
    p_persona_add.add_argument("--description", required=True)
    p_persona_add.add_argument(
        "--review-focus",
        default="",
        help="comma-separated focus axes (e.g. 'security,perf')",
    )
    p_persona_remove = p_persona_sub.add_parser(
        "remove", help="remove a custom persona (built-ins cannot be removed)"
    )
    p_persona_remove.add_argument("--name", required=True)

    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    level = (
        logging.DEBUG
        if args.verbose
        else logging.WARNING
        if args.quiet
        else logging.INFO
    )
    logging.basicConfig(level=level, format="%(message)s")

    if args.cmd == "list":
        return _cmd_list()
    if args.cmd == "status":
        if getattr(args, "watch", False) or getattr(args, "once", False):
            return _cmd_status_watch(once=getattr(args, "once", False))
        return _cmd_status()
    if args.cmd == "doctor":
        return _cmd_doctor()
    if args.cmd == "install":
        return _cmd_install(args)
    if args.cmd == "uninstall":
        return _cmd_uninstall(args)
    if args.cmd == "config":
        return _cmd_config(args)
    if args.cmd == "bench":
        return _cmd_bench(args)
    if args.cmd == "report":
        return _cmd_report(args)
    if args.cmd == "dashboard":
        return _cmd_dashboard()
    if args.cmd == "demo":
        return _cmd_demo(args)
    if args.cmd == "persona":
        return _cmd_persona(args, parser)
    parser.print_help()
    return 0


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _instantiate(cls: Any) -> Any:
    """Construct an adapter instance for CLI ops.

    Typed as ``Any`` to avoid the ``type-abstract`` mypy noise: the CLI
    instantiates concrete subclasses found via ``__subclasses__``, which
    mypy cannot prove are non-abstract from the static graph.
    """
    return cls(kernel=None)


def _enumerate_adapters() -> list[Any]:
    """Return all importable AdapterBase subclasses (currently bundled = 0).

    V1 adapters are not yet bundled (Phase 1+ will populate this list). The
    enumeration uses ``__subclasses__()`` so plug-in adapters (third party)
    appear automatically once they import AdapterBase. Return type is
    ``list[Any]`` because the strict mypy "type-abstract" check forbids
    treating a ``type[AdapterBase]`` as concrete; CLI code only uses these
    via the ``_instantiate`` helper above, which is also ``Any``-typed.
    """

    def _walk(cls: Any) -> list[Any]:
        out: list[Any] = []
        for sub in cls.__subclasses__():
            out.append(sub)
            out.extend(_walk(sub))
        return out

    return _walk(AdapterBase)


def _cmd_list() -> int:
    adapters = _enumerate_adapters()
    if not adapters:
        print("(no bundled adapters in V1.0 — Phase 1 ships claude-code/langchain)")
        return 0
    for a in adapters:
        try:
            detected = a.detect()
        except Exception:
            detected = False
        print(f"{a.framework_name:24s} v{a.version:8s} detected={detected}")
    return 0


def _cmd_status() -> int:
    adapters = _enumerate_adapters()
    if not adapters:
        print("(no adapters bundled in V1.0)")
        return 0
    for a in adapters:
        try:
            inst = _instantiate(a).is_installed()
        except Exception:
            inst = False
        if inst:
            print(f"{a.framework_name:24s} INSTALLED")
    return 0


def _render_token_bar(*, used: int, limit: int, width: int = 30) -> str:
    """Render a single-line token usage bar for --watch output."""
    if limit <= 0:
        return f"Tokens: {used:,} (no limit set)"
    pct = used / limit
    filled = min(int(pct * width), width)
    bar = "#" * filled + "-" * (width - filled)
    label = "OVER!" if used > limit else f"{pct:.0%}"
    return f"Tokens: [{bar}] {used:,} / {limit:,} ({label})"


def _cmd_status_watch(*, once: bool = False) -> int:
    """Live token usage bar from state.db."""
    import contextlib
    import sqlite3

    from agent_amplifier.adapters.claude_code import state as _state

    db_path = Path(_state._DEFAULT_STATE_DIR) / _state._STATE_DB_FILENAME
    if not db_path.exists():
        print("No state.db found. Run 'agent-amp install claude-code' first.")
        return 1

    def _read_tokens() -> tuple[int, int]:
        with contextlib.closing(
            sqlite3.connect(str(db_path), timeout=2.0)
        ) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens_used), 0) FROM outcomes"
            ).fetchone()
            total_used = int(row[0]) if row else 0
        # Default soft cap for the --watch bar. The kernel's actual budget
        # varies per effort tier; this constant is a visual reference point,
        # not a hard limit. Override via AGENT_AMP_WATCH_BUDGET env var.
        budget_env = os.environ.get("AGENT_AMP_WATCH_BUDGET", "")
        try:
            soft_cap = int(budget_env) if budget_env.strip() else 250_000
        except ValueError:
            soft_cap = 250_000
        return total_used, soft_cap

    if once:
        used, limit = _read_tokens()
        print(_render_token_bar(used=used, limit=limit))
        return 0

    import time as _time  # pragma: no cover

    try:  # pragma: no cover
        while True:
            used, limit = _read_tokens()
            line = _render_token_bar(used=used, limit=limit)
            print(f"\r{line}", end="", flush=True)
            _time.sleep(2)
    except KeyboardInterrupt:  # pragma: no cover
        print()
    return 0  # pragma: no cover


def _cmd_doctor() -> int:
    """Print environment diagnostics + per-adapter detection status.

    MED-2 (QA-M01): post V2.1 universal-memory pivot, the
    relevant signal is which host adapters detect a memory source on this
    machine — not whether SLM is installed. We enumerate the bundled
    adapters and report each one's ``detect()`` status. The SLM binary
    check is preserved as one line under "third-party providers" so
    existing users still see the wiring hint.
    """
    print(f"agent-amp {__version__}")
    print(f"Python   {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"OS       {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"anyio    {_anyio_version()}")
    print(f"config   primary={USER_CONFIG_PATH}")
    print(f"         legacy={LEGACY_USER_CONFIG_PATH}")
    print("adapters:")
    # Import inline to avoid module-top adapters import (the package
    # __init__ already pulls them, but the linear-import order here keeps
    # the CLI surface explicit + readable).
    from agent_amplifier.adapters import (
        AgentScopeAdapter,
        ClaudeCodeAdapter,
        CrewAIAdapter,
        CursorAdapter,
        GitHubCopilotAdapter,
        LangGraphAdapter,
    )

    # Tuple of (display name, adapter class) so the doctor output is stable
    # and predictable across runs. Order matches the README adapter table.
    bundled: tuple[tuple[str, type[AdapterBase]], ...] = (
        ("Claude Code", ClaudeCodeAdapter),
        ("Cursor", CursorAdapter),
        ("GitHub Copilot", GitHubCopilotAdapter),
        ("LangGraph", LangGraphAdapter),
        ("CrewAI", CrewAIAdapter),
        ("AgentScope", AgentScopeAdapter),
    )
    for label, cls in bundled:
        try:
            detected = cls.detect()
        except Exception:  # pragma: no cover - defensive: detect MUST NOT raise
            detected = False
        status = "DETECTED" if detected else "missing"
        print(f"  {label:18s} ({cls.framework_name:14s}) {status}")
    # Optional third-party memory providers — preserved for back-compat
    # so users wiring SLM still see the hint. NOT a headline.
    print("third-party memory providers (optional):")
    slm_path = shutil.which("slm")
    if slm_path:
        print(f"  slm              {slm_path}")
    else:
        print(
            "  slm              not installed "
            "(pip install superlocalmemory && slm init)"
        )
    return 0


def _install_message(adapter_cls: type[AdapterBase]) -> str:
    """print an honest message based on whether the
    adapter actually persists state, or is just a process-local marker.

    File-based adapters (Claude Code, Cursor, GitHub Copilot) read host
    files on demand — there is no real "install" step.  Saying
    ``installed`` to the user implies persistent registration.  We
    instead say ``ready`` with the file paths the adapter will read.
    """
    persistent = getattr(adapter_cls, "INSTALL_PERSISTENT", False)
    name = adapter_cls.framework_name
    if persistent:
        return f"installed: {name}"
    return (
        f"ready: {name} (no persistent install — instantiate "
        f"{adapter_cls.__name__} in code; the adapter reads host files on demand)"
    )


def _cmd_install(args: argparse.Namespace) -> int:
    adapters = _enumerate_adapters()
    if args.auto:
        if not adapters:
            print("No bundled adapters in V1.0; nothing to install.")
            return 0
        installed_count = 0
        for a in adapters:
            try:
                if a.detect():
                    _instantiate(a).install()
                    installed_count += 1
                    print(_install_message(a))
            except Exception as e:
                print(f"FAILED: {a.framework_name}: {e}", file=sys.stderr)
        print(f"installed {installed_count} adapter(s)")
        return 0

    if not args.target:
        print("install requires <target> or --auto", file=sys.stderr)
        return 2

    # Accept conventional package-style hyphens (claude-code, github-copilot)
    # as aliases for the snake_case framework_name. Users typing the README
    # one-liner verbatim must not hit ``unknown target``.
    target_normalized = args.target.replace("-", "_")
    for a in adapters:
        if a.framework_name == target_normalized:
            try:
                _instantiate(a).install()
                print(_install_message(a))
                return 0
            except PermissionError as e:
                print(f"permission denied: {e}", file=sys.stderr)
                return 4
            except Exception as e:
                msg = str(e).lower()
                if "already" in msg:
                    print(f"already installed: {a.framework_name}", file=sys.stderr)
                    return 3
                print(f"install failed: {e}", file=sys.stderr)
                return 1
    print(f"unknown target: {args.target}", file=sys.stderr)
    return 2


def _cmd_uninstall(args: argparse.Namespace) -> int:
    adapters = _enumerate_adapters()
    # Symmetric with _cmd_install: accept hyphen-style aliases.
    target_normalized = args.target.replace("-", "_")
    for a in adapters:
        if a.framework_name == target_normalized:
            try:
                _instantiate(a).uninstall()
                print(f"uninstalled: {a.framework_name}")
                return 0
            except Exception as e:
                print(f"uninstall failed: {e}", file=sys.stderr)
                return 1
    print(f"unknown target: {args.target}", file=sys.stderr)
    return 2


def _cmd_config(args: argparse.Namespace) -> int:
    if args.op == "path":
        print(USER_CONFIG_PATH)
        return 0
    if args.op == "show":
        try:
            cfg = load_config()
        except Exception as e:
            print(f"config error: {e}", file=sys.stderr)
            return 1
        for k, v in cfg.to_dict().items():
            print(f"{k} = {v!r}")
        return 0
    if args.op == "set":
        if not args.kv or "=" not in args.kv:
            print("usage: agent-amp config set key=value", file=sys.stderr)
            return 2
        # this command is NOT implemented in V1.  We
        # exit non-zero so a wrapper script using ``set -e`` does not
        # silently believe the config was changed.  Programmatic TOML
        # round-trip ships in V1.1 (LLD scope).
        print(
            f"agent-amp config set: not implemented in V1. "
            f"Edit {USER_CONFIG_PATH} directly.",
            file=sys.stderr,
        )
        return 5
    return 2  # unreachable per choices guard


def _cmd_report(args: argparse.Namespace) -> int:
    """Render the amplification dashboard. Imports the report module lazily
    so ``agent-amp --help`` and unrelated commands stay fast."""
    from agent_amplifier.report import render_report

    return render_report(last=args.last)


def _cmd_dashboard() -> int:
    import os

    raw_port = os.environ.get("AGENT_AMP_DASHBOARD_PORT", "8765")
    try:
        port = int(raw_port)
    except ValueError:
        print(
            "AGENT_AMP_DASHBOARD_PORT must be an integer",
            file=sys.stderr,
        )
        return 2
    if not 1 <= port <= 65535:
        print(
            "AGENT_AMP_DASHBOARD_PORT must be between 1 and 65535",
            file=sys.stderr,
        )
        return 2
    try:
        import uvicorn
    except ImportError:
        print(
            "dashboard requires uvicorn; install agent-amplifier with dashboard dependencies",
            file=sys.stderr,
        )
        return 1
    uvicorn.run(
        "agent_amplifier.dashboard.backend.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=port,
    )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from agent_amplifier import bench

    # H-6 single-prompt mode: ``--prompt PROMPT`` (with optional
    # ``--baseline`` / ``--vs-amplified``) delegates to bench_demo.
    # Default (neither flag set) shows both halves so the launch-GIF
    # capture command is just ``agent-amp bench --prompt "..."``.
    if getattr(args, "prompt", None):
        from agent_amplifier.bench_demo import run_demo

        baseline = bool(args.baseline)
        amplified = bool(args.vs_amplified)
        if not baseline and not amplified:
            baseline = True
            amplified = True
        return run_demo(
            args.prompt,
            show_baseline=baseline,
            show_amplified=amplified,
        )
    return bench.run_cli(args)


def _cmd_demo(args: argparse.Namespace) -> int:
    """``agent-amp demo <prompt>`` — friendly alias for ``bench --prompt``.
    Always shows both halves (the launch-GIF default)."""
    from agent_amplifier.bench_demo import run_demo

    return run_demo(args.prompt, show_baseline=True, show_amplified=True)


# ---------------------------------------------------------------------------
# Persona subcommand
# ---------------------------------------------------------------------------


def _builtin_slugs() -> set[str]:
    from agent_amplifier.persona_docs import BUILTIN_PERSONA_DOCS

    return {d.slug for d in BUILTIN_PERSONA_DOCS}


def _cmd_persona(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    sub_cmd = getattr(args, "persona_cmd", None)
    if sub_cmd == "list":
        return _cmd_persona_list()
    if sub_cmd == "show":
        return _cmd_persona_show(args.slug)
    if sub_cmd == "add":
        return _cmd_persona_add(args)
    if sub_cmd == "remove":
        return _cmd_persona_remove(args.name)
    # No subcommand → print help for the persona group.
    # We synthesize the help via argparse's parser_for_subcommand pattern by
    # calling the prog with --help; safer is to just print a focused message.
    print(
        "Usage: agent-amp persona <list|show|add|remove> [options]\n"
        "\n"
        "  list                       show built-in and custom personas\n"
        "  show <slug>                show one persona's details\n"
        "  add --name <slug> --label <label> --description <desc>\n"
        "      [--review-focus a,b,c] add a custom persona\n"
        "  remove --name <slug>       remove a custom persona (built-ins protected)"
    )
    _ = parser  # parser kept for future help-via-parser routing
    return 0


def _cmd_persona_list() -> int:
    from agent_amplifier.persona_docs import list_all_personas

    print("BUILT-IN PERSONAS\n")
    customs: list[dict[str, Any]] = []
    for entry in list_all_personas():
        if entry["custom"]:
            customs.append(entry)
            continue
        print(f"  {entry['slug']:38s}  {entry['label']}")
        print(f"    Value:    {entry['value_tagline']}")
        print(f"    When:     {entry['when_to_use']}")
        print()
    if customs:
        print("CUSTOM PERSONAS\n")
        for entry in customs:
            focus = ", ".join(entry["focus"]) if entry["focus"] else "-"
            print(f"  {entry['slug']:38s}  {entry['label']}")
            print(f"    Value:    {entry['value_tagline']}")
            print(f"    Focus:    {focus}")
            print()
    return 0


def _cmd_persona_show(slug: str) -> int:
    from agent_amplifier.persona_docs import list_all_personas

    for entry in list_all_personas():
        if entry["slug"] == slug:
            print(f"slug:        {entry['slug']}")
            print(f"label:       {entry['label']}")
            print(f"value:       {entry['value_tagline']}")
            print(f"when to use: {entry['when_to_use']}")
            focus = ", ".join(entry["focus"]) if entry["focus"] else "(none)"
            print(f"focus:       {focus}")
            if not entry["custom"]:
                print(f"role:        {entry['role']}")
                print(f"strictness:  {entry['strictness']}")
                print(f"severity:    {entry['severity_threshold']}")
                print(f"level:       {entry['level']}")
            else:
                print("type:        user-defined custom persona")
            return 0
    print(f"persona not found: {slug}")
    return 2


def _cmd_persona_add(args: argparse.Namespace) -> int:
    from agent_amplifier.custom_personas import (
        CustomPersona,
        InvalidPersonaError,
        save_custom_persona,
    )

    if args.name in _builtin_slugs():
        print(f"cannot use built-in slug as a custom persona name: {args.name}")
        return 2
    review_focus_raw: str = args.review_focus or ""
    review_focus = tuple(
        x.strip() for x in review_focus_raw.split(",") if x.strip()
    )
    persona = CustomPersona(
        name=args.name,
        label=args.label,
        description=args.description,
        review_focus=review_focus,
    )
    try:
        save_custom_persona(persona)
    except InvalidPersonaError as exc:
        print(f"invalid persona: {exc}")
        return 2
    print(f"added custom persona: {args.name}")
    return 0


def _cmd_persona_remove(name: str) -> int:
    from agent_amplifier.custom_personas import delete_custom_persona

    if name in _builtin_slugs():
        print(f"cannot remove built-in persona: {name}")
        return 2
    removed = delete_custom_persona(name)
    if not removed:
        print(f"persona not found: {name}")
        return 2
    print(f"removed custom persona: {name}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anyio_version() -> str:
    try:
        from importlib.metadata import version

        return version("anyio")
    except Exception:
        return "missing"


__all__ = ["main"]
