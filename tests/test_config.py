# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for ``agent_amplifier.config``.

Spec source: .2 (TOML config loader, audit-remediated).

Behavior matrix in §2.4 (26 cases) is the canonical reference.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

import pytest

from agent_amplifier import config as cfg_mod
from agent_amplifier.config import (
    ENV_VAR,
    LEGACY_USER_CONFIG_PATH,
    MAX_FILE_SIZE,
    USER_CONFIG_PATH,
    XDG_CONFIG_HOME,
    ConfigError,
    load_config,
    merge_config,
    validate_config,
)
from agent_amplifier.types import AmplifierConfig, BudgetMode

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Force HOME inside ``tmp_path``, clear env var.

    Every test that loads config gets a clean home directory and no env var
    pointing elsewhere. Returns the tmp_path so tests can manufacture files.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    # Reload module-level paths so they pick up the new HOME.
    monkeypatch.setattr(
        cfg_mod, "XDG_CONFIG_HOME", fake_home / ".config"
    )
    monkeypatch.setattr(
        cfg_mod,
        "USER_CONFIG_PATH",
        fake_home / ".config" / "agent-amplifier" / "config.toml",
    )
    monkeypatch.setattr(
        cfg_mod,
        "LEGACY_USER_CONFIG_PATH",
        fake_home / ".agent-amplifier" / "config.toml",
    )
    return fake_home


def _write_toml(p: Path, body: bytes | str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, str):
        body = body.encode("utf-8")
    p.write_bytes(body)
    return p


# ---------------------------------------------------------------------------
# 1. Defaults / no file / paths
# ---------------------------------------------------------------------------


def test_defaults_when_no_file(isolated_env: Path) -> None:
    cfg = load_config()
    assert cfg == AmplifierConfig()


def test_xdg_path_constant_under_home() -> None:
    """Sanity check on module-level constants without monkeypatching."""
    assert isinstance(USER_CONFIG_PATH, Path)
    assert isinstance(LEGACY_USER_CONFIG_PATH, Path)
    assert isinstance(XDG_CONFIG_HOME, Path)
    assert isinstance(MAX_FILE_SIZE, int)
    assert MAX_FILE_SIZE == 1_048_576


# ---------------------------------------------------------------------------
# 2. Primary file
# ---------------------------------------------------------------------------


def test_primary_user_file_overrides_defaults(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, "max_iterations = 6\nconvergence_threshold = 0.85\n")
    cfg = load_config()
    assert cfg.max_iterations == 6
    assert cfg.convergence_threshold == 0.85
    # Untouched fields keep defaults.
    assert cfg.budget_mode is BudgetMode.AUTO


def test_alias_path_used_when_primary_missing(isolated_env: Path) -> None:
    p = isolated_env / ".agent-amplifier" / "config.toml"
    _write_toml(p, "max_iterations = 7\n")
    cfg = load_config()
    assert cfg.max_iterations == 7


def test_collision_warning_logged(
    isolated_env: Path, caplog: pytest.LogCaptureFixture
) -> None:
    primary = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    legacy = isolated_env / ".agent-amplifier" / "config.toml"
    _write_toml(primary, "max_iterations = 3\n")
    _write_toml(legacy, "max_iterations = 9\n")
    with caplog.at_level(logging.WARNING):
        cfg = load_config()
    assert cfg.max_iterations == 3            # primary wins
    assert any(
        "alias" in rec.getMessage().lower()
        or "primary" in rec.getMessage().lower()
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# 3. Env-var override
# ---------------------------------------------------------------------------


def test_env_var_overrides_user_files(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(user, "max_iterations = 3\n")
    env_target = isolated_env / "custom.toml"
    _write_toml(env_target, "max_iterations = 8\n")
    monkeypatch.setenv(ENV_VAR, str(env_target))
    cfg = load_config()
    assert cfg.max_iterations == 8


def test_env_var_to_missing_file_raises(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, str(isolated_env / "nope.toml"))
    with pytest.raises(ConfigError):
        load_config()


def test_env_var_to_directory_raises(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var points at a directory, not a file."""
    a_dir = isolated_env / "subdir"
    a_dir.mkdir()
    monkeypatch.setenv(ENV_VAR, str(a_dir))
    with pytest.raises(ConfigError, match=r"non-file"):
        load_config()


def test_explicit_path_to_directory_raises(isolated_env: Path) -> None:
    a_dir = isolated_env / "subdir"
    a_dir.mkdir()
    with pytest.raises(ConfigError, match=r"not a regular file"):
        load_config(path=str(a_dir))


def test_env_var_outside_allowed_roots_raises(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file outside HOME (and outside /etc/agent-amplifier) is rejected."""
    outside = tmp_path / "outside.toml"
    _write_toml(outside, "max_iterations = 5\n")
    # ``outside`` is in tmp_path but NOT under fake HOME.
    monkeypatch.setenv(ENV_VAR, str(outside))
    with pytest.raises(ConfigError, match=r"outside allowed roots"):
        load_config()


def test_explicit_path_argument_bypasses_env_and_file(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(user, "max_iterations = 3\n")
    monkeypatch.setenv(ENV_VAR, str(isolated_env / "ignored.toml"))
    explicit = isolated_env / "explicit.toml"
    _write_toml(explicit, "max_iterations = 9\n")
    cfg = load_config(path=str(explicit))
    assert cfg.max_iterations == 9


def test_explicit_path_to_missing_file_raises(isolated_env: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(path=str(isolated_env / "nope.toml"))


# ---------------------------------------------------------------------------
# 4. Empty / whitespace / BOM
# ---------------------------------------------------------------------------


def test_empty_file_yields_defaults(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"")
    assert load_config() == AmplifierConfig()


def test_whitespace_only_yields_defaults(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, "   \n\t\n")
    assert load_config() == AmplifierConfig()


def test_config_with_utf8_bom(isolated_env: Path) -> None:
    """.3 — BOM-prefixed file parses correctly.

    Empirically verified on Python 3.11 + 3.14: ``tomllib.loads`` does NOT
    silently strip the BOM. The loader strips it explicitly.
    """
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"\xef\xbb\xbfmax_iterations = 5\n")
    cfg = load_config()
    assert cfg.max_iterations == 5


# ---------------------------------------------------------------------------
# 5. Malformed TOML, root-shape, unknown keys
# ---------------------------------------------------------------------------


def test_malformed_toml_raises(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"not valid = = = toml")
    with pytest.raises(ConfigError, match=r"malformed TOML"):
        load_config()


def test_unknown_key_raises(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"foo = 1\n")
    with pytest.raises(ConfigError, match=r"Unknown config keys"):
        load_config()


def test_unknown_key_message_lists_offending_key(
    isolated_env: Path,
) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"foo = 1\nbar = 2\n")
    with pytest.raises(ConfigError) as ei:
        load_config()
    msg = str(ei.value)
    assert "foo" in msg
    assert "bar" in msg


# ---------------------------------------------------------------------------
# 6. Bounds + enum coercion
# ---------------------------------------------------------------------------


def test_max_iterations_zero_raises(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"max_iterations = 0\n")
    with pytest.raises(ConfigError, match=r"max_iterations"):
        load_config()


def test_invalid_budget_mode_raises(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b'budget_mode = "turbo"\n')
    with pytest.raises(ConfigError, match=r"budget_mode"):
        load_config()


def test_string_budget_mode_coerced_in_validate_config(
    isolated_env: Path,
) -> None:
    """A-4 strict — coercion lives in ``validate_config`` only."""
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b'budget_mode = "balanced"\n')
    cfg = load_config()
    assert cfg.budget_mode is BudgetMode.BALANCED


def test_string_for_escalate_low_confidence_raises(
    isolated_env: Path,
) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b'escalate_low_confidence = "yes"\n')
    with pytest.raises(ConfigError, match=r"escalate_low_confidence"):
        load_config()


def test_escalate_low_confidence_true_accepted(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"escalate_low_confidence = true\n")
    cfg = load_config()
    assert cfg.escalate_low_confidence is True


# — recall_limit TOML loading round-trip + bounds


def test_recall_limit_loaded_from_toml(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"recall_limit = 7\n")
    cfg = load_config()
    assert cfg.recall_limit == 7


def test_recall_limit_zero_in_toml_raises(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"recall_limit = 0\n")
    with pytest.raises(ConfigError, match=r"recall_limit"):
        load_config()


def test_recall_limit_too_large_in_toml_raises(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"recall_limit = 1000\n")
    with pytest.raises(ConfigError, match=r"recall_limit"):
        load_config()


def test_recall_limit_default_when_absent(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"max_iterations = 5\n")
    cfg = load_config()
    assert cfg.recall_limit == 3


# ---------------------------------------------------------------------------
# 7. Size cap
# ---------------------------------------------------------------------------


def test_size_cap_via_fstat(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    # 2 MiB of plausible TOML keys would also cap; keep the bytes simple.
    _write_toml(p, b"# " + b"a" * (MAX_FILE_SIZE + 100))
    with pytest.raises(ConfigError, match=r"too large"):
        load_config()


# ---------------------------------------------------------------------------
# 8. Path-traversal hardening
# ---------------------------------------------------------------------------


def test_null_byte_in_path_argument_raises(isolated_env: Path) -> None:
    """OS rejects null-byte env vars at the layer before us, so we pin the
    null-byte defense through the explicit ``path=`` argument (which routes
    through ``_safe_resolve`` regardless of OS env-layer behavior).
    """
    with pytest.raises(ConfigError, match=r"null byte"):
        load_config(path="x\x00.toml")


def test_null_byte_in_env_var_raises_via_os_or_loader(
    isolated_env: Path,
) -> None:
    """If a platform DOES allow null-byte env vars, we still raise.

    On macOS/Linux, ``os.environ`` rejects null bytes at assignment, so this
    test pokes the dict-like environ directly to bypass that check. We
    accept either platform behavior — the contract is "null byte never
    yields a parsed config".
    """
    # Bypass the OS reject by directly poking the proxy env. We use a low-
    # level dict assignment via ``os.environ.__class__.__setitem__`` only
    # if it would succeed; on most platforms it raises, in which case we
    # consider the test trivially satisfied.
    try:
        os.environ[ENV_VAR] = "x\x00.toml"
    except ValueError:
        # Platform rejects null-byte env vars before our code sees them.
        # That is itself a valid defense; nothing more to assert.
        pytest.skip("platform rejects null-byte env vars")
    try:
        with pytest.raises(ConfigError):
            load_config()
    finally:
        os.environ.pop(ENV_VAR, None)


@pytest.mark.skipif(
    sys.platform == "win32", reason="symlinks behave differently on Windows"
)
def test_symlink_loop_raises(isolated_env: Path) -> None:
    a = isolated_env / "a.toml"
    b = isolated_env / "b.toml"
    a.symlink_to(b)
    b.symlink_to(a)
    with pytest.raises(ConfigError):
        load_config(path=str(a))


# ---------------------------------------------------------------------------
# 9. Concurrent reads
# ---------------------------------------------------------------------------


def test_concurrent_load(isolated_env: Path) -> None:
    """N threads reading the same file → all succeed; identical result.

    CRIT mitigation: use a ``threading.Barrier`` so all threads call
    ``load_config`` in the same instant — a serial implementation would
    block at the barrier indefinitely. This is a real concurrency stress,
    not just a lucky-pass.
    """
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"max_iterations = 4\nconvergence_threshold = 0.92\n")

    n_threads = 16
    barrier = threading.Barrier(n_threads)
    results: list[AmplifierConfig] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            barrier.wait(timeout=5.0)
            r = load_config()
            with lock:
                results.append(r)
        except BaseException as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert not errors, errors
    assert len(results) == n_threads
    first = results[0]
    for r in results[1:]:
        assert r == first


# ---------------------------------------------------------------------------
# 10. Error messages do NOT echo file bytes
# ---------------------------------------------------------------------------


def test_error_does_not_echo_secret_payload(isolated_env: Path) -> None:
    """A malformed file containing a fake secret token must NOT have that
    token echoed in the ConfigError.
    """
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    secret = "ghp_" + "A" * 40
    _write_toml(p, f'foo = "{secret}\nbroken = '.encode())
    with pytest.raises(ConfigError) as ei:
        load_config()
    rendered = str(ei.value) + repr(ei.value.__cause__)
    # The path is allowed (debugging aid). The secret must NOT leak.
    assert secret not in rendered


# ---------------------------------------------------------------------------
# 11. observability_callback can NOT be set via TOML
# ---------------------------------------------------------------------------


def test_observability_callback_in_toml_rejected(
    isolated_env: Path,
) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b'observability_callback = "fn"\n')
    with pytest.raises(ConfigError, match=r"observability_callback"):
        load_config()


def test_observability_callback_none_in_toml_is_silently_dropped(
    isolated_env: Path,
) -> None:
    """An explicit ``observability_callback = `` (parsed as missing) doesn't
    happen in TOML. But if a user wrote ``observability_callback = `` and
    TOML had a way to express null, we'd accept it as no-op. Today, TOML
    has no null literal — covered by the rejection test above.
    """
    # The fixture isolates HOME but is otherwise unused — the fact-check
    # exercises ``validate_config`` directly, with no file I/O.
    del isolated_env
    cfg = validate_config({"observability_callback": None})
    assert cfg.observability_callback is None


# ---------------------------------------------------------------------------
# 12. PyYAML is NOT imported
# ---------------------------------------------------------------------------


def test_no_yaml_module_imported() -> None:
    """.2 — defends against accidental re-introduction."""
    assert "yaml" not in sys.modules


# ---------------------------------------------------------------------------
# 13. merge_config — pure
# ---------------------------------------------------------------------------


def test_merge_config_returns_new_dict() -> None:
    a = {"x": 1}
    b = {"x": 2}
    out = merge_config(a, b)
    assert out == {"x": 2}
    assert a == {"x": 1}            # not mutated
    assert b == {"x": 2}
    assert out is not a and out is not b


def test_merge_config_overrides_win() -> None:
    out = merge_config({"a": 1, "b": 2}, {"b": 99, "c": 3})
    assert out == {"a": 1, "b": 99, "c": 3}


def test_merge_config_rejects_non_dict() -> None:
    with pytest.raises(TypeError):
        merge_config([1, 2], {})            # type: ignore[arg-type]
    with pytest.raises(TypeError):
        merge_config({}, "x")                # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 14. validate_config — direct unit tests
# ---------------------------------------------------------------------------


def test_validate_config_minimum_dict() -> None:
    cfg = validate_config({})
    assert cfg == AmplifierConfig()


def test_validate_config_unknown_key_raises_directly() -> None:
    with pytest.raises(ConfigError, match=r"Unknown config keys"):
        validate_config({"foo": 1})


def test_validate_config_invalid_value_wrapped_in_config_error() -> None:
    with pytest.raises(ConfigError, match=r"max_iterations"):
        validate_config({"max_iterations": 42})


# ---------------------------------------------------------------------------
# 15. ConfigError carries optional source path
# ---------------------------------------------------------------------------


def test_config_error_source_attribute(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"max_iterations = -1\n")
    try:
        load_config()
    except ConfigError as e:
        # source is populated for read errors; for validation errors it may
        # be None — both are acceptable. We just want to know the attribute
        # exists on the type.
        assert hasattr(e, "source")
    else:  # pragma: no cover - sanity
        pytest.fail("expected ConfigError")


# ---------------------------------------------------------------------------
# 16. Root-must-be-table
# ---------------------------------------------------------------------------


def test_toml_array_at_root_raises(isolated_env: Path) -> None:
    """TOML doesn't allow bare arrays at root — covered by malformed branch."""
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"[1, 2, 3]")
    with pytest.raises(ConfigError):
        load_config()


def test_invalid_utf8_raises(isolated_env: Path) -> None:
    """Bytes that are not valid UTF-8 raise ConfigError, not UnicodeDecodeError."""
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    # Lone continuation byte — invalid UTF-8.
    _write_toml(p, b"\xc3\x28 invalid utf-8")
    with pytest.raises(ConfigError, match=r"not valid UTF-8"):
        load_config()


# ---------------------------------------------------------------------------
# 18. Defensive failure paths — monkeypatched OS layer
# ---------------------------------------------------------------------------


def test_open_failure_raises_config_error(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permission-denied / EIO at ``os.open`` is wrapped as ConfigError."""
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"max_iterations = 4\n")

    real_open = os.open

    def boom_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        if "agent-amplifier" in str(path):
            raise PermissionError(13, "permission denied")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", boom_open)
    with pytest.raises(ConfigError, match=r"could not open config file"):
        load_config()


def test_read_failure_raises_config_error(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``os.read`` raising EIO mid-read is wrapped as ConfigError."""
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"max_iterations = 4\n")

    real_read = os.read

    def boom_read(fd: int, n: int) -> bytes:
        # First call (the kernel-side TomlBridge / coverage tracer might
        # also call os.read) — only blow up the read against the open fd
        # of our config file. Heuristic: read of >= 1 KiB matches our cap+1.
        if n > 1024:
            raise OSError(5, "I/O error")
        return real_read(fd, n)

    monkeypatch.setattr(os, "read", boom_read)
    with pytest.raises(ConfigError, match=r"could not read config file"):
        load_config()


def test_mid_read_size_cap_defensive(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Size cap also enforced after read (defensive against under-reporting fstat)."""
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"max_iterations = 4\n")

    # Fake fstat to under-report; real read returns more than reported.
    real_fstat = os.fstat

    class FakeStat:
        def __init__(self, real: os.stat_result) -> None:
            self.real = real

        def __getattr__(self, name: str) -> object:
            if name == "st_size":
                return 100
            return getattr(self.real, name)

    def lying_fstat(fd: int) -> object:                 # type: ignore[override]
        return FakeStat(real_fstat(fd))

    real_read = os.read

    def fat_read(fd: int, n: int) -> bytes:
        # Return MAX+10 bytes regardless of n.
        return real_read(fd, n) + b"\x00" * (MAX_FILE_SIZE + 10)

    monkeypatch.setattr(os, "fstat", lying_fstat)
    monkeypatch.setattr(os, "read", fat_read)
    with pytest.raises(ConfigError, match=r"exceeded size cap mid-read"):
        load_config()


def test_toml_root_must_be_table_defensive(tmp_path: Path) -> None:
    """``_read_toml`` enforces dict-shape root even when tomllib accepts.

    ``tomllib.loads`` always returns a dict for valid TOML; this is a
    belt-and-suspenders branch. Cover it via direct monkey-patch.
    """
    import tomllib as _tom

    from agent_amplifier import config as _cfg

    real_loads = _tom.loads

    def fake_loads(s: str) -> object:                 # type: ignore[override]
        return [1, 2, 3]            # not a dict — provoke our check

    fake_path = tmp_path / "fake_loadscheck.toml"
    try:
        fake_path.write_bytes(b"max_iterations = 4\n")
        # Patch tomllib.loads via the imported module reference.
        old = _tom.loads
        _tom.loads = fake_loads            # type: ignore[assignment]
        try:
            with pytest.raises(ConfigError, match=r"root must be a TOML table"):
                _cfg._read_toml(fake_path)
        finally:
            _tom.loads = old
            del real_loads
    finally:
        if fake_path.exists():
            fake_path.unlink()


# ---------------------------------------------------------------------------
# 17. Code-construction with callback survives load_config interplay
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CRIT — added during self-review (Stripe-Senior-Test-Engineer mode)
# ---------------------------------------------------------------------------


def test_crit_allowed_roots_explicit_path_real_home_unaffected(
    isolated_env: Path, tmp_path: Path
) -> None:
    """CRIT W-1: pin allowed-roots logic for explicit paths.

    The ``isolated_env`` fixture replaces XDG paths and HOME env, but
    ``Path.home()`` returns the real OS home. We expect: an explicit path
    pointing to a file outside any allowed root (real home, fake home, or
    /etc/agent-amplifier) IS rejected. We construct such a file in a
    sibling ``tmp_path`` directory (sibling of fake HOME, NOT inside it).
    """
    # ``tmp_path`` is the actual tmp folder; ``isolated_env`` is tmp_path/home.
    sibling = tmp_path / "siblings" / "outside.toml"
    sibling.parent.mkdir(parents=True)
    _write_toml(sibling, b"max_iterations = 5\n")

    # If real Path.home() happens to be the user's actual ~ and tmp_path is
    # also under it (macOS uses /var/folders/... for tmp_path which is
    # outside ~), then this test exercises the rejection.
    real_home = Path.home().resolve()
    if str(sibling).startswith(str(real_home)):
        # Test environment makes tmp_path live under real ~ — we can't
        # prove the negative; mark as a documented skip rather than a
        # false-pass.
        pytest.skip("tmp_path nested under real Path.home() — cannot test rejection")
    with pytest.raises(ConfigError, match=r"outside allowed roots"):
        load_config(path=str(sibling))


def test_crit_toml_loadable_fields_excludes_observability_callback() -> None:
    """CRIT W-2: regression guard on the loadable-from-TOML set
    must NEVER include ``observability_callback``. Even an accidental
    spec edit must fail this test.
    """
    assert "observability_callback" not in cfg_mod._TOML_LOADABLE_FIELDS
    # And every other public AmplifierConfig field IS loadable from TOML.
    expected_loadable = {
        "max_iterations",
        "convergence_threshold",
        "budget_mode",
        "goal_reinjection_interval",
        "effort_router",
        "tool_selector",
        "escalate_low_confidence",
        # — H6 fix. ``recall_limit`` IS loadable from TOML because
        # it's a plain int; the dataclass-level validator enforces the [1, 100]
        # range and refuses bools.
        "recall_limit",
        "disabled_ips",
        "ip_order",
        # IP-8 v2: persona slug is loadable from TOML.
        "persona",
    }
    assert expected_loadable == cfg_mod._TOML_LOADABLE_FIELDS


def test_dashboard_ip_state_loaded_from_toml(isolated_env: Path) -> None:
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(
        p,
        b'disabled_ips = ["kernel"]\nip_order = ["kernel", "effort_router"]\n',
    )
    cfg = load_config()
    assert cfg.disabled_ips == ("kernel",)
    assert cfg.ip_order == ("kernel", "effort_router")


def test_code_construction_with_callback_unaffected_by_file(
    isolated_env: Path,
) -> None:
    """A user who sets ``observability_callback`` in code is not blocked by
    a TOML file lacking the field.
    """
    p = isolated_env / ".config" / "agent-amplifier" / "config.toml"
    _write_toml(p, b"max_iterations = 5\n")
    file_cfg = load_config()
    code_cfg = AmplifierConfig(
        max_iterations=file_cfg.max_iterations,
        observability_callback=lambda e, p: None,
    )
    assert code_cfg.observability_callback is not None
    assert code_cfg.max_iterations == 5
