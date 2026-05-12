# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier — config loader (, TOML).

Search order:
    1. explicit ``path=`` argument
    2. ``$AGENT_AMPLIFIER_CONFIG`` (if set, file must exist)
    3. ``$XDG_CONFIG_HOME/agent-amplifier/config.toml``
       (default: ``~/.config/...``)
    4. ``~/.agent-amplifier/config.toml`` (back-compat alias; warn on collision)
    5. ``AmplifierConfig()`` built-in defaults

Public API:
    * ``load_config(path=None) -> AmplifierConfig``
    * ``merge_config(defaults, overrides) -> dict``
    * ``validate_config(raw) -> AmplifierConfig``
    * ``ConfigError``


    * YAML→TOML, XDG path, drop pyyaml.
    * realpath + allowed-roots + fstat + truncated errors.
    * explicit BOM strip; symlink-loop catch; null-byte refusal;
                concurrent reads safe.
    * ``validate_config`` refuses TOML-loaded
                ``observability_callback``.
    * ``_ALLOWED_CONFIG_FIELDS`` sourced from ``types.py`` via
                ``MappingProxyType``.
"""

from __future__ import annotations

import logging
import os
import tomllib  # PEP 680 (Python 3.11+)
from pathlib import Path
from typing import Any

from agent_amplifier.types import (
    _ALLOWED_CONFIG_FIELDS,  # MappingProxyType
    AmplifierConfig,
    BudgetMode,
)

LOG = logging.getLogger(__name__)

ENV_VAR = "AGENT_AMPLIFIER_CONFIG"

# Primary path — XDG-compliant.
XDG_CONFIG_HOME = Path(
    os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
)
USER_CONFIG_PATH = XDG_CONFIG_HOME / "agent-amplifier" / "config.toml"

# Backward-compat alias path (for users on the older spec).
LEGACY_USER_CONFIG_PATH = Path.home() / ".agent-amplifier" / "config.toml"

# Hard cap on config file size — enforced via fstat on the open fd.
MAX_FILE_SIZE = 1_048_576  # 1 MiB

# UTF-8 BOM bytes. Empirically verified (Python 3.11 + 3.14, 2026-04-26):
# ``tomllib`` does NOT silently strip the BOM via either ``.load`` or
# ``.loads``. We strip it explicitly before parsing.
_UTF8_BOM = b"\xef\xbb\xbf"

# Fields that may appear in a TOML file. ``observability_callback`` is
# INTENTIONALLY absent — callbacks must be wired in code, not loaded from
# TOML (would require importing arbitrary names; security regression).
_TOML_LOADABLE_FIELDS: frozenset[str] = frozenset(_ALLOWED_CONFIG_FIELDS) - {
    "observability_callback",
}


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or is invalid."""

    def __init__(
        self, message: str, *, source: Path | str | None = None
    ) -> None:
        super().__init__(message)
        self.source = source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    path: str | os.PathLike[str] | None = None,
) -> AmplifierConfig:
    """Load Amplifier configuration.

    Loading is path-traversal-hardened: the resolved path must lie
    inside an allowed root (user home, or ``/etc/agent-amplifier``). Errors
    NEVER echo file contents.

    Note: ``observability_callback`` cannot be loaded from a config file —
    callbacks are wired in code. If a TOML file declares it (with a
    non-None value), an error is raised.
    """
    source = _resolve_source(path)
    raw_user: dict[str, Any] = (
        _read_toml(source) if source is not None else {}
    )

    defaults = _defaults_for_toml()
    merged = merge_config(defaults, raw_user)
    return validate_config(merged)


def merge_config(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Shallow merge: ``overrides`` wins per key. Returns a NEW dict.

    NOTE: Shallow merge only. If schema gains nested keys (e.g.
    ``effort_router: { kind: ..., thresholds: {...} }``), revisit with a
    dedicated merge implementation that preserves nested defaults.
    """
    if not isinstance(defaults, dict) or not isinstance(overrides, dict):
        raise TypeError("merge_config arguments must both be dict")
    return {**defaults, **overrides}


def validate_config(raw: dict[str, Any]) -> AmplifierConfig:
    """Validate a raw config dict and produce a frozen ``AmplifierConfig``.

    A-3 strict: unknown keys → ``ConfigError``.
    A-4 strict: ``budget_mode`` may be a string here (we coerce); the
    ``AmplifierConfig`` ``__init__`` refuses strings — coercion lives only
    here.

    refuses ``observability_callback`` if it came in with a
    non-None value (we cannot represent a callable in TOML; presence
    implies an attempt to load arbitrary names — security regression).
    Code-construction users may still pass a callable via
    ``AmplifierConfig(observability_callback=fn)`` directly — this function
    is for the file path only.
    """
    allowed = set(_ALLOWED_CONFIG_FIELDS) - {"observability_callback"}
    unknown = set(raw.keys()) - allowed - {"observability_callback"}
    if unknown:
        raise ConfigError(
            f"Unknown config keys: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )

    # ``observability_callback`` in raw (with non-None value) means a TOML
    # file tried to declare it. We refuse — even literal `"<callable>"`.
    if (
        "observability_callback" in raw
        and raw["observability_callback"] is not None
    ):
        raise ConfigError(
            "observability_callback cannot be set via TOML — wire it in "
            "code: AmplifierConfig(observability_callback=fn). "
            "See .5."
        )

    coerced = dict(raw)
    coerced.pop("observability_callback", None)        # not loadable from TOML

    if "budget_mode" in coerced and isinstance(coerced["budget_mode"], str):
        try:
            coerced["budget_mode"] = BudgetMode(coerced["budget_mode"])
        except ValueError as e:
            raise ConfigError(
                f"Invalid budget_mode {coerced['budget_mode']!r}; "
                f"allowed: {[m.value for m in BudgetMode]}"
            ) from e
    for key in ("disabled_ips", "ip_order"):
        if key in coerced and isinstance(coerced[key], list):
            coerced[key] = tuple(coerced[key])

    try:
        return AmplifierConfig(**coerced)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Invalid config: {e}") from e


# ---------------------------------------------------------------------------
# Internals — path safety + TOML reading
# ---------------------------------------------------------------------------


def _allowed_roots() -> list[Path]:
    """Roots that ``$AGENT_AMPLIFIER_CONFIG`` / explicit paths may resolve under.

    Includes user home and ``/etc/agent-amplifier`` (sysadmin-deployed
    configs). Implementer may extend in private fork — but every entry
    MUST be a canonicalized directory.
    """
    roots: list[Path] = [Path.home().resolve()]
    sys_root = Path("/etc/agent-amplifier")
    if sys_root.exists():
        roots.append(sys_root.resolve())
    return roots


def _safe_resolve(p: Path) -> Path:
    """canonicalize + allowed-roots check.

    ``strict=True`` raises on missing target — we want that. The caller
    checks for existence as needed before calling.

    Refuses paths containing null bytes (Python ``pathlib`` crashes on
    these on some platforms; raise our own clean error first).
    """
    if "\x00" in str(p):
        raise ConfigError(f"config path contains null byte: {p!r}")
    try:
        rp = p.expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as e:
        # ``OSError`` covers ``FileNotFoundError``, permission denied.
        # ``RuntimeError`` is raised by Python 3.11 ``pathlib.resolve``
        # for ELOOP symlink loops (CPython pathlib.py: ``raise
        # RuntimeError("Symlink loop from ...")``).
        # ``ValueError`` is raised on Windows for embedded null bytes
        # discovered late in resolve.
        raise ConfigError(f"could not resolve config path: {p}") from e
    for root in _allowed_roots():
        try:
            rp.relative_to(root)
            return rp
        except ValueError:
            continue
    raise ConfigError(
        f"config path outside allowed roots: {rp} "
        f"(allowed: {[str(r) for r in _allowed_roots()]})"
    )


def _resolve_source(
    explicit: str | os.PathLike[str] | None,
) -> Path | None:
    """Return the file to read, or ``None`` if defaults should be used."""
    if explicit is not None:
        rp = _safe_resolve(Path(explicit))
        if not rp.is_file():
            raise ConfigError(
                f"Explicit config file is not a regular file: {rp}",
                source=rp,
            )
        return rp

    env_value = os.environ.get(ENV_VAR)
    if env_value:
        rp = _safe_resolve(Path(env_value))
        if not rp.is_file():
            raise ConfigError(
                f"{ENV_VAR} points to non-file: {rp}", source=rp
            )
        return rp

    # Try primary path then alias. If both exist, primary wins, log a warning.
    primary_exists = USER_CONFIG_PATH.is_file()
    legacy_exists = LEGACY_USER_CONFIG_PATH.is_file()
    if primary_exists and legacy_exists:
        LOG.warning(
            "Both %s and %s exist; using primary. "
            "Consider deleting the alias.",
            USER_CONFIG_PATH,
            LEGACY_USER_CONFIG_PATH,
        )
        return _safe_resolve(USER_CONFIG_PATH)
    if primary_exists:
        return _safe_resolve(USER_CONFIG_PATH)
    if legacy_exists:
        return _safe_resolve(LEGACY_USER_CONFIG_PATH)
    return None


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file. Empty file → empty dict.

    Race-free single-fd pattern: open once with ``O_RDONLY``,
    ``fstat`` the open fd, then read from the same fd. No TOCTOU between
    stat and read.

    :
        * UTF-8 BOM is stripped (``tomllib`` does NOT handle it — raises
          ``TOMLDecodeError`` otherwise; verified empirically 2026-04-26
          on Python 3.11 + 3.14).
        * Symlink loops fail at ``_safe_resolve`` (``resolve(strict=True)``
          raises).
        * Null bytes in the path fail at ``_safe_resolve``.
        * Concurrent reads of the same file are safe — we use only the
          open fd; ``tomllib`` parses an in-memory bytes object once read.

    Errors NEVER echo file bytes. On ``TOMLDecodeError`` we
    surface line/column only via ``str(e)`` and suppress chained context
    (``from None``) that might include parser-internal echoing.
    """
    # Open binary (tomllib requires bytes). Open ONCE, stat the fd.
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError as e:
        raise ConfigError(f"could not open config file: {path}") from e

    try:
        st = os.fstat(fd)
        if st.st_size > MAX_FILE_SIZE:
            raise ConfigError(
                f"config file too large ({st.st_size} bytes > "
                f"{MAX_FILE_SIZE}): {path}",
                source=path,
            )
        try:
            data = os.read(fd, MAX_FILE_SIZE + 1)
        except OSError as e:
            raise ConfigError(
                f"could not read config file: {path}"
            ) from e
    finally:
        os.close(fd)

    if len(data) > MAX_FILE_SIZE:
        # Defensive: on filesystems that under-report stat (rare), enforce.
        raise ConfigError(
            f"config file exceeded size cap mid-read: {path}",
            source=path,
        )

    # Strip UTF-8 BOM if present.
    if data.startswith(_UTF8_BOM):
        data = data[len(_UTF8_BOM):]

    if not data.strip():
        return {}

    try:
        loaded = tomllib.loads(data.decode("utf-8"))
    except UnicodeDecodeError as e:
        raise ConfigError(
            f"config file is not valid UTF-8: {path} "
            f"(byte {e.start})",
            source=path,
        ) from None
    except tomllib.TOMLDecodeError as e:
        # NEVER echo file bytes. ``str(e)`` is parser-generated and does
        # not include the file path or contents (line/col only).
        raise ConfigError(
            f"malformed TOML in {path} ({e})", source=path
        ) from None

    if not isinstance(loaded, dict):
        raise ConfigError(
            f"config root must be a TOML table, got {type(loaded).__name__}",
            source=path,
        )
    return loaded


def _defaults_for_toml() -> dict[str, Any]:
    """Return ``AmplifierConfig`` defaults serialized for the TOML merge layer.

    The ``observability_callback`` field is excluded — TOML cannot represent
    it and ``validate_config`` refuses it.
    """
    base = AmplifierConfig().to_dict()
    base.pop("observability_callback", None)
    return base


__all__ = [
    "ENV_VAR",
    "LEGACY_USER_CONFIG_PATH",
    "MAX_FILE_SIZE",
    "USER_CONFIG_PATH",
    "XDG_CONFIG_HOME",
    "ConfigError",
    "load_config",
    "merge_config",
    "validate_config",
]
