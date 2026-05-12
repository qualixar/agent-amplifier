# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Validated TOML config read/write helpers for the dashboard backend."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import IO

import tomli_w

from agent_amplifier.adapters._path_safety import safe_open_write, safe_read_text
from agent_amplifier.config import ConfigError, load_config, validate_config
from agent_amplifier.dashboard.backend.ip_catalog import IP_CATALOG, IP_IDS
from agent_amplifier.dashboard.backend.models import IpInfo
from agent_amplifier.types import AmplifierConfig

_CONFIG_WRITE_LOCK = threading.Lock()


def config_to_dict(config: AmplifierConfig) -> dict[str, object]:
    """Return a JSON-safe config dict with no callable sentinel."""
    out = config.to_dict()
    out.pop("observability_callback", None)
    return out


def load_config_dict(config_path: Path) -> dict[str, object]:
    return config_to_dict(_load_dashboard_config(config_path))


def save_config_dict(config_path: Path, raw: dict[str, object]) -> dict[str, object]:
    config = validate_config(dict(raw))
    config_dict = config_to_dict(config)
    _write_config_atomically(config_path, config_dict)
    return config_dict


def list_ips(config_path: Path) -> list[IpInfo]:
    cfg = _load_dashboard_config(config_path)
    disabled = set(cfg.disabled_ips)
    ordered_ids = _normalized_order(cfg.ip_order)
    by_id = {entry.id: entry for entry in IP_CATALOG}
    infos: list[IpInfo] = []
    for index, ip_id in enumerate(ordered_ids, start=1):
        entry = by_id[ip_id]
        infos.append(
            IpInfo(
                id=entry.id,
                name=entry.name,
                file=entry.file,
                enabled=entry.id not in disabled,
                order=index,
            )
        )
    return infos


def toggle_ip(config_path: Path, ip_id: str) -> IpInfo | None:
    if ip_id not in IP_IDS:
        return None
    cfg = _load_dashboard_config(config_path)
    disabled = set(cfg.disabled_ips)
    if ip_id in disabled:
        disabled.remove(ip_id)
    else:
        disabled.add(ip_id)
    raw = config_to_dict(cfg)
    raw["disabled_ips"] = sorted(disabled)
    save_config_dict(config_path, raw)
    return next(ip for ip in list_ips(config_path) if ip.id == ip_id)


def reorder_ips(config_path: Path, ip_ids: list[str]) -> list[IpInfo]:
    if len(ip_ids) != len(IP_IDS) or set(ip_ids) != IP_IDS:
        raise ConfigError("reorder must include all IP ids exactly once")
    cfg = _load_dashboard_config(config_path)
    raw = config_to_dict(cfg)
    raw["ip_order"] = ip_ids
    save_config_dict(config_path, raw)
    return list_ips(config_path)


def _load_dashboard_config(config_path: Path) -> AmplifierConfig:
    if config_path.exists():
        return load_config(path=config_path)
    return validate_config({})


def _normalized_order(configured_order: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for ip_id in configured_order:
        if ip_id in IP_IDS and ip_id not in seen:
            ordered.append(ip_id)
            seen.add(ip_id)
    for entry in IP_CATALOG:
        if entry.id not in seen:
            ordered.append(entry.id)
    return ordered


def _write_config_atomically(config_path: Path, config_dict: dict[str, object]) -> None:
    body = tomli_w.dumps(config_dict)
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    backup_path = config_path.with_name(config_path.name + ".bak")
    tmp_path = config_path.with_name(config_path.name + ".tmp")

    with _CONFIG_WRITE_LOCK:
        if config_path.exists():
            previous = safe_read_text(config_path, config_dir)
            if previous is None:
                raise ConfigError(f"refusing to back up unsafe config path: {config_path}")
            with safe_writer(backup_path, config_dir) as backup:
                backup.write(previous)
                backup.flush()
                os.fsync(backup.fileno())
        with safe_writer(tmp_path, config_dir) as tmp:
            tmp.write(body)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, config_path)


class safe_writer:
    """Context manager wrapper around ``safe_open_write`` with typed failure."""

    def __init__(self, path: Path, allowed_root: Path) -> None:
        self._path = path
        self._allowed_root = allowed_root
        self._handle: IO[str] | None = None

    def __enter__(self) -> IO[str]:
        handle = safe_open_write(self._path, self._allowed_root)
        if handle is None:
            raise ConfigError(f"refusing unsafe config write: {self._path}")
        self._handle = handle
        return handle

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        handle = self._handle
        if handle is not None:
            handle.close()


__all__ = [
    "config_to_dict",
    "list_ips",
    "load_config_dict",
    "reorder_ips",
    "save_config_dict",
    "toggle_ip",
]
