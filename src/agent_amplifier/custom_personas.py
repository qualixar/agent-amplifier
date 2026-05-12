# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Custom persona storage with prompt-injection defense.

User-defined personas live at ``~/.config/agent-amplifier/personas.toml``
(or wherever ``AGENT_AMP_PERSONAS_PATH`` / ``XDG_CONFIG_HOME`` points).

The threat model: a custom persona ``description`` is user-supplied free text
that flows into the LLM prompt. An attacker (or a careless user who pasted in
content from somewhere) could embed ``<system-reminder>``, ``<tool_use>``, or
lookalike-character tags to attempt prompt injection. We apply
``apply_recall_safety()`` at the save boundary AND the load boundary AND the
render boundary — three layers of defense in depth.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import tomli_w

from agent_amplifier._internal.recall_safety import apply_recall_safety
from agent_amplifier.semantic_modifiers import _neutralize_xml

# ---------------------------------------------------------------------------
# Public types + exceptions
# ---------------------------------------------------------------------------


class InvalidPersonaError(ValueError):
    """Raised when a persona's fields fail validation before save."""


@dataclass(frozen=True, slots=True)
class CustomPersona:
    """User-defined persona persisted to ``personas.toml``.

    Fields:
        name: slug, ``[a-z][a-z0-9-_]{1,63}``. Stable identifier.
        label: human-readable display label (UI dropdown text).
        description: free text prompt — fed to the LLM after
            ``apply_recall_safety()`` neutralization.
        review_focus: tuple of audit axes (e.g. ``("security", "perf")``).
    """

    name: str
    label: str
    description: str
    review_focus: tuple[str, ...]


# ---------------------------------------------------------------------------
# Storage path resolution
# ---------------------------------------------------------------------------

_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "name", "label", "description", "review_focus",
)


def storage_path() -> Path:
    """Resolve ``personas.toml`` path with XDG + env-var precedence.

    Precedence (highest first):
    1. ``AGENT_AMP_PERSONAS_PATH`` env var (absolute path to TOML file).
    2. ``$XDG_CONFIG_HOME/agent-amplifier/personas.toml``.
    3. ``$HOME/.config/agent-amplifier/personas.toml``.
    """
    env = os.environ.get("AGENT_AMP_PERSONAS_PATH")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path(os.environ.get("HOME", "~")).expanduser() / ".config"
    return base / "agent-amplifier" / "personas.toml"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(persona: CustomPersona) -> None:
    """Raise ``InvalidPersonaError`` if any field fails the schema."""
    if not _NAME_RE.match(persona.name):
        raise InvalidPersonaError(
            "Invalid persona 'name' — must match [a-z][a-z0-9_-]{0,63}"
        )
    if not persona.label.strip():
        raise InvalidPersonaError("Persona 'label' must not be empty")
    if not persona.description.strip():
        raise InvalidPersonaError("Persona 'description' must not be empty")


def _sanitize_description(description: str) -> str:
    """Apply ``apply_recall_safety`` to neutralize prompt-injection vectors."""
    safe, _signals = apply_recall_safety(description)
    return safe


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def _coerce_entry(entry: dict[str, object]) -> CustomPersona | None:
    """Convert a raw TOML dict into a CustomPersona, or None if malformed."""
    if not all(field in entry for field in _REQUIRED_FIELDS):
        return None
    name = entry["name"]
    label = entry["label"]
    description = entry["description"]
    review_focus = entry["review_focus"]
    if not isinstance(name, str) or not isinstance(label, str):
        return None
    if not isinstance(description, str) or not isinstance(review_focus, list):
        return None
    # Defense in depth: re-apply sanitization on load (file may have been
    # hand-edited to insert hostile content after a clean save).
    safe_description = _sanitize_description(description)
    safe_label = _neutralize_xml(label)
    safe_focus = tuple(
        _neutralize_xml(str(x)) for x in review_focus if isinstance(x, str)
    )
    if not _NAME_RE.match(name):
        return None
    return CustomPersona(
        name=name,
        label=safe_label,
        description=safe_description,
        review_focus=safe_focus,
    )


def load_custom_personas() -> tuple[CustomPersona, ...]:
    """Return all user-defined personas, in insertion order.

    Fails open: missing file, empty file, corrupt TOML, or missing fields
    all produce an empty tuple rather than raising.
    """
    path = storage_path()
    if not path.is_file():
        return ()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return ()
    personas_block = data.get("personas")
    if not isinstance(personas_block, dict):
        return ()
    custom = personas_block.get("custom")
    if not isinstance(custom, list):
        return ()
    out: list[CustomPersona] = []
    for entry in custom:
        if isinstance(entry, dict):
            coerced = _coerce_entry(entry)
            if coerced is not None:
                out.append(coerced)
    return tuple(out)


def _write_all(personas: list[CustomPersona], path: Path) -> None:
    """Serialize the list atomically via tmp-file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "personas": {
            "custom": [
                {
                    "name": p.name,
                    "label": p.label,
                    "description": p.description,
                    "review_focus": list(p.review_focus),
                }
                for p in personas
            ]
        }
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(tomli_w.dumps(payload).encode("utf-8"))
    tmp.replace(path)


def save_custom_persona(persona: CustomPersona) -> None:
    """Add or replace ``persona`` in storage (by ``name``).

    Validates, sanitizes the description, then atomically writes the new list.
    """
    _validate(persona)
    safe_persona = CustomPersona(
        name=persona.name,
        label=_neutralize_xml(persona.label),
        description=_sanitize_description(persona.description),
        review_focus=tuple(_neutralize_xml(x) for x in persona.review_focus),
    )
    existing = list(load_custom_personas())
    replaced = False
    for i, p in enumerate(existing):
        if p.name == safe_persona.name:
            existing[i] = safe_persona
            replaced = True
            break
    if not replaced:
        existing.append(safe_persona)
    _write_all(existing, storage_path())


def delete_custom_persona(name: str) -> bool:
    """Remove the persona with ``name`` if present. Returns True on removal."""
    existing = list(load_custom_personas())
    remaining = [p for p in existing if p.name != name]
    if len(remaining) == len(existing):
        return False
    _write_all(remaining, storage_path())
    return True


def find_custom_persona(name: str) -> CustomPersona | None:
    for p in load_custom_personas():
        if p.name == name:
            return p
    return None


# ---------------------------------------------------------------------------
# Render — produce an LLM-ready PERSONA block
# ---------------------------------------------------------------------------


def render_custom_persona_prompt(persona: CustomPersona) -> str:
    """Render the persona for injection into a phase prompt.

    Defense in depth: even though save+load sanitize the description, we
    neutralize once more during render so a misconfigured caller that
    constructs a ``CustomPersona`` in memory cannot inject tags.
    """
    safe_label = _neutralize_xml(persona.label)
    safe_description = _neutralize_xml(persona.description)
    focus_clean = tuple(_neutralize_xml(x) for x in persona.review_focus)
    lines = [
        f"PERSONA: {safe_label}",
        f"Description: {safe_description}",
    ]
    if focus_clean:
        lines.append("Focus: " + ", ".join(focus_clean))
    return "\n".join(lines)


__all__ = [
    "CustomPersona",
    "InvalidPersonaError",
    "delete_custom_persona",
    "find_custom_persona",
    "load_custom_personas",
    "render_custom_persona_prompt",
    "save_custom_persona",
    "storage_path",
]
