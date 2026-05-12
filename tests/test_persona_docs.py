# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Tests for the persona value-education catalog."""

from __future__ import annotations


def test_builtin_catalog_has_one_entry_per_persona_level() -> None:
    """Every LEVEL_0..LEVEL_3 has a corresponding catalog entry."""
    from agent_amplifier.persona_docs import BUILTIN_PERSONA_DOCS
    from agent_amplifier.personas import PERSONA_LADDER

    assert len(BUILTIN_PERSONA_DOCS) == len(PERSONA_LADDER)


def test_each_builtin_entry_has_required_fields() -> None:
    import dataclasses

    from agent_amplifier.persona_docs import BUILTIN_PERSONA_DOCS

    required = {"slug", "label", "value_tagline", "when_to_use"}
    for entry in BUILTIN_PERSONA_DOCS:
        present = {f.name for f in dataclasses.fields(entry)}
        missing = required - present
        assert not missing, f"Entry {entry} is missing {missing}"


def test_builtin_slugs_are_unique_and_url_safe() -> None:
    import re

    from agent_amplifier.persona_docs import BUILTIN_PERSONA_DOCS

    slugs = [e.slug for e in BUILTIN_PERSONA_DOCS]
    assert len(slugs) == len(set(slugs)), "slugs are not unique"
    slug_re = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
    for slug in slugs:
        assert slug_re.match(slug), f"slug {slug!r} not URL-safe"


def test_value_taglines_are_user_friendly_short() -> None:
    """Tagline is a single sentence under ~150 chars (UI dropdown line)."""
    from agent_amplifier.persona_docs import BUILTIN_PERSONA_DOCS

    for entry in BUILTIN_PERSONA_DOCS:
        assert entry.value_tagline.strip(), entry.slug
        assert len(entry.value_tagline) < 200, entry.slug


def test_when_to_use_is_non_empty() -> None:
    from agent_amplifier.persona_docs import BUILTIN_PERSONA_DOCS

    for entry in BUILTIN_PERSONA_DOCS:
        assert entry.when_to_use.strip(), entry.slug


def test_get_builtin_doc_by_level() -> None:
    from agent_amplifier.persona_docs import get_builtin_doc

    doc = get_builtin_doc(0)
    assert doc.slug == "senior-engineer"
    doc3 = get_builtin_doc(3)
    assert "distinguished" in doc3.slug or "pre-launch" in doc3.slug


def test_get_builtin_doc_clamps_out_of_range() -> None:
    from agent_amplifier.persona_docs import get_builtin_doc

    high = get_builtin_doc(99)
    assert high.slug == get_builtin_doc(3).slug
    low = get_builtin_doc(-5)
    assert low.slug == get_builtin_doc(0).slug


def test_describe_builtin_persona_clamps_negative_level() -> None:
    from agent_amplifier.persona_docs import describe_builtin_persona

    desc = describe_builtin_persona(-5)
    assert desc["level"] == 0
    assert desc["slug"] == "senior-engineer"


def test_describe_builtin_persona_clamps_excess_level() -> None:
    from agent_amplifier.persona_docs import MAX_LEVEL_FOR_DOCS, describe_builtin_persona

    desc = describe_builtin_persona(MAX_LEVEL_FOR_DOCS + 99)
    assert desc["level"] == MAX_LEVEL_FOR_DOCS


def test_describe_persona_returns_combined_dict_for_builtin() -> None:
    from agent_amplifier.persona_docs import describe_builtin_persona

    desc = describe_builtin_persona(1)
    assert desc["slug"] == "security-paranoid-engineer"
    assert "OWASP" in desc["value_tagline"] or "OWASP" in desc["when_to_use"]
    assert desc["level"] == 1
    assert "role" in desc
    assert "focus" in desc
    assert isinstance(desc["focus"], list)


def test_describe_custom_persona_uses_user_supplied_text() -> None:
    from agent_amplifier.custom_personas import CustomPersona
    from agent_amplifier.persona_docs import describe_custom_persona

    p = CustomPersona("ml-eng", "ML Engineer", "PyTorch expert", ("pytorch",))
    desc = describe_custom_persona(p)
    assert desc["slug"] == "ml-eng"
    assert desc["label"] == "ML Engineer"
    assert desc["value_tagline"] == "PyTorch expert"
    assert desc["custom"] is True
    assert desc["focus"] == ["pytorch"]


def test_list_all_personas_returns_builtin_and_custom(
    monkeypatch: object,
    tmp_path: object,
) -> None:
    """Public summary endpoint that the UI/CLI/backend share."""
    import os

    from agent_amplifier.custom_personas import CustomPersona, save_custom_persona
    from agent_amplifier.persona_docs import BUILTIN_PERSONA_DOCS, list_all_personas

    target = str(tmp_path) + "/personas.toml"  # type: ignore[operator]
    os.environ["AGENT_AMP_PERSONAS_PATH"] = target
    try:
        save_custom_persona(CustomPersona("ml", "ML", "PyTorch", ("pytorch",)))
        all_personas = list_all_personas()
        assert len(all_personas) == len(BUILTIN_PERSONA_DOCS) + 1
        # Built-ins come first, then customs.
        assert all_personas[-1]["slug"] == "ml"
        assert all_personas[-1]["custom"] is True
        assert all_personas[0]["custom"] is False
    finally:
        del os.environ["AGENT_AMP_PERSONAS_PATH"]
