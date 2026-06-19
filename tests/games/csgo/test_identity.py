"""Tests for CS2 team identity helpers (alias generation + normalization)."""

from games.csgo.identity import alias_keys, build_aliases, normalize


def test_normalize_strips_punct_and_prefix():
    assert normalize("Na'Vi") == "navi"
    assert normalize("Team Vitality") == "vitality"
    assert normalize("G2 Esports") == "g2 esports"
    assert normalize("  FaZe  Clan ") == "faze clan"
    assert normalize(None) == ""


def test_build_aliases_includes_name_acronym_and_known():
    keys = alias_keys(build_aliases("Natus Vincere", "NAVI"))
    assert "natus vincere" in keys
    assert "navi" in keys  # acronym + curated "Na'Vi" both normalize to navi


def test_build_aliases_dedupes_by_normalized_key():
    aliases = build_aliases("FaZe Clan", "FAZE")
    norm = [normalize(a) for a in aliases]
    assert len(norm) == len(set(norm))           # no duplicate keys
    assert {"faze clan", "faze"} <= set(norm)


def test_build_aliases_strips_team_prefix():
    keys = alias_keys(build_aliases("Team Spirit", "SPIRIT"))
    assert "spirit" in keys                       # "Team Spirit" and "SPIRIT" collapse to one key
