"""CS2 team identity helpers — alias generation + normalization.

Markets name teams inconsistently ("NAVI", "Natus Vincere", "Na'Vi"). A small,
deterministic alias set per team makes match↔market matching robust without a
fragile fuzzy-string dependency. Shared by the data adapters and the matcher.
"""

from __future__ import annotations

import re

# Hand-curated extra aliases for teams whose market names diverge from their
# PandaScore name/acronym. Keyed by uppercased acronym. Extend as needed.
_KNOWN_ALIASES: dict[str, list[str]] = {
    "NAVI": ["Natus Vincere", "Na'Vi", "NaVi"],
    "FAZE": ["FaZe Clan", "FaZe"],
    "VIT": ["Team Vitality", "Vitality"],
    "G2": ["G2 Esports"],
    "SPIRIT": ["Team Spirit"],
    "MOUZ": ["mousesports", "MOUZ"],
    "VP": ["Virtus.pro", "Virtus Pro", "VirtusPro"],
    "LIQUID": ["Team Liquid"],
    "C9": ["Cloud9"],
    "EG": ["Evil Geniuses"],
    "FNATIC": ["fnatic"],
    "ASTRALIS": ["Astralis"],
    "HEROIC": ["Heroic"],
    "COMPLEXITY": ["Complexity", "coL"],
}

_PREFIXES = ("team ", "the ")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    """Lowercase, strip punctuation and common prefixes, collapse whitespace.

    "Na'Vi" -> "navi";  "Team Vitality" -> "vitality";  "G2 Esports" -> "g2 esports".
    Used both to build alias keys and to compare market titles to team names.
    """
    if not text:
        return ""
    s = text.lower().strip()
    for p in _PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    s = _PUNCT.sub("", s)  # drop intra-word punctuation: "Na'Vi" -> "navi"
    return _WS.sub(" ", s).strip()


def build_aliases(name: str | None, acronym: str | None) -> list[str]:
    """Return a de-duplicated, display-cased alias list for a team.

    Includes the full name, the acronym, a "Team "-stripped variant, and any
    curated known aliases. Order is stable (insertion order) for testability.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if not value:
            return
        key = normalize(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value.strip())

    add(name)
    add(acronym)
    if name:
        low = name.lower()
        for p in _PREFIXES:
            if low.startswith(p):
                add(name[len(p):])
    for extra in _KNOWN_ALIASES.get((acronym or "").upper(), []):
        add(extra)

    return out


def alias_keys(aliases: list[str]) -> set[str]:
    """Normalized comparison keys for an alias list (used by the matcher)."""
    return {k for k in (normalize(a) for a in aliases) if k}
