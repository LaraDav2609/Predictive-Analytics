"""LLM-based feature extractor for unstructured race signals.

Inputs (text streams):
  - Team radio transcripts (driver complaints, engineer instructions, pit prep)
  - FIA steward bulletins (incidents, investigations, penalties)
  - News / pre-race press (driver health, parts shortage, setup direction)
  - Live commentary feed (during in-race trading)

Outputs (structured features per (driver, lap)):
  - penalty_risk_now ∈ [0,1]
  - driver_mood ∈ {-1: frustrated, 0: neutral, +1: confident}
  - mechanical_concern ∈ [0,1]   (boosts DNF hazard)
  - setup_direction ∈ {-1: more_drag, 0: balanced, +1: less_drag}

Implementation: Claude / GPT with a structured-output schema (JSON mode).
Cache aggressively — same bulletin shouldn't be re-extracted twice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SoftFeatures:
    driver_code: str
    timestamp: str
    penalty_risk_now: float
    driver_mood: int       # -1, 0, +1
    mechanical_concern: float
    setup_direction: int   # -1, 0, +1
    raw_excerpt: str       # for audit


class LLMExtractor:
    def __init__(self, model: str = "claude-opus-4-7", schema_version: str = "v1") -> None:
        self.model = model
        self.schema_version = schema_version

    def extract_radio(self, transcript: str, driver_code: str, race_id: str) -> list[SoftFeatures]:
        raise NotImplementedError("structured output via JSON schema; cache by transcript hash")

    def extract_bulletin(self, bulletin_text: str, race_id: str) -> list[SoftFeatures]:
        raise NotImplementedError

    def extract_news(self, article: str, race_id: str) -> list[SoftFeatures]:
        raise NotImplementedError
