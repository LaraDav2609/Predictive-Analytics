"""Versioned F1 model weight configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from f1_predictor.config import MODEL_VERSION

PRODUCTION_MODEL_ID = "production_v1"


@dataclass(frozen=True)
class RaceModelConfig:
    model_id: str
    label: str
    model_version: str
    description: str
    race_weights: dict[str, float]
    track_fit_weights: dict[str, float]
    tire_strategy_weights: dict[str, float]
    wdc_future_weights: dict[str, float]
    wdc_standings_base: float
    wdc_standings_progress: float
    reliability_dnf_multiplier: float
    probability_temperature: float = 1.0

    def public_dict(self) -> dict:
        return {
            **asdict(self),
            "is_default": self.model_id == PRODUCTION_MODEL_ID,
        }


MODEL_CONFIGS: dict[str, RaceModelConfig] = {
    "production_v1": RaceModelConfig(
        model_id="production_v1",
        label="Production v1",
        model_version=MODEL_VERSION,
        description="Current production ensemble weights.",
        race_weights={
            "standing": 0.13,
            "form": 0.17,
            "team": 0.15,
            "performance": 0.18,
            "qualifying_pace": 0.10,
            "race_pace": 0.12,
            "track_fit": 0.08,
            "tire_strategy": 0.04,
            "weather_risk": 0.03,
        },
        track_fit_weights={
            "race_pace": 0.32,
            "qualifying_pace": 0.24,
            "track_history": 0.18,
            "teammate": 0.14,
            "overtaking": 0.12,
        },
        tire_strategy_weights={
            "race_pace": 0.30,
            "reliability": 0.20,
            "team": 0.18,
            "undercut": 0.16,
            "degradation": 0.10,
            "trend": 0.06,
        },
        wdc_future_weights={
            "team": 0.30,
            "form": 0.22,
            "race_pace": 0.16,
            "qualifying_pace": 0.12,
            "reliability": 0.10,
            "track_fit": 0.10,
        },
        wdc_standings_base=0.14,
        wdc_standings_progress=0.60,
        reliability_dnf_multiplier=0.75,
        probability_temperature=1.0,
    ),
    "calibrated_candidate_v1": RaceModelConfig(
        model_id="calibrated_candidate_v1",
        label="Calibrated candidate v1",
        model_version=f"{MODEL_VERSION}-calibrated-candidate-v1",
        description="Backtest-oriented candidate with more race pace, track fit, and reliability influence.",
        race_weights={
            "standing": 0.09,
            "form": 0.15,
            "team": 0.14,
            "performance": 0.17,
            "qualifying_pace": 0.09,
            "race_pace": 0.18,
            "track_fit": 0.10,
            "tire_strategy": 0.05,
            "weather_risk": 0.03,
        },
        track_fit_weights={
            "race_pace": 0.36,
            "qualifying_pace": 0.18,
            "track_history": 0.20,
            "teammate": 0.14,
            "overtaking": 0.12,
        },
        tire_strategy_weights={
            "race_pace": 0.34,
            "reliability": 0.22,
            "team": 0.14,
            "undercut": 0.14,
            "degradation": 0.10,
            "trend": 0.06,
        },
        wdc_future_weights={
            "team": 0.27,
            "form": 0.20,
            "race_pace": 0.20,
            "qualifying_pace": 0.10,
            "reliability": 0.12,
            "track_fit": 0.11,
        },
        wdc_standings_base=0.11,
        wdc_standings_progress=0.56,
        reliability_dnf_multiplier=0.82,
        probability_temperature=1.08,
    ),
    "conservative_v1": RaceModelConfig(
        model_id="conservative_v1",
        label="Conservative v1",
        model_version=f"{MODEL_VERSION}-conservative-v1",
        description="Flatter probability candidate with lower standings dominance and lower overconfidence risk.",
        race_weights={
            "standing": 0.08,
            "form": 0.16,
            "team": 0.15,
            "performance": 0.16,
            "qualifying_pace": 0.10,
            "race_pace": 0.16,
            "track_fit": 0.10,
            "tire_strategy": 0.05,
            "weather_risk": 0.04,
        },
        track_fit_weights={
            "race_pace": 0.34,
            "qualifying_pace": 0.20,
            "track_history": 0.18,
            "teammate": 0.16,
            "overtaking": 0.12,
        },
        tire_strategy_weights={
            "race_pace": 0.30,
            "reliability": 0.24,
            "team": 0.14,
            "undercut": 0.14,
            "degradation": 0.12,
            "trend": 0.06,
        },
        wdc_future_weights={
            "team": 0.26,
            "form": 0.20,
            "race_pace": 0.18,
            "qualifying_pace": 0.10,
            "reliability": 0.14,
            "track_fit": 0.12,
        },
        wdc_standings_base=0.09,
        wdc_standings_progress=0.52,
        reliability_dnf_multiplier=0.90,
        probability_temperature=1.18,
    ),
}


def get_model_config(model_id: str | None = None) -> RaceModelConfig:
    return MODEL_CONFIGS.get(model_id or PRODUCTION_MODEL_ID) or MODEL_CONFIGS[PRODUCTION_MODEL_ID]


def list_model_configs() -> list[dict]:
    return [config.public_dict() for config in MODEL_CONFIGS.values()]
