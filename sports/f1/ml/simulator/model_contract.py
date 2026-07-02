"""Safe model contract for the F1 Monte Carlo simulator.

The simulator can still run from raw initial-state arrays, but this module gives
learned providers a stable interface for lap-aware pace and DNF inputs.
Adapters are deliberately defensive: bad model payloads return ``None`` and add
fallback metadata instead of throwing through prediction endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class PaceDistribution:
    mean_seconds: float | None = None
    sigma_seconds: float | None = None
    quantiles: dict[float, float] = field(default_factory=dict)
    source: str = "fallback"
    confidence: float = 0.0


@dataclass
class AdapterResult:
    value: float | PaceDistribution | None
    source: str = "fallback"
    confidence: float = 0.0
    fallback_reason: str | None = None


@runtime_checkable
class PaceModelAdapter(Protocol):
    source: str
    confidence: float

    def pace_distribution(
        self,
        driver_code: str,
        lap: int,
        state: dict[str, Any],
        features: dict[str, Any],
    ) -> PaceDistribution | None:
        ...


@runtime_checkable
class DNFModelAdapter(Protocol):
    source: str
    confidence: float

    def dnf_hazard(
        self,
        driver_code: str,
        lap: int,
        state: dict[str, Any],
        features: dict[str, Any],
    ) -> float | None:
        ...


@runtime_checkable
class SurvivalModelAdapter(DNFModelAdapter, Protocol):
    ...


@runtime_checkable
class RatingPriorAdapter(Protocol):
    source: str
    confidence: float

    def rating_prior(self, driver_code: str) -> float | None:
        ...


@dataclass
class SimulatorModelBundle:
    pace_adapter: PaceModelAdapter | None = None
    dnf_adapter: DNFModelAdapter | None = None
    survival_adapter: SurvivalModelAdapter | None = None
    rating_adapter: RatingPriorAdapter | None = None
    artifact_id: str | None = None
    artifact_version: str | None = None
    source: str = "model_contract"
    confidence: float = 0.0
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime: "SimulatorModelRuntime" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.runtime = SimulatorModelRuntime(self)

    def adapters_used(self) -> list[str]:
        used = []
        if self.pace_adapter is not None:
            used.append("pace")
        if self.dnf_adapter is not None:
            used.append("dnf")
        if self.survival_adapter is not None:
            used.append("survival")
        if self.rating_adapter is not None:
            used.append("rating")
        return used

    def source_summary(self) -> dict[str, Any]:
        return {
            "ml_model_contract_used": bool(self.adapters_used()),
            "ml_model_adapters_used": self.adapters_used(),
            "ml_model_fallback_reason": self.fallback_reason,
            "ml_artifact_id": self.artifact_id,
            "ml_artifact_version": self.artifact_version,
            "pace_adapter_source": getattr(self.pace_adapter, "source", None),
            "dnf_adapter_source": getattr(self.dnf_adapter or self.survival_adapter, "source", None),
            "rating_adapter_source": getattr(self.rating_adapter, "source", None),
            "ml_model_contract_confidence": self.confidence,
            "ml_artifact_readiness": self.metadata.get("artifact_readiness") or {},
        }


class StaticDriverPaceAdapter:
    def __init__(self, rows: dict[str, dict[str, Any]], *, source: str = "artifact_static_pace", confidence: float = 0.70):
        self.rows = rows
        self.source = source
        self.confidence = confidence

    def pace_distribution(self, driver_code: str, lap: int, state: dict[str, Any], features: dict[str, Any]) -> PaceDistribution | None:
        row = self.rows.get(driver_code) or {}
        mean = _float(row.get("pace_mean_seconds"))
        sigma = _float(row.get("pace_sigma_seconds"))
        if mean is None and sigma is None:
            return None
        return PaceDistribution(
            mean_seconds=mean,
            sigma_seconds=sigma,
            source=self.source,
            confidence=_float(row.get("confidence")) or self.confidence,
        )


class StaticDNFAdapter:
    def __init__(self, rows: dict[str, dict[str, Any]], *, source: str = "artifact_static_dnf", confidence: float = 0.70):
        self.rows = rows
        self.source = source
        self.confidence = confidence

    def dnf_hazard(self, driver_code: str, lap: int, state: dict[str, Any], features: dict[str, Any]) -> float | None:
        row = self.rows.get(driver_code) or {}
        value = _float(row.get("dnf_hazard_per_lap") or row.get("dnf_rate_per_lap"))
        if value is None:
            return None
        return max(0.0, min(0.25, value))


class StaticRatingPriorAdapter:
    def __init__(self, priors: dict[str, float], *, source: str = "artifact_rating_prior", confidence: float = 0.65):
        self.priors = priors
        self.source = source
        self.confidence = confidence

    def rating_prior(self, driver_code: str) -> float | None:
        value = _float(self.priors.get(driver_code))
        if value is None:
            return None
        return max(0.0, min(1.0, value))


class ObjectPaceModelAdapter:
    """Adapter for in-memory fitted objects with predict/predict_with_quantiles."""

    def __init__(self, model: Any, feature_rows: dict[str, dict[str, Any]], *, source: str = "object_pace_model", confidence: float = 0.74):
        self.model = model
        self.feature_rows = feature_rows
        self.source = source
        self.confidence = confidence

    def pace_distribution(self, driver_code: str, lap: int, state: dict[str, Any], features: dict[str, Any]) -> PaceDistribution | None:
        row = {**(self.feature_rows.get(driver_code) or {}), "lap": lap}
        if not row:
            return None
        frame = _frame([row])
        if hasattr(self.model, "predict_with_quantiles"):
            quantiles_raw = self.model.predict_with_quantiles(frame, alphas=(0.1, 0.5, 0.9))[0]
            low, median, high = [float(value) for value in quantiles_raw]
            return PaceDistribution(
                mean_seconds=median,
                sigma_seconds=max(0.08, abs(high - low) / 2.563),
                quantiles={0.1: low, 0.5: median, 0.9: high},
                source=f"{self.source}:quantiles",
                confidence=self.confidence,
            )
        if hasattr(self.model, "predict"):
            prediction = self.model.predict(frame)[0]
            return PaceDistribution(mean_seconds=float(prediction), source=self.source, confidence=self.confidence)
        return None


class ObjectDNFModelAdapter:
    """Adapter for in-memory DNF/survival objects."""

    def __init__(self, model: Any, feature_rows: dict[str, dict[str, Any]], *, source: str = "object_dnf_model", confidence: float = 0.72):
        self.model = model
        self.feature_rows = feature_rows
        self.source = source
        self.confidence = confidence

    def dnf_hazard(self, driver_code: str, lap: int, state: dict[str, Any], features: dict[str, Any]) -> float | None:
        row = {**(self.feature_rows.get(driver_code) or {}), "lap": lap}
        if not row:
            return None
        frame = _frame([row])
        if hasattr(self.model, "hazard_per_lap"):
            values = self.model.hazard_per_lap(frame)
        elif hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(frame)
            values = [proba[0][1] if len(proba[0]) > 1 else proba[0][0]]
        elif callable(self.model):
            values = self.model(frame)
        else:
            return None
        return max(0.0, min(0.25, float(values[0])))


class SimulatorModelRuntime:
    def __init__(self, bundle: SimulatorModelBundle):
        self.bundle = bundle
        self.fallback_counts: dict[str, int] = {}
        self.adapter_counts: dict[str, int] = {}
        self.errors: list[str] = []

    def pace_arrays(
        self,
        driver_codes: list[str],
        lap: int,
        base_means: Any,
        base_sigmas: Any,
        state: dict[str, Any],
        features: dict[str, Any],
    ) -> tuple[Any, Any]:
        means = base_means.copy()
        sigmas = base_sigmas.copy()
        for index, code in enumerate(driver_codes):
            try:
                distribution = self.bundle.pace_adapter.pace_distribution(code, lap, state, features) if self.bundle.pace_adapter else None
                if distribution is None:
                    self._fallback("pace")
                    continue
                if distribution.mean_seconds is not None:
                    means[index] = float(distribution.mean_seconds)
                if distribution.sigma_seconds is not None:
                    sigmas[index] = max(0.08, min(3.0, float(distribution.sigma_seconds)))
                self._used("pace")
            except Exception as exc:
                self._error("pace", exc)
        return means, sigmas

    def dnf_array(
        self,
        driver_codes: list[str],
        lap: int,
        base_dnf: Any,
        state: dict[str, Any],
        features: dict[str, Any],
    ) -> Any:
        values = base_dnf.copy()
        adapter = self.bundle.survival_adapter or self.bundle.dnf_adapter
        for index, code in enumerate(driver_codes):
            try:
                hazard = adapter.dnf_hazard(code, lap, state, features) if adapter else None
                if hazard is None:
                    self._fallback("dnf")
                    continue
                values[index] = max(0.0, min(0.25, float(hazard)))
                self._used("dnf")
            except Exception as exc:
                self._error("dnf", exc)
        return values

    def apply_rating_priors(self, driver_codes: list[str], pace_means: Any, pace_sigmas: Any) -> tuple[Any, Any]:
        if self.bundle.rating_adapter is None:
            return pace_means, pace_sigmas
        means = pace_means.copy()
        sigmas = pace_sigmas.copy()
        for index, code in enumerate(driver_codes):
            try:
                prior = self.bundle.rating_adapter.rating_prior(code)
                if prior is None:
                    self._fallback("rating")
                    continue
                centered = float(prior) - 0.5
                means[index] -= centered * 0.55
                sigmas[index] = max(0.08, float(sigmas[index]) * (1.0 - min(0.18, abs(centered) * 0.20)))
                self._used("rating")
            except Exception as exc:
                self._error("rating", exc)
        return means, sigmas

    def metadata(self) -> dict[str, Any]:
        summary = self.bundle.source_summary()
        summary.update({
            "model_contract_adapter_counts": dict(self.adapter_counts),
            "model_contract_fallback_counts": dict(self.fallback_counts),
            "model_contract_errors": list(self.errors[:8]),
        })
        return summary

    def _used(self, adapter: str) -> None:
        self.adapter_counts[adapter] = self.adapter_counts.get(adapter, 0) + 1

    def _fallback(self, adapter: str) -> None:
        self.fallback_counts[adapter] = self.fallback_counts.get(adapter, 0) + 1

    def _error(self, adapter: str, exc: Exception) -> None:
        self._fallback(adapter)
        if len(self.errors) < 8:
            self.errors.append(f"{adapter}:{exc.__class__.__name__}")


def coerce_simulator_model_bundle(models: dict[str, Any] | SimulatorModelBundle | None) -> SimulatorModelBundle | None:
    if isinstance(models, SimulatorModelBundle):
        return models
    if not isinstance(models, dict):
        return None
    for key in ("simulator_model_bundle", "model_bundle", "bundle"):
        value = models.get(key)
        if isinstance(value, SimulatorModelBundle):
            return value
    return None


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _frame(rows: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:
        return rows
