# f1_ml — Formula 1 race prediction & ML stack

Heavy ML stack for F1 race prediction. Lives alongside the lightweight per-sport
predictors in `analytics/`; integrated with the FastAPI server via the routes
in `api/f1_routes.py` (`/api/f1/predictions/{race_id}`).

The .NET dashboard (`Predictions.Dashboard` in the SportsPredictions C# solution)
proxies to those endpoints and renders the F1 Predictions sub-tab.

## Layout

```
f1_ml/
  common/         shared types, model registry, IO, calibration metrics
  providers/      TelemetryProvider abstraction (FastF1, vendor, synthetic)
  features/       pace estimation: mini-sector, tire deg, fuel, dirty air, Kalman, GP, initial_state
  ratings/        driver/team strength: Bradley-Terry, Plackett-Luce, TrueSkill, hierarchical Bayes
  events/         stochastic event models: DNF (Weibull/Cox/DeepSurv), safety car, pit, overtake
  core/           probability engine: GBM heads, conformal, deep ensemble, MC dropout
  sequence/       lap-by-lap models: LSTM, Transformer, GNN, online Bayesian
  strategy/       pit policy: DP baseline (working), offline RL (CQL/IQL), inverse RL
  multiplicative/ multi-task, self-supervised, stacking, mixture-of-experts
  soft_signals/   LLM extractor, causal inference (double ML, synthetic control)
  stretch/        CV tire wear, MAML meta-learning, diffusion counterfactuals
  simulator/      Monte Carlo race simulator (thin slice + physical mode)
  markets/        outcome distribution → market probabilities, edge, Kelly, slippage
  backtest/       walk-forward, in-race replay, look-ahead audit
  bridge/         Redis publisher (writes f1:snapshot:{race_id} hashes consumed by api/f1_routes.py)
```

## Quickstart

```bash
# From repo root (one-time setup):
python -m venv .venv
.venv/Scripts/activate    # Windows
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch  # CPU torch on Windows

# Run prediction (synthetic):
python -m f1_ml.cli predict

# Run prediction on a real historical race (downloads ~50MB on first run):
python -m f1_ml.cli predict --provider fastf1 --race 2024-16-MONZA --physical

# Publish to Redis (consumed by /api/f1/predictions/{race_id}):
python -m f1_ml.cli predict --provider fastf1 --race 2024-16-MONZA --physical --publish

# Start the existing API server (it now also serves f1_ml predictions):
python main.py

# Run tests:
pytest
```

## Working modules vs. stubs

**Working** (verified by tests):
- `providers/` — FastF1Provider, SyntheticProvider
- `features/` — fuel_correction, tire_degradation (linear / cliff / exponential), dirty_air, pace_kalman, knowable_as_of, initial_state
- `simulator/` — race_sim with thin-slice and physical modes
- `markets/` — mapper, kelly, edge
- `strategy/` — dp_optimal_stop
- `ratings/` — hierarchical_bayes (slow first run; tested)
- `bridge/` — redis_publisher
- `common/` — types, registry, calibration

**Scaffolded but stubbed** (clear interface, NotImplementedError bodies):
- Sequence models (LSTM, Transformer, GNN)
- Other rating systems (Bradley-Terry, Plackett-Luce, TrueSkill)
- Most event models (Weibull / Cox DNF, safety-car Poisson, pit time)
- Multiplicative wins (multi-task, self-supervised, stacking, MoE)
- Soft signals (LLM extractor, causal inference)
- Stretch (CV tire wear, MAML, diffusion)
- Backtest harness (walk-forward, in-race replay)
- RL pit strategy (CQL, IQL, inverse RL)

See each module's docstring for the technique and when to wire it up.
