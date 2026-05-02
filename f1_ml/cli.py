"""f1-ml CLI — entry points for ingestion, training, backtesting, prediction.

Examples:
    f1-ml backtest --season 2024 --provider fastf1
    f1-ml predict --race 2026-emilia-romagna --pre-race
    f1-ml live --race 2026-emilia-romagna   # v2 mode
"""

from __future__ import annotations

import typer

app = typer.Typer(help="Formula 1 race prediction & trading ML CLI.")


@app.command()
def backtest(
    season: int = typer.Option(..., help="Season to test on; trains on all earlier seasons."),
    provider: str = typer.Option("synthetic", help="TelemetryProvider name. 'synthetic' or 'fastf1'."),
    training_seasons: str = typer.Option(
        "",
        help="Comma-separated training seasons (e.g. '2022,2023'). Default: all seasons before --season.",
    ),
    min_training_races: int = typer.Option(20, help="Skip a test race if fewer training races available."),
    output_dir: str = typer.Option("artifacts/backtest", help="Directory to write per-race + aggregate CSVs."),
) -> None:
    """Walk-forward backtest over a full season.

    Default uses a synthetic provider — quick smoke run without network. Pass
    `--provider fastf1` to use historical telemetry (slow first time).
    """
    import json
    import os
    from datetime import datetime, timezone

    from f1_ml.backtest.walk_forward import RaceData, WalkForwardConfig, run as run_walk_forward

    if training_seasons:
        train_seasons = tuple(int(s.strip()) for s in training_seasons.split(",") if s.strip())
    else:
        train_seasons = tuple(range(season - 3, season + 1))

    if provider == "synthetic":
        # Smoke-test path: synthesize a few races so the harness exercises end-to-end
        # without any external dependencies.
        races: list[RaceData] = []
        all_drivers = ["VER", "HAM", "LEC", "NOR", "PIA", "RUS", "ALO", "SAI"]
        for s in (*train_seasons, season):
            for r in range(1, 11):
                races.append(RaceData(
                    race_id=f"{s}-{r:02d}-SYN",
                    season=s, round=r,
                    decision_time=datetime(s, max(1, min(12, r)), 1, 12, 0, tzinfo=timezone.utc),
                    finish_order=all_drivers.copy(),
                ))

        class _UniformPredictor:
            """Equal-probability baseline; useful as a calibration floor."""
            def predict(self, race: RaceData) -> dict[str, dict[str, float]]:
                n = len(race.finish_order)
                return {
                    "winner": {d: 1.0 / n for d in race.finish_order},
                    "podium": {d: 3.0 / n for d in race.finish_order},
                }

        result = run_walk_forward(
            WalkForwardConfig(
                test_season=season,
                training_seasons=train_seasons,
                min_training_races=min_training_races,
            ),
            races,
            model_factory=lambda _train: _UniformPredictor(),
        )
    else:
        raise typer.BadParameter(
            f"provider '{provider}' is not yet wired into backtest; only 'synthetic' is supported"
        )

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"per_race_metrics_{season}.csv")
    aggregate_path = os.path.join(output_dir, f"aggregate_metrics_{season}.json")
    result.per_race_metrics.to_csv(metrics_path, index=False)
    with open(aggregate_path, "w") as f:
        json.dump(result.aggregate_metrics, f, indent=2)

    typer.echo(f"\nWalk-forward backtest done.")
    typer.echo(f"  per-race metrics: {metrics_path}")
    typer.echo(f"  aggregate:        {aggregate_path}")
    if result.aggregate_metrics:
        for k, v in result.aggregate_metrics.items():
            typer.echo(f"    {k:35s} {v:.4f}")


@app.command()
def predict(
    race: str = typer.Option("SYN-2026-01", help="Race id. Synthetic: 'SYN-2026-01'. Real: 'YYYY-RR-TRACK' e.g. '2024-14-MONZA'."),
    provider: str = typer.Option("synthetic", help="Data source: 'synthetic' (no network) or 'fastf1' (downloads historical race)."),
    total_laps: int = typer.Option(0, help="When provider=fastf1: total race laps (0 = use prior race's lap count)."),
    pre_race: bool = typer.Option(True, help="Pre-race-only inputs (v1 mode)"),
    publish: bool = typer.Option(False, help="Publish probability snapshots to Redis"),
    n_iterations: int = typer.Option(20_000, help="Monte Carlo iterations"),
    redis_host: str = typer.Option("localhost"),
    redis_port: int = typer.Option(6379),
    physical: bool = typer.Option(False, "--physical/--no-physical", help="Enable physical mode (fuel + tire deg + dirty air + pit)"),
    pit_lap: int = typer.Option(0, help="When --physical: pit lap (0 = no pit)"),
    fastf1_cache: str = typer.Option(".fastf1_cache", help="FastF1 cache directory."),
) -> None:
    """One-shot prediction for a single race.

    Default (synthetic): controlled fake race; no network.
    `--provider fastf1`: downloads the previous season's race at the same track,
    fits per-driver pace and per-compound tire deg, and runs the simulator on it.
    """
    from datetime import datetime, timezone

    from f1_ml.bridge.redis_publisher import InMemoryPublisher, RedisPublisher
    from f1_ml.common.types import RaceOutcomeProbability
    from f1_ml.markets.mapper import (
        dnf_probabilities,
        fastest_lap_probabilities,
        podium_probabilities,
        winner_probabilities,
    )
    from f1_ml.simulator.race_sim import PhysicalParams, SimConfig, simulate_race

    sim_config_kwargs = {"n_iterations": n_iterations, "seed": 42}

    if provider == "synthetic":
        from f1_ml.providers.synthetic_provider import SyntheticProvider, SyntheticRaceConfig

        if not race.startswith("SYN-"):
            raise typer.BadParameter("synthetic provider expects a 'SYN-...' race id")
        prov = SyntheticProvider(SyntheticRaceConfig(race_id=race))
        pace_table = prov.driver_pace_table()
        sigma_table = prov.driver_sigma_table()
        dnf_table = prov.dnf_rate_table()
        drivers = list(pace_table.keys())

        initial_state = {
            "driver_codes": drivers,
            "driver_mean_pace_s": [pace_table[d] for d in drivers],
            "driver_pace_sigma_s": [sigma_table[d] for d in drivers],
            "driver_dnf_rate_per_lap": [dnf_table[d] for d in drivers],
            "total_laps": prov.config.n_laps,
        }
        race_obj = prov.list_races(prov.config.season)[0]

        if physical:
            initial_state["driver_starting_compound"] = ["MEDIUM"] * len(drivers)
            initial_state["driver_pit_lap"] = [pit_lap if pit_lap > 0 else None] * len(drivers)
            initial_state["driver_pit_compound"] = ["HARD" if pit_lap > 0 else None] * len(drivers)

    elif provider == "fastf1":
        from f1_ml.features.initial_state import build_initial_state_from_provider, parse_race_id
        from f1_ml.providers.fastf1_provider import FastF1Provider

        prov = FastF1Provider(cache_dir=fastf1_cache)
        parsed = parse_race_id(race)

        # If user didn't specify total_laps, use the prior race's actual race-distance.
        # Default to 50 if we can't determine it.
        target_total_laps = total_laps if total_laps > 0 else 50
        typer.echo(f"loading FastF1 history for {race} (this may take 10-30 s on first run)…")
        build = build_initial_state_from_provider(
            prov,
            race_id=race,
            total_laps=target_total_laps,
            # Caller-forced pit_lap when set (>0); otherwise DP solves per-driver.
            pit_lap=pit_lap if (physical and pit_lap > 0) else None,
            optimize_strategy=physical,  # only meaningful when physical=True
        )
        initial_state = build.initial_state
        drivers = initial_state["driver_codes"]
        race_obj = next(
            (r for r in prov.list_races(parsed.season) if r.round == parsed.round and r.track_code == parsed.track_code),
            None,
        )
        if race_obj is None:
            from f1_ml.common.types import Race
            race_obj = Race(season=parsed.season, round=parsed.round, track_code=parsed.track_code,
                            name=f"R{parsed.round} {parsed.track_code}", scheduled_start=datetime(parsed.season, 6, 1))

        diag = build.diagnostics
        typer.echo(f"  history: {', '.join(diag['history_races_used']) or '(none)'}")
        typer.echo(f"  drivers: {diag['n_drivers_loaded']}, laps in sample: {diag['n_historical_laps']}")
        typer.echo(f"  median pace: {diag['median_lap_seconds']:.3f} s")
        if diag["compound_deg_slopes"]:
            slopes = ", ".join(f"{c}={v:+.3f}" for c, v in diag["compound_deg_slopes"].items())
            typer.echo(f"  fitted tire deg (s/lap): {slopes}")
        if diag.get("strategy_optimized") and diag.get("per_driver_strategy"):
            # Aggregate the per-driver DP picks into "X drivers pit at lap L on COMPOUND".
            from collections import Counter
            picks = Counter(
                (s["pit_lap"], s["pit_compound"]) for s in diag["per_driver_strategy"]
            )
            summary = "; ".join(
                f"{count}x pit L{lap}->{comp}" for (lap, comp), count in picks.most_common()
            )
            typer.echo(f"  DP strategy: {summary}")

        if physical and build.deg_fits:
            # Override the simulator's default per-compound deg with the freshly-fit values.
            params = PhysicalParams()
            for compound, fit in build.deg_fits.items():
                if compound in params.deg_per_lap_per_compound:
                    params.deg_per_lap_per_compound[compound] = max(0.0, fit.deg_per_lap_s)
            sim_config_kwargs["physical"] = params
    else:
        raise typer.BadParameter(f"unknown provider '{provider}'; use 'synthetic' or 'fastf1'")

    if physical:
        sim_config_kwargs["enable_physical"] = True
        sim_config_kwargs.setdefault("physical", PhysicalParams())
        typer.echo(f"physical mode: pit_lap={pit_lap or 'none'}, fuel + tire deg + dirty air enabled")

    typer.echo(f"running {n_iterations:,} MC iterations × {initial_state['total_laps']} laps × {len(drivers)} drivers")
    sim_result = simulate_race(
        race=race_obj,
        initial_state=initial_state,
        models={},
        config=SimConfig(**sim_config_kwargs),
    )

    winner = winner_probabilities(sim_result)
    podium = podium_probabilities(sim_result)
    fl = fastest_lap_probabilities(sim_result)
    dnf = dnf_probabilities(sim_result)

    # Pretty top-5 winners
    top_5 = sorted(winner.items(), key=lambda kv: -kv[1])[:5]
    typer.echo("\nTop 5 winner probabilities:")
    for drv, p in top_5:
        typer.echo(f"  {drv:8s}  {p * 100:5.1f}%   podium {podium[drv] * 100:5.1f}%   FL {fl[drv] * 100:4.1f}%   DNF {dnf[drv] * 100:4.1f}%")

    # Build RaceOutcomeProbability records and publish.
    now = datetime.now(timezone.utc)
    records: list[RaceOutcomeProbability] = []
    for drv in drivers:
        for market, prob in (("winner", winner[drv]), ("podium", podium[drv]), ("fastest_lap", fl[drv]), ("dnf", dnf[drv])):
            records.append(RaceOutcomeProbability(
                race_id=race,
                driver_code=drv,
                market=market,
                probability=prob,
                knowable_as_of=now,
                model_version="thin-slice-v0",
            ))

    if publish:
        publisher = RedisPublisher(host=redis_host, port=redis_port)
        publisher.publish_batch(records)
        typer.echo(f"\npublished {len(records)} records to redis @ {redis_host}:{redis_port}")
    else:
        # Always run through InMemoryPublisher so we exercise the publisher path.
        mem = InMemoryPublisher()
        mem.publish_batch(records)
        typer.echo(f"\nemitted {len(records)} records (in-memory; pass --publish to push to redis)")


@app.command()
def live(
    race: str = typer.Option(..., help="Race id"),
    redis_host: str = typer.Option("localhost"),
) -> None:
    """v2 live mode — stream lap-by-lap updates to Redis."""
    typer.echo(f"[stub] live race={race}")
    raise NotImplementedError


@app.command()
def list_models() -> None:
    """List all model factories registered in the registry."""
    from f1_ml.common.registry import list_registered

    # Force imports so all @register decorators run.
    import f1_ml.ratings.hierarchical_bayes  # noqa: F401
    import f1_ml.core.gbm_pace  # noqa: F401
    import f1_ml.core.gbm_dnf  # noqa: F401
    import f1_ml.core.gbm_overtake  # noqa: F401

    for name in list_registered():
        typer.echo(name)


if __name__ == "__main__":
    app()
