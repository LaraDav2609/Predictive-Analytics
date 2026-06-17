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
    model: str = typer.Option("uniform", help="Model to evaluate: uniform, production_v1, ml_simulator_v1, or all."),
    training_seasons: str = typer.Option(
        "",
        help="Comma-separated training seasons (e.g. '2022,2023'). Default: all seasons before --season.",
    ),
    min_training_races: int = typer.Option(20, help="Skip a test race if fewer training races available."),
    refit_every_n_races: int = typer.Option(1, help="Refit cadence for walk-forward harness metadata."),
    output_dir: str = typer.Option("artifacts/backtest", help="Directory to write per-race + aggregate CSVs."),
    artifact_dir: str = typer.Option("artifacts/backtest/ml_simulator_v1", help="Directory for generated ML simulator artifact bundles."),
    n_iterations: int = typer.Option(400, help="Monte Carlo iterations per race for ml_simulator_v1."),
    physical: bool = typer.Option(False, "--physical/--no-physical", help="Enable physical simulator mode for ml_simulator_v1."),
    fastf1_cache: str = typer.Option(".fastf1_cache", help="FastF1 cache directory."),
) -> None:
    """Walk-forward backtest over a full season.

    Default uses a synthetic provider — quick smoke run without network. Pass
    `--provider fastf1` to use historical telemetry (slow first time).
    """
    from sports.f1.ml.backtest.walk_forward import MLBacktestConfig, expand_model_choice, run_backtest

    if training_seasons:
        train_seasons = tuple(int(s.strip()) for s in training_seasons.split(",") if s.strip())
    else:
        train_seasons = tuple(range(season - 3, season))

    try:
        models = expand_model_choice(model)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        result = run_backtest(
            MLBacktestConfig(
                season=season,
                provider=provider,
                models=models,
                training_seasons=train_seasons,
                min_training_races=min_training_races,
                refit_every_n_races=refit_every_n_races,
                output_dir=output_dir,
                artifact_dir=artifact_dir,
                n_iterations=n_iterations,
                physical=physical,
                fastf1_cache=fastf1_cache,
            )
        )
    except Exception as exc:
        if provider == "fastf1":
            raise typer.BadParameter(f"fastf1 backtest unavailable: {exc}") from exc
        raise

    typer.echo(f"\nWalk-forward backtest done.")
    for label, path in result.output_paths.items():
        typer.echo(f"  {label:22s} {path}")
    typer.echo(f"  best model: {result.model_comparison.get('best_model') or '(none)'}")
    for model_id, metrics in (result.aggregate_metrics.get("models") or {}).items():
        typer.echo(
            f"    {model_id:18s} "
            f"winner_acc={metrics.get('winner_accuracy', 0.0):.3f} "
            f"brier={metrics.get('winner_brier', 0.0):.4f} "
            f"log_loss={metrics.get('winner_log_loss', 0.0):.4f}"
        )


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
    artifact_path: str = typer.Option("", help="Optional F1 ML artifact bundle path."),
    no_models: bool = typer.Option(False, "--no-models", help="Use raw initial-state tables and ignore artifacts."),
) -> None:
    """One-shot prediction for a single race.

    Default (synthetic): controlled fake race; no network.
    `--provider fastf1`: downloads the previous season's race at the same track,
    fits per-driver pace and per-compound tire deg, and runs the simulator on it.
    """
    from datetime import datetime, timezone

    from sports.f1.ml.bridge.redis_publisher import InMemoryPublisher, RedisPublisher
    from sports.f1.ml.common.types import RaceOutcomeProbability
    from sports.f1.ml.markets.mapper import (
        dnf_probabilities,
        fastest_lap_probabilities,
        podium_probabilities,
        winner_probabilities,
    )
    from sports.f1.ml.artifacts.loader import apply_ml_trained_inputs_to_initial_state, load_ml_trained_inputs, load_simulator_model_bundle
    from sports.f1.ml.simulator.race_sim import PhysicalParams, SimConfig, simulate_race

    sim_config_kwargs = {"n_iterations": n_iterations, "seed": 42}
    artifact_mode = "raw_initial_state"
    artifact_meta = {"applied_rows": 0, "artifact_id": None, "artifact_version": None, "fallback_reason": None}
    simulator_models = {}

    if provider == "synthetic":
        from sports.f1.ml.providers.synthetic_provider import SyntheticProvider, SyntheticRaceConfig

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
        from sports.f1.ml.features.initial_state import build_initial_state_from_provider, parse_race_id
        from sports.f1.ml.providers.fastf1_provider import FastF1Provider

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
            from sports.f1.ml.common.types import Race
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

    if no_models:
        artifact_mode = "no_models"
        typer.echo("artifact mode: no-models; using raw initial-state tables")
    elif artifact_path:
        trained_inputs = load_ml_trained_inputs(artifact_path, target_race=race)
        initial_state, artifact_meta = apply_ml_trained_inputs_to_initial_state(initial_state, trained_inputs)
        model_bundle = load_simulator_model_bundle(artifact_path, target_race=race)
        simulator_models = {"simulator_model_bundle": model_bundle}
        if artifact_meta["applied_rows"]:
            artifact_mode = "artifact"
            typer.echo(
                f"artifact mode: {artifact_meta['artifact_id'] or 'artifact'} "
                f"applied rows={artifact_meta['applied_rows']}"
            )
        else:
            artifact_mode = "artifact_fallback"
            typer.echo(f"artifact mode: fallback ({artifact_meta.get('fallback_reason') or 'no usable rows'})")
        model_summary = model_bundle.source_summary()
        typer.echo(
            "model contract: "
            f"{'used' if model_summary.get('ml_model_contract_used') else 'fallback'}; "
            f"adapters={','.join(model_summary.get('ml_model_adapters_used') or []) or 'none'}; "
            f"pace={model_summary.get('pace_adapter_source') or '-'}; "
            f"dnf={model_summary.get('dnf_adapter_source') or '-'}; "
            f"rating={model_summary.get('rating_adapter_source') or '-'}; "
            f"fallback={model_summary.get('ml_model_fallback_reason') or '-'}"
        )
    else:
        typer.echo("artifact mode: raw initial-state tables")

    typer.echo(f"running {n_iterations:,} MC iterations × {initial_state['total_laps']} laps × {len(drivers)} drivers")
    sim_result = simulate_race(
        race=race_obj,
        initial_state=initial_state,
        models=simulator_models,
        config=SimConfig(**sim_config_kwargs),
    )
    if sim_result.metadata.get("ml_model_contract_used"):
        typer.echo(f"model contract runtime: {sim_result.metadata.get('model_contract_adapter_counts')}")

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
                model_version=f"thin-slice-v0+{artifact_mode}",
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


@app.command("train-artifacts")
def train_artifacts(
    season_start: int = typer.Option(2026, help="First season included in artifact metadata."),
    season_end: int = typer.Option(2026, help="Last season included in artifact metadata."),
    output: str = typer.Option("artifacts/f1_ml/synthetic_artifact.json", help="Artifact JSON output path."),
    provider: str = typer.Option("synthetic", help="Training data source. Default path is synthetic."),
    fastf1_cache: str = typer.Option(".fastf1_cache", help="FastF1 cache directory."),
) -> None:
    """Build a lightweight F1 ML artifact bundle."""
    from sports.f1.ml.artifacts.builder import build_fastf1_artifact_bundle, build_synthetic_artifact_bundle
    from sports.f1.ml.artifacts.store import save_artifact_bundle, validate_artifact_bundle

    provider_key = provider.strip().lower()
    if provider_key == "synthetic":
        from sports.f1.ml.providers.synthetic_provider import SyntheticProvider, SyntheticRaceConfig

        synthetic = SyntheticProvider(SyntheticRaceConfig(season=season_end))
        bundle = build_synthetic_artifact_bundle(
            synthetic,
            artifact_id=f"synthetic-{season_start}-{season_end}-ml-simulator",
            season_start=season_start,
            season_end=season_end,
        )
    elif provider_key == "fastf1":
        from sports.f1.ml.providers.fastf1_provider import FastF1Provider

        typer.echo("provider: fastf1 (network/cache-backed; first run may be slow)")
        ff1 = FastF1Provider(cache_dir=fastf1_cache)
        races = []
        for season in range(season_start, season_end + 1):
            races.extend(ff1.list_races(season))
        if not races:
            raise typer.BadParameter("FastF1 returned no races for the requested season window")
        bundle = build_fastf1_artifact_bundle(
            ff1,
            training_races=races,
            artifact_id=f"fastf1-{season_start}-{season_end}-ml-simulator",
            season_start=season_start,
            season_end=season_end,
        )
    else:
        raise typer.BadParameter("provider must be 'synthetic' or 'fastf1'")

    validation = validate_artifact_bundle(bundle)
    result = save_artifact_bundle(output, bundle)
    typer.echo(f"artifact written: {result['path']}")
    typer.echo(f"artifact id: {result['artifact_id']}")
    typer.echo(f"validation: {'ok' if validation.ok else validation.reason}")
    typer.echo(f"provider: {bundle.source_metadata.get('provider')}")
    typer.echo(f"coverage: {bundle.source_metadata.get('coverage_counts') or bundle.validation_metrics.get('coverage') or {}}")
    typer.echo(f"fallback groups: {bundle.source_metadata.get('fallback_groups') or []}")


@app.command()
def live(
    race: str = typer.Option(..., help="Race id. Synthetic smoke accepts SYN-2026-01 or synthetic_2026_r01."),
    season: int | None = typer.Option(None, help="Season override for synthetic/estimated live runs."),
    round_num: int | None = typer.Option(None, "--round", help="Round override for synthetic/estimated live runs."),
    session: str = typer.Option("race", help="Session: race, qualifying, sprint."),
    model_id: str = typer.Option("ml_simulator_v1", help="Model id to label and run."),
    artifact_path: str = typer.Option("", help="Optional F1 ML artifact bundle path."),
    source: str = typer.Option("auto", help="auto, openf1, fastf1-recorded, replay, or estimated."),
    recording_path: str = typer.Option("", help="Optional local FastF1 recording file."),
    poll_seconds: float = typer.Option(10.0, help="Polling interval for loop mode."),
    once: bool = typer.Option(False, "--once", help="Run one update and exit."),
    publish: bool = typer.Option(False, "--publish", help="Publish common OutcomeProbability records to Redis."),
    redis_url: str = typer.Option("redis://localhost:6379/0", help="Redis URL for --publish."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Use in-memory publisher and print payload."),
    n_iterations: int = typer.Option(800, help="Monte Carlo iterations per live refresh."),
    physical: bool = typer.Option(True, "--physical/--no-physical", help="Enable physical simulator layer."),
    lap: int | None = typer.Option(None, help="Replay lap override for deterministic smoke tests."),
) -> None:
    """v2 live mode — stream lap-by-lap updates to Redis."""
    from sports.f1.ml.live_runner import LiveRunnerConfig, payload_to_json, run_live_loop

    config = LiveRunnerConfig(
        race=race,
        season=season,
        round_num=round_num,
        session=session,
        model_id=model_id,
        artifact_path=artifact_path or None,
        source=source,
        recording_path=recording_path or None,
        poll_seconds=poll_seconds,
        once=once or dry_run,
        publish=publish,
        redis_url=redis_url,
        dry_run=dry_run,
        n_iterations=n_iterations,
        physical=physical,
        lap=lap,
    )
    def _print_tick(result) -> None:
        payload = result.payload
        top = (payload.get("probabilities") or [{}])[0]
        typer.echo(
            f"{payload.get('generated_at')} "
            f"race={payload.get('race_id')} source={payload.get('source_mode')} "
            f"conf={payload.get('confidence')} lap={payload.get('lap')} "
            f"top={top.get('driver_code')} {float(top.get('win_probability') or 0.0) * 100:.1f}% "
            f"records={(payload.get('publish_status') or {}).get('records', 0)}"
        )

    try:
        results = run_live_loop(config, on_result=None if config.once else _print_tick)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    latest = results[-1].payload if results else {}
    typer.echo(
        f"live runner: race={latest.get('race_id') or race} "
        f"source={latest.get('source_mode')} confidence={latest.get('confidence')} "
        f"records={(latest.get('publish_status') or {}).get('records', 0)}"
    )
    if dry_run or once:
        typer.echo(payload_to_json(latest))


@app.command()
def list_models() -> None:
    """List all model factories registered in the registry."""
    from common.ml.registry import list_registered

    # Force imports so all @register decorators run.
    import sports.f1.ml.ratings.hierarchical_bayes  # noqa: F401
    import sports.f1.ml.core.gbm_pace  # noqa: F401
    import sports.f1.ml.core.gbm_dnf  # noqa: F401
    import sports.f1.ml.core.gbm_overtake  # noqa: F401

    for name in list_registered():
        typer.echo(name)


if __name__ == "__main__":
    app()
