"""Tests for the model-vs-market (CLV) backtest: odds math + betting simulation,
plus the historical-odds → gamePk matcher (odds_ingest)."""
from datetime import datetime
from types import SimpleNamespace

from sports.baseball.analytics.clv import (
    american_to_decimal, american_to_prob, devig_pair, market_fair_home,
    run_clv_backtest,
)
from sports.baseball.analytics.odds_ingest import (
    normalize_team, load_and_match,
)


def _rec(gid, prob_home, actual):
    return SimpleNamespace(game_id=gid, prob_home=prob_home, actual=actual)


def test_american_to_prob():
    assert abs(american_to_prob(-110) - 0.5238) < 1e-3
    assert abs(american_to_prob(+100) - 0.5) < 1e-9
    assert abs(american_to_prob(+150) - 0.4) < 1e-9
    assert abs(american_to_prob(-200) - 0.6667) < 1e-3


def test_american_to_decimal():
    assert abs(american_to_decimal(-110) - 1.9091) < 1e-3
    assert abs(american_to_decimal(+100) - 2.0) < 1e-9
    assert abs(american_to_decimal(+150) - 2.5) < 1e-9


def test_devig_removes_the_overround():
    # Two -110 sides: each implies ~0.524, sum ~1.048 (4.8% vig) → fair 0.5/0.5.
    h, a = devig_pair(american_to_prob(-110), american_to_prob(-110))
    assert abs(h - 0.5) < 1e-6 and abs(a - 0.5) < 1e-6
    assert abs((h + a) - 1.0) < 1e-9


def test_market_fair_home_from_american_row():
    # Home favourite -150, away +130.
    p = market_fair_home({"home_ml": -150, "away_ml": +130})
    assert 0.5 < p < 0.65                       # de-vigged favourite, below raw implied


def test_clv_no_odds_is_graceful():
    res = run_clv_backtest([_rec(1, 0.6, 1.0)], {})
    assert res["available"] is False and res["matched"] == 0


def test_clv_model_beats_a_wrong_market_is_profitable():
    # Market prices every game 50/50 (+100/+100, decimal 2.0). Model says home 0.70
    # and home really wins 70% → model is sharper and flat bets clear a profit.
    odds = {}
    records = []
    for i in range(100):
        odds[i] = {"home_ml": +100, "away_ml": +100}
        records.append(_rec(i, 0.70, 1.0 if i < 70 else 0.0))
    res = run_clv_backtest(records, odds, edge_threshold_bps=200)
    assert res["available"] is True
    assert res["model_beats_market"] is True                 # 0.21 brier < 0.25 market
    assert res["betting"]["bets"] == 100                     # edge on home every game
    assert res["betting"]["flat_roi"] > 0                    # 70 win / 30 lose at 2.0 → +0.40
    assert res["gate"]["tradeable"] is True


def test_clv_model_equal_to_market_makes_no_bets():
    # Model == market fair prob → no exploitable edge, and it does not beat the market.
    odds, records = {}, []
    for i in range(60):
        odds[i] = {"home_ml": +100, "away_ml": +100}         # fair 0.5
        records.append(_rec(i, 0.50, 1.0 if i % 2 == 0 else 0.0))
    res = run_clv_backtest(records, odds, edge_threshold_bps=200)
    assert res["betting"]["bets"] == 0
    assert res["model_beats_market"] is False                # equal brier, strict <
    assert res["gate"]["tradeable"] is False


# ─────────────────────────── odds_ingest: team normalisation ───────────────────────────
def test_normalize_team_variants():
    for v in ("NY Yankees", "Yankees", "NYY", "New York Yankees", "new york yankees",
              "NYA"):
        assert normalize_team(v) == "new york yankees"
    assert normalize_team("Red Sox") == "boston red sox"
    assert normalize_team("BOS") == "boston red sox"
    assert normalize_team("St. Louis") == "st. louis cardinals"
    assert normalize_team("STL") == "st. louis cardinals"
    assert normalize_team("Cardinals") == "st. louis cardinals"
    assert normalize_team("Cleveland Indians") == "cleveland guardians"   # historical alias
    assert normalize_team("A's") == "oakland athletics"
    assert normalize_team("D-backs") == "arizona diamondbacks"
    assert normalize_team("LA Angels") == "los angeles angels"
    assert normalize_team("Chi White Sox") == "chicago white sox"


def test_normalize_team_full_statsapi_names_from_odds_file():
    # The benchmark odds file carries FULL StatsAPI names — the schedule side uses the
    # exact same strings, so both must land on the identical canonical key.
    cases = {
        "St. Louis Cardinals": "st. louis cardinals",
        "Cleveland Guardians": "cleveland guardians",
        "Oakland Athletics": "oakland athletics",
        "Los Angeles Angels": "los angeles angels",
        "Arizona Diamondbacks": "arizona diamondbacks",
        "Chicago White Sox": "chicago white sox",
        "Tampa Bay Rays": "tampa bay rays",
        "San Diego Padres": "san diego padres",
    }
    for full, canon in cases.items():
        assert normalize_team(full) == canon


def test_normalize_team_all_30_canonical():
    canon = {
        "arizona diamondbacks", "atlanta braves", "baltimore orioles", "boston red sox",
        "chicago cubs", "chicago white sox", "cincinnati reds", "cleveland guardians",
        "colorado rockies", "detroit tigers", "houston astros", "kansas city royals",
        "los angeles angels", "los angeles dodgers", "miami marlins", "milwaukee brewers",
        "minnesota twins", "new york mets", "new york yankees", "oakland athletics",
        "philadelphia phillies", "pittsburgh pirates", "san diego padres",
        "san francisco giants", "seattle mariners", "st. louis cardinals",
        "tampa bay rays", "texas rangers", "toronto blue jays", "washington nationals",
    }
    for c in canon:
        assert normalize_team(c) == c
    assert len(canon) == 30
    assert normalize_team("Not A Team") is None
    assert normalize_team("") is None


def _game(gid, y, m, d, home, away):
    return SimpleNamespace(id=gid, date=datetime(y, m, d, 19, 5),
                           home_team=home, away_team=away)


def _spaced_date(i: int):
    """A unique (month, day) for game ``i`` spaced >1 day apart so the +/-1 day match
    tolerance can't cross games (April/May/June, every 3rd day)."""
    n = i * 3
    return 4 + (n // 28), 1 + (n % 28)


# ─────────────────────────── odds_ingest: matching ───────────────────────────
def test_load_and_match_generic_csv(tmp_path):
    schedule = [
        _game(700001, 2023, 4, 2, "New York Yankees", "Boston Red Sox"),
        _game(700002, 2023, 4, 3, "Los Angeles Dodgers", "San Diego Padres"),
    ]
    csv_path = tmp_path / "odds.csv"
    csv_path.write_text(
        "date,home_team,away_team,home_ml,away_ml\n"
        "2023-04-02,Yankees,Red Sox,-150,+130\n"
        "2023-04-03,LA Dodgers,Padres,-120,+100\n",
        encoding="utf-8",
    )
    matched = load_and_match(str(csv_path), schedule, season=2023)
    assert set(matched) == {700001, 700002}
    assert matched[700001]["home_ml"] == -150.0
    assert matched[700001]["away_ml"] == 130.0
    # And the row feeds market_fair_home cleanly.
    assert 0.5 < market_fair_home(matched[700001]) < 0.65


def test_load_and_match_date_tolerance(tmp_path):
    # Odds file is one day off (common tz shift) — +/-1 day tolerance still matches.
    schedule = [_game(700003, 2023, 5, 10, "Chicago Cubs", "Milwaukee Brewers")]
    csv_path = tmp_path / "odds.csv"
    csv_path.write_text(
        "date,home_team,away_team,home_ml,away_ml\n"
        "2023-05-09,Cubs,Brewers,+105,-125\n",
        encoding="utf-8",
    )
    matched = load_and_match(str(csv_path), schedule, season=2023)
    assert set(matched) == {700003}


def test_load_and_match_sbr_format(tmp_path):
    # sportsbookreviewsonline: two rows per game (visitor then home), MMDD date.
    schedule = [_game(700004, 2023, 4, 2, "Houston Astros", "Seattle Mariners")]
    csv_path = tmp_path / "sbr.csv"
    csv_path.write_text(
        "Date,VH,Team,Close\n"
        "402,V,Mariners,+140\n"
        "402,H,Astros,-160\n",
        encoding="utf-8",
    )
    matched = load_and_match(str(csv_path), schedule, season=2023)
    assert set(matched) == {700004}
    assert matched[700004]["home_ml"] == -160.0
    assert matched[700004]["away_ml"] == 140.0


def test_load_and_match_missing_file_returns_empty():
    assert load_and_match("does-not-exist.csv", []) == {}


def test_load_and_match_skips_unparseable_rows(tmp_path):
    schedule = [_game(700005, 2023, 4, 2, "Atlanta Braves", "Miami Marlins")]
    csv_path = tmp_path / "odds.csv"
    csv_path.write_text(
        "date,home_team,away_team,home_ml,away_ml\n"
        "garbage,,,,,\n"
        "2023-04-02,Braves,Marlins,-140,+120\n"
        "2023-04-02,Nonexistent Team,Marlins,-140,+120\n",
        encoding="utf-8",
    )
    matched = load_and_match(str(csv_path), schedule, season=2023)
    assert set(matched) == {700005}


def test_load_and_match_doubleheader_two_gamepks_one_matchup(tmp_path):
    # A doubleheader: two scheduled games share the same (date, home, away) but have
    # distinct gamePks. The odds file supplies two rows for that matchup+date; BOTH
    # gamePks must receive odds (game 1 and game 2 of the DH), not collapse onto one.
    schedule = [
        _game(700010, 2023, 7, 4, "New York Mets", "Atlanta Braves"),
        _game(700011, 2023, 7, 4, "New York Mets", "Atlanta Braves"),
    ]
    csv_path = tmp_path / "dh.csv"
    csv_path.write_text(
        "date,home_team,away_team,home_ml,away_ml\n"
        "2023-07-04,New York Mets,Atlanta Braves,-115,+105\n"
        "2023-07-04,New York Mets,Atlanta Braves,-130,+110\n",
        encoding="utf-8",
    )
    matched = load_and_match(str(csv_path), schedule, season=2023)
    assert set(matched) == {700010, 700011}                 # both DH gamePks matched
    # The two rows carry different lines; each gamePk gets a distinct one.
    lines = {(m["home_ml"], m["away_ml"]) for m in matched.values()}
    assert lines == {(-115.0, 105.0), (-130.0, 110.0)}


def test_load_and_match_utc_local_date_boundary(tmp_path):
    # Schedule gameDate is UTC; a night game at a US venue rolls into the NEXT UTC
    # calendar day, so the schedule date is one day AHEAD of the game-local odds date.
    # The +/-1 tolerance must still match. Schedule 2023-08-16 (UTC), odds 2023-08-15.
    schedule = [_game(700020, 2023, 8, 16, "Los Angeles Dodgers", "San Diego Padres")]
    csv_path = tmp_path / "tz.csv"
    csv_path.write_text(
        "date,home_team,away_team,home_ml,away_ml\n"
        "2023-08-15,Los Angeles Dodgers,San Diego Padres,-150,+130\n",
        encoding="utf-8",
    )
    matched = load_and_match(str(csv_path), schedule, season=2023)
    assert set(matched) == {700020}


def test_load_and_match_series_across_date_boundary_no_collision(tmp_path):
    # A 3-game series where EVERY odds date is one day behind its UTC schedule date.
    # Sequence alignment must recover all three without a neighbour stealing a game.
    schedule = [
        _game(700030, 2023, 6, 11, "Seattle Mariners", "Texas Rangers"),  # UTC
        _game(700031, 2023, 6, 12, "Seattle Mariners", "Texas Rangers"),
        _game(700032, 2023, 6, 13, "Seattle Mariners", "Texas Rangers"),
    ]
    csv_path = tmp_path / "series.csv"
    csv_path.write_text(
        "date,home_team,away_team,home_ml,away_ml\n"
        "2023-06-10,Mariners,Rangers,-120,+100\n"
        "2023-06-11,Mariners,Rangers,-125,+105\n"
        "2023-06-12,Mariners,Rangers,-130,+110\n",
        encoding="utf-8",
    )
    matched = load_and_match(str(csv_path), schedule, season=2023)
    assert set(matched) == {700030, 700031, 700032}


def test_load_and_match_diagnostics_counts(tmp_path):
    schedule = [
        _game(700040, 2023, 4, 2, "Boston Red Sox", "New York Yankees"),
        _game(700041, 2023, 4, 5, "Boston Red Sox", "New York Yankees"),
    ]
    csv_path = tmp_path / "diag.csv"
    csv_path.write_text(
        "date,home_team,away_team,home_ml,away_ml\n"
        "2023-04-02,Red Sox,Yankees,-120,+100\n"     # matches 700040 (exact)
        "2023-04-20,Red Sox,Yankees,-120,+100\n"     # matchup exists, no game within +/-1 -> by_date
        "2023-04-02,Not A Team,Yankees,-120,+100\n"  # team fails -> by_team
        "2023-04-02,Miami Marlins,Chicago Cubs,-120,+100\n",  # matchup never scheduled -> no_matchup
        encoding="utf-8",
    )
    matched, diag = load_and_match(str(csv_path), schedule, season=2023, diagnostics=True)
    assert set(matched) == {700040}
    assert diag["matched"] == 1
    assert diag["unmatched_by_team"] == 1
    assert diag["unmatched_by_date"] == 1
    assert diag["unmatched_no_matchup"] == 1
    assert diag["total_rows"] == 4


# ─────────────────────────── end-to-end gate ───────────────────────────
def test_clv_end_to_end_good_model_matched_via_ingest(tmp_path):
    # Build a schedule + a generic odds file, match them, and run the gate. Market
    # prices 50/50; a sharp model (home 0.70, home wins 70%) should be tradeable.
    # Distinct dates per game so each (date, home, away) key is unique in the index.
    schedule = [_game(800000 + i, 2023, *_spaced_date(i), "Chicago Cubs", "Cincinnati Reds")
                for i in range(60)]
    rows = ["date,home_team,away_team,home_ml,away_ml"]
    records = []
    for i in range(60):
        m, d = _spaced_date(i)
        rows.append(f"2023-{m:02d}-{d:02d},Cubs,Reds,+100,+100")
        records.append(_rec(800000 + i, 0.70, 1.0 if i < 42 else 0.0))   # 70% home wins
    csv_path = tmp_path / "odds.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    odds_by_game = load_and_match(str(csv_path), schedule, season=2023)
    assert len(odds_by_game) == 60
    res = run_clv_backtest(records, odds_by_game, edge_threshold_bps=200)
    assert res["available"] is True
    assert set(res["gate"]) >= {"beats_market_brier", "positive_roi", "tradeable", "verdict"}
    assert res["model_beats_market"] is True
    assert res["gate"]["tradeable"] is True


def test_clv_end_to_end_bad_model_not_tradeable(tmp_path):
    # Model is confidently WRONG (home 0.70 but home wins only 30%) → market sharper,
    # negative ROI → not tradeable.
    schedule = [_game(810000 + i, 2023, *_spaced_date(i), "Texas Rangers", "Houston Astros")
                for i in range(60)]
    rows = ["date,home_team,away_team,home_ml,away_ml"]
    records = []
    for i in range(60):
        m, d = _spaced_date(i)
        rows.append(f"2023-{m:02d}-{d:02d},Rangers,Astros,+100,+100")
        records.append(_rec(810000 + i, 0.70, 1.0 if i < 18 else 0.0))   # only 30% home wins
    csv_path = tmp_path / "odds.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    odds_by_game = load_and_match(str(csv_path), schedule, season=2023)
    assert len(odds_by_game) == 60
    res = run_clv_backtest(records, odds_by_game, edge_threshold_bps=200)
    assert res["model_beats_market"] is False
    assert res["gate"]["tradeable"] is False
