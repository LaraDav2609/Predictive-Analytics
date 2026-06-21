"""CS2 Game State Integration (GSI) → LiveMatchState.

CS2 can POST live game state to an HTTP endpoint (Valve's built-in Game State
Integration). This maps that payload to our LiveMatchState so the live updater can
turn it into a live win probability — a FREE real-time source (you run a CS2 client
spectating the match via GOTV and drop in a gamestate_integration_*.cfg).

The one subtlety: GSI reports the current CT and T sides, which swap at halftime, so
to produce a stable team1/team2 view the caller passes which team is team1 (by name);
otherwise team1 defaults to whoever is CT right now.
"""

from __future__ import annotations

from games.csgo.identity import normalize
from games.csgo.live import LiveMatchState


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sides_from_allplayers(allplayers) -> tuple:
    """(alive_ct, alive_t, equip_ct, equip_t) from the spectator-only `allplayers`
    block; all None when it isn't present (you only get it with GOTV/spec access)."""
    if not isinstance(allplayers, dict) or not allplayers:
        return None, None, None, None
    alive_ct = alive_t = equip_ct = equip_t = 0
    seen_ct = seen_t = 0
    for p in allplayers.values():
        if not isinstance(p, dict):
            continue
        team = (p.get("team") or "").upper()
        st = p.get("state") or {}
        hp = _int(st.get("health")) or 0
        equip = _int(st.get("equip_value")) or 0
        if team == "CT":
            seen_ct += 1
            equip_ct += equip
            alive_ct += 1 if hp > 0 else 0
        elif team == "T":
            seen_t += 1
            equip_t += equip
            alive_t += 1 if hp > 0 else 0
    if seen_ct == 0 and seen_t == 0:
        return None, None, None, None
    return alive_ct, alive_t, equip_ct, equip_t


def gsi_to_live_state(payload: dict, team1_name: str | None = None) -> tuple[LiveMatchState, str, str]:
    """Return (LiveMatchState, team1_name, team2_name) from a CS2 GSI payload."""
    mp = payload.get("map") or {}
    rnd = payload.get("round") or {}
    ct = mp.get("team_ct") or {}
    t = mp.get("team_t") or {}
    ct_name = ct.get("name") or "CT"
    t_name = t.get("name") or "T"

    # Which current side is team1? Default CT unless team1_name matches the T side.
    team1_is_ct = not (team1_name and normalize(team1_name) == normalize(t_name))

    if team1_is_ct:
        name1, name2 = ct_name, t_name
        rounds1, rounds2 = _int(ct.get("score")) or 0, _int(t.get("score")) or 0
        maps1, maps2 = _int(ct.get("matches_won_this_series")) or 0, _int(t.get("matches_won_this_series")) or 0
        team1_side = "CT"
    else:
        name1, name2 = t_name, ct_name
        rounds1, rounds2 = _int(t.get("score")) or 0, _int(ct.get("score")) or 0
        maps1, maps2 = _int(t.get("matches_won_this_series")) or 0, _int(ct.get("matches_won_this_series")) or 0
        team1_side = "T"

    n_series = _int(mp.get("num_matches_to_win_series"))
    best_of = (2 * n_series - 1) if n_series and n_series > 0 else 3

    bomb_planted = str(rnd.get("bomb")) == "planted"
    # The T side plants; that's whichever team currently plays T.
    bomb_planter_team = (1 if team1_side == "T" else 2) if bomb_planted else None

    alive_ct, alive_t, equip_ct, equip_t = _sides_from_allplayers(payload.get("allplayers"))
    if team1_is_ct:
        alive1, alive2, equip1, equip2 = alive_ct, alive_t, equip_ct, equip_t
    else:
        alive1, alive2, equip1, equip2 = alive_t, alive_ct, equip_t, equip_ct

    state = LiveMatchState(
        best_of=best_of,
        maps_won_team1=maps1, maps_won_team2=maps2,
        current_map=mp.get("name"),
        map_in_progress=str(mp.get("phase")) == "live",
        team1_rounds=rounds1, team2_rounds=rounds2,
        rounds_to_win=13,
        team1_side=team1_side,
        players_alive_team1=alive1, players_alive_team2=alive2,
        bomb_planted=bomb_planted, bomb_planter_team=bomb_planter_team,
        team1_equipment_value=equip1, team2_equipment_value=equip2,
        note=f"round={rnd.get('phase')}; bomb={rnd.get('bomb')}",
    )
    return state, name1, name2
