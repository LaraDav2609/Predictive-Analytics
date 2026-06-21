from __future__ import annotations

import pytest

from sports.f1.assistant.context import parse_f1_route
from sports.f1.assistant.safety import sanitize_action
from sports.f1.assistant.schemas import AssistantChatRequest
from sports.f1.assistant.service import F1AssistantService
from sports.f1.assistant.tools import list_tools


class _Race:
    round = 6
    name = "Monaco GP"
    country = "Monaco"
    circuit = "Circuit de Monaco"
    status = "upcoming"
    date = "2026-06-07"


class _Driver:
    id = "hamilton"
    code = "HAM"
    first_name = "Lewis"
    last_name = "Hamilton"
    team = "Ferrari"
    points = 72
    position = 4


class _Constructor:
    id = "mercedes"
    name = "Mercedes"
    points = 219
    position = 1


class _Client:
    season = 2026

    def get_races(self):
        return [_Race()]

    def get_drivers(self):
        return [_Driver()]

    def get_constructors(self):
        return [_Constructor()]

    def get_race_by_round(self, round_num):
        return _Race() if int(round_num) == 6 else None

    async def get_race_profile(self, round_num, predictor=None):
        return {
            "race": {
                "round": round_num,
                "name": "Monaco GP",
                "prediction": {"top_pick": "HAM", "confidence": 0.42},
            }
        }

    async def get_driver_profile(self, driver_id, season=None):
        return {"driver": {"id": driver_id, "code": "HAM", "name": "Lewis Hamilton", "team": "Ferrari"}}

    async def get_constructor_profile(self, constructor_id, season=None):
        return {"constructor": {"id": constructor_id, "name": "Mercedes", "points": 219}}


def test_parse_f1_route_race_tab():
    parsed = parse_f1_route("/F1/Race/6?tab=live&session=race")
    assert parsed["page_type"] == "race"
    assert parsed["round"] == 6
    assert parsed["tab"] == "live"
    assert parsed["session"] == "race"


def test_sanitize_navigation_blocks_non_f1_url():
    action = sanitize_action({"type": "navigate", "url": "https://example.com"})
    assert action["safe"] is False
    assert action["blocked_reason"] == "external_url_not_allowed"


@pytest.mark.asyncio
async def test_assistant_context_and_navigation_response():
    service = F1AssistantService(client=_Client())
    response = await service.chat(AssistantChatRequest(message="open monaco race page", route="/F1"))
    assert response.mode == "navigation"
    assert response.actions
    assert response.actions[0]["url"] == "/F1/Race/6"


@pytest.mark.asyncio
async def test_assistant_diagnostics_response_is_scoped_to_f1():
    service = F1AssistantService(client=_Client())
    response = await service.chat(AssistantChatRequest(message="check backend health", route="/F1/Race/6?tab=live"))
    assert response.mode == "diagnostic"
    assert "Backend has" in response.message
    assert response.diagnostics["backend"]["drivers"] == 1


def test_assistant_lists_car_performance_tool():
    tools = list_tools()
    assert any(tool["id"] == "car_performance_analysis" for tool in tools)


@pytest.mark.asyncio
async def test_assistant_car_performance_prompt_returns_read_only_action():
    service = F1AssistantService(client=_Client())
    response = await service.chat(AssistantChatRequest(message="compare Mercedes and Ferrari car performance", route="/F1/Race/6?tab=race"))

    assert response.mode == "answer"
    assert "Car Performance Analyst" in response.message
    assert any(action.get("tool") == "car_performance_analysis" for action in response.actions)
    assert all(action.get("safe") for action in response.actions)
