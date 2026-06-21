"""Deterministic F1 assistant service with tool-ready responses."""

from __future__ import annotations

import re
from typing import Any

from .context import build_context
from .safety import sanitize_action
from .schemas import AssistantActionRequest, AssistantChatRequest, AssistantResponse
from .tools import list_tools, run_diagnostics


class F1AssistantService:
    def __init__(self, *, client, predictor=None, openf1=None, live_engine=None, live_recorder=None, storage=None) -> None:
        self.client = client
        self.predictor = predictor
        self.openf1 = openf1
        self.live_engine = live_engine
        self.live_recorder = live_recorder
        self.storage = storage

    async def context(self, route: str):
        return await build_context(
            route=route,
            client=self.client,
            predictor=self.predictor,
            live_engine=self.live_engine,
            storage=self.storage,
        )

    def tools(self) -> dict[str, Any]:
        return {"ok": True, "tools": list_tools()}

    async def diagnostics(self, route: str) -> dict[str, Any]:
        context = await self.context(route)
        return await run_diagnostics(
            context=context,
            client=self.client,
            openf1=self.openf1,
            live_engine=self.live_engine,
            live_recorder=self.live_recorder,
            storage=self.storage,
        )

    async def chat(self, request: AssistantChatRequest) -> AssistantResponse:
        context = await self.context(request.route)
        message = (request.message or "").strip()
        normalized = message.lower()
        if not message:
            return self._welcome(context)

        if self._is_diagnostic_intent(normalized):
            diagnostics = await self.diagnostics(request.route)
            return AssistantResponse(
                mode="diagnostic",
                message=self._diagnostics_answer(diagnostics),
                context=context,
                diagnostics=diagnostics,
                citations=context.citations,
                suggestions=self._suggestions(context),
                missing_data_notice=self._missing_notice(context),
            )

        if self._is_car_performance_intent(normalized):
            return AssistantResponse(
                mode="answer",
                message=self._car_performance_answer(normalized, context),
                context=context,
                actions=self._car_performance_actions(normalized, context),
                citations=context.citations,
                suggestions=[
                    "Compare Mercedes and Ferrari car performance",
                    "Show tyre degradation risk",
                    "Why is confidence low?",
                ],
                missing_data_notice=self._missing_notice(context),
            )

        nav = self._navigation_intent(normalized, context)
        if nav:
            return AssistantResponse(
                mode="navigation",
                message=f"I can take you to {nav.get('label', 'that F1 page')}.",
                context=context,
                actions=[nav],
                citations=context.citations,
                suggestions=self._suggestions(context),
            )

        return AssistantResponse(
            mode="answer",
            message=self._answer_from_context(message, normalized, context),
            context=context,
            actions=self._context_actions(context),
            citations=context.citations,
            suggestions=self._suggestions(context),
            missing_data_notice=self._missing_notice(context),
        )

    async def action(self, request: AssistantActionRequest) -> AssistantResponse:
        action = sanitize_action(request.action or {})
        if not action.get("safe"):
            return AssistantResponse(
                ok=False,
                mode="error",
                message=f"I blocked that action because it is outside the F1 assistant scope: {action.get('blocked_reason') or 'unsafe_action'}.",
                actions=[action],
            )
        if action.get("requires_confirmation") and not request.confirmed:
            return AssistantResponse(
                mode="action_required",
                message="That action can change live state, so I need confirmation before running it.",
                actions=[action],
                requires_confirmation=True,
            )
        if action.get("type") == "navigate":
            return AssistantResponse(
                mode="navigation",
                message=f"Opening {action.get('label') or action.get('url')}.",
                actions=[action],
            )
        return AssistantResponse(
            mode="answer",
            message="The action is valid, but this v1 assistant only executes read-only diagnostics and navigation from the dashboard.",
            actions=[action],
        )

    def _welcome(self, context) -> AssistantResponse:
        return AssistantResponse(
            message=(
                f"I’m ready on {context.title}. Ask me to explain a probability, check live data, "
                "open a driver/team/race page, or run F1 diagnostics."
            ),
            context=context,
            actions=self._context_actions(context),
            suggestions=self._suggestions(context),
            citations=context.citations,
            missing_data_notice=self._missing_notice(context),
        )

    @staticmethod
    def _is_diagnostic_intent(text: str) -> bool:
        return any(token in text for token in [
            "diagnostic", "debug", "health", "status", "redis", "clickhouse", "openf1",
            "recorder", "live data", "source", "confidence low", "why is confidence",
            "smoke", "test",
        ])

    @staticmethod
    def _is_car_performance_intent(text: str) -> bool:
        return any(token in text for token in [
            "car performance", "car model", "low-speed", "low speed", "top speed",
            "straight line", "tyre degradation", "tire degradation", "sector strength",
            "sector strengths", "reliability", "teammate delta", "why is ferrari strong",
            "why is mercedes strong", "compare mercedes", "compare ferrari",
        ])

    def _navigation_intent(self, text: str, context) -> dict[str, Any] | None:
        routes = context.navigation
        if "model lab" in text or "global lab" in text:
            return self._nav("Global Model Lab", routes.get("model_lab", "/F1/ModelLab"))
        if "market" in text or "workspace" in text or "kalshi" in text or "polymarket" in text:
            return self._nav("F1 Workspace", routes.get("workspace", "/F1/Workspace"))
        if "next gp" in text or "next race" in text:
            return self._nav("next Grand Prix", routes.get("next_gp", "/F1"))
        if "live" in text and context.round:
            return self._nav("live race cockpit", routes.get("live", f"/F1/Race/{context.round}?tab=live"))
        if "qualifying" in text and context.round:
            return self._nav("qualifying tab", routes.get("qualifying", f"/F1/Race/{context.round}?tab=qualifying"))
        if "race tab" in text and context.round:
            return self._nav("race tab", routes.get("race", f"/F1/Race/{context.round}?tab=race"))

        race = self._find_race(text)
        if race:
            return self._nav(str(getattr(race, "name", "Grand Prix")), f"/F1/Race/{getattr(race, 'round', '')}")
        driver = self._find_driver(text)
        if driver:
            return self._nav(getattr(driver, "name", None) or getattr(driver, "last_name", "driver"), f"/F1/Driver/{getattr(driver, 'id', '')}")
        constructor = self._find_constructor(text)
        if constructor:
            return self._nav(getattr(constructor, "name", "constructor"), f"/F1/Constructor/{getattr(constructor, 'id', '')}")
        return None

    @staticmethod
    def _nav(label: str, url: str) -> dict[str, Any]:
        return sanitize_action({"type": "navigate", "label": label, "url": url, "requires_confirmation": False})

    def _find_race(self, text: str):
        for race in self.client.get_races() if self.client else []:
            haystack = " ".join(str(part or "").lower() for part in [
                getattr(race, "name", ""),
                getattr(race, "country", ""),
                getattr(race, "circuit", ""),
            ])
            if haystack and any(part and part in text for part in re.split(r"\s+", haystack)):
                name = str(getattr(race, "name", "") or "").lower()
                country = str(getattr(race, "country", "") or "").lower()
                if name in text or country in text or any(word in text for word in name.split()[:2]):
                    return race
        return None

    def _find_driver(self, text: str):
        for driver in self.client.get_drivers() if self.client else []:
            terms = [getattr(driver, "id", ""), getattr(driver, "code", ""), getattr(driver, "first_name", ""), getattr(driver, "last_name", "")]
            full = f"{getattr(driver, 'first_name', '')} {getattr(driver, 'last_name', '')}".strip()
            terms.append(full)
            if any(str(term).lower() and str(term).lower() in text for term in terms):
                return driver
        return None

    def _find_constructor(self, text: str):
        for constructor in self.client.get_constructors() if self.client else []:
            terms = [getattr(constructor, "id", ""), getattr(constructor, "name", "")]
            if any(str(term).lower() and str(term).lower() in text for term in terms):
                return constructor
        return None

    def _answer_from_context(self, original: str, text: str, context) -> str:
        current = context.current or {}
        if "probability" in text or "why" in text or "rank" in text or "prediction" in text:
            parts = [f"On {context.title}, the assistant is reading current F1 backend context first."]
            if context.source_mode:
                parts.append(f"The current source mode is `{context.source_mode}`.")
            if context.confidence is not None:
                parts.append(f"Evidence confidence is about {context.confidence:.0%}.")
            if context.missing_groups:
                parts.append("Missing or weak groups: " + ", ".join(context.missing_groups[:6]) + ".")
            profile = current.get("profile") or {}
            if profile.get("top_pick"):
                parts.append(f"Top pick currently resolves to `{profile.get('top_pick')}`.")
            parts.append("Use the diagnostics action if you want the exact live/source blockers.")
            return " ".join(parts)
        if "where am i" in text or "page" in text:
            return f"You are on the F1 `{context.page_type}` surface: {context.title}. I know route `{context.route}` and session `{context.session}`."
        return (
            f"I can help with `{original}` using the current F1 context for {context.title}. "
            "Ask for diagnostics, a probability explanation, live source status, or a route like `open Monaco race page`."
        )

    def _car_performance_answer(self, text: str, context) -> str:
        parts = [
            "I can run the F1 Car Performance Analyst for this scope. It reads the existing car-model pipeline and separates low-speed performance, tyre degradation, top speed, sector strengths, reliability, and teammate deltas."
        ]
        if context.round:
            parts.append(f"For this page I will use round {context.round} and session `{context.session or 'race'}`.")
        if context.source_mode:
            parts.append(f"Current source mode is `{context.source_mode}`.")
        if context.confidence is not None:
            parts.append(f"Current evidence confidence is about {context.confidence:.0%}.")
        if context.missing_groups:
            parts.append("Missing groups can cap the car analysis: " + ", ".join(context.missing_groups[:5]) + ".")
        else:
            parts.append("If OpenF1 car data is unavailable, the tool will clearly label track/historical fallback instead of pretending it is telemetry.")
        return " ".join(parts)

    def _car_performance_actions(self, text: str, context) -> list[dict[str, Any]]:
        constructor = self._find_constructor(text)
        constructor_id = getattr(constructor, "id", None) if constructor else context.constructor_id
        url = None
        if context.round:
            url = f"/F1/Race/{context.round}?handler=CarPerformanceAgent&session={context.session or 'race'}"
            if constructor_id:
                url += f"&constructorId={constructor_id}"
        elif constructor_id:
            url = f"/F1/Constructor/{constructor_id}?handler=CarPerformanceAgent"
        actions = [
            sanitize_action({
                "type": "open_panel",
                "label": "Run car performance analysis",
                "tool": "car_performance_analysis",
                "constructor_id": constructor_id,
                "round": context.round,
                "session": context.session or "race",
            })
        ]
        if url:
            actions.append(sanitize_action({"type": "navigate", "label": "Open car analysis data", "url": url}))
        return actions

    @staticmethod
    def _diagnostics_answer(diagnostics: dict[str, Any]) -> str:
        backend = diagnostics.get("backend") or {}
        storage = diagnostics.get("storage") or {}
        live = diagnostics.get("live") or {}
        redis = (storage.get("redis") or {}) if isinstance(storage, dict) else {}
        clickhouse = (storage.get("clickhouse") or {}) if isinstance(storage, dict) else {}
        lines = [
            f"Backend has {backend.get('drivers', 0)} drivers, {backend.get('constructors', 0)} constructors, and {backend.get('races', 0)} races loaded.",
            f"OpenF1 client available: {'yes' if backend.get('openf1_available') else 'no'}.",
            f"Live source mode: `{live.get('source_mode', 'unavailable')}` with confidence `{live.get('confidence', 'n/a')}`.",
            f"Redis: `{redis.get('status') or redis.get('reason') or redis.get('available')}`. ClickHouse: `{clickhouse.get('last_error') or clickhouse.get('available')}`.",
        ]
        if live.get("fallback_reason"):
            lines.append(f"Fallback reason: {live['fallback_reason']}.")
        if live.get("missing_groups"):
            lines.append("Missing live groups: " + ", ".join(live["missing_groups"][:6]) + ".")
        return " ".join(lines)

    @staticmethod
    def _context_actions(context) -> list[dict[str, Any]]:
        actions = [
            sanitize_action({"type": "diagnose", "label": "Run diagnostics", "tool": "live_diagnostics"}),
            sanitize_action({"type": "open_panel", "label": "Car Performance Analyst", "tool": "car_performance_analysis"}),
            sanitize_action({"type": "navigate", "label": "Global Lab", "url": context.navigation.get("model_lab", "/F1/ModelLab")}),
        ]
        if context.navigation.get("next_gp"):
            actions.append(sanitize_action({"type": "navigate", "label": "Open next GP", "url": context.navigation["next_gp"]}))
        if context.round:
            actions.append(sanitize_action({"type": "navigate", "label": "Live / Replay", "url": context.navigation.get("live", f"/F1/Race/{context.round}?tab=live")}))
        return actions

    @staticmethod
    def _suggestions(context) -> list[str]:
        suggestions = ["Check backend health", "Why is confidence low?", "Explain this probability", "Analyze car performance"]
        if context.round:
            suggestions.insert(0, "Show live data status")
        else:
            suggestions.insert(0, "Open next GP")
        return suggestions

    @staticmethod
    def _missing_notice(context) -> str | None:
        if context.source_mode in {"estimated", "unavailable"}:
            return f"Current F1 data is `{context.source_mode}`; exact live telemetry may be missing or unavailable."
        if context.missing_groups:
            return "Some evidence groups are missing or weak: " + ", ".join(context.missing_groups[:5])
        return None
