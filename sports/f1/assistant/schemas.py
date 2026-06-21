"""Schemas for the F1 assistant API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AssistantChatRequest(BaseModel):
    message: str = Field(default="", max_length=4000)
    route: str = "/F1"
    page_context: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None


class AssistantActionRequest(BaseModel):
    action: dict[str, Any] = Field(default_factory=dict)
    route: str = "/F1"
    confirmed: bool = False


class AssistantToolSpec(BaseModel):
    id: str
    label: str
    description: str
    safe: bool = True
    requires_confirmation: bool = False


class F1AssistantContext(BaseModel):
    ok: bool = True
    route: str = "/F1"
    page_type: str = "home"
    round: int | None = None
    driver_id: str | None = None
    constructor_id: str | None = None
    tab: str | None = None
    session: str = "race"
    title: str = "Formula 1 Command Wall"
    source_mode: str | None = None
    confidence: float | None = None
    missing_groups: list[str] = Field(default_factory=list)
    current: dict[str, Any] = Field(default_factory=dict)
    navigation: dict[str, str] = Field(default_factory=dict)
    citations: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    ok: bool = True
    mode: Literal["answer", "diagnostic", "navigation", "action_required", "error"] = "answer"
    message: str
    context: F1AssistantContext | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, str]] = Field(default_factory=list)
    requires_confirmation: bool = False
    missing_data_notice: str | None = None
