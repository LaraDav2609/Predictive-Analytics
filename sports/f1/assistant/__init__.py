"""Formula 1 command assistant.

This package keeps assistant context, tool routing, and safety rules inside the
F1 vertical slice so the dashboard can expose one assistant across every F1 page
without coupling to other sports or games.
"""

from .schemas import (
    AssistantActionRequest,
    AssistantChatRequest,
    AssistantResponse,
    F1AssistantContext,
)
from .service import F1AssistantService

__all__ = [
    "AssistantActionRequest",
    "AssistantChatRequest",
    "AssistantResponse",
    "F1AssistantContext",
    "F1AssistantService",
]
