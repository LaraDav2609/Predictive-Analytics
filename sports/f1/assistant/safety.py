"""Safety rules for F1 assistant actions."""

from __future__ import annotations

from urllib.parse import urlparse


READ_ONLY_ACTIONS = {"navigate", "explain", "diagnose", "smoke_check", "open_panel"}
MUTATING_ACTIONS = {"refresh", "start_recorder", "stop_recorder"}


def action_requires_confirmation(action: dict) -> bool:
    action_type = str(action.get("type") or "").strip().lower()
    return action_type in MUTATING_ACTIONS or bool(action.get("requires_confirmation"))


def validate_navigation_url(url: str) -> tuple[bool, str | None]:
    """Only allow dashboard-local F1 navigation targets."""
    if not url:
        return False, "missing_url"
    parsed = urlparse(url)
    if parsed.scheme or parsed.netloc:
        return False, "external_url_not_allowed"
    if not parsed.path.startswith("/F1"):
        return False, "non_f1_route_not_allowed"
    return True, None


def sanitize_action(action: dict) -> dict:
    action_type = str(action.get("type") or "").strip().lower()
    cleaned = {**action, "type": action_type}
    cleaned["requires_confirmation"] = action_requires_confirmation(cleaned)
    if action_type == "navigate":
        ok, reason = validate_navigation_url(str(cleaned.get("url") or ""))
        cleaned["safe"] = ok
        if reason:
            cleaned["blocked_reason"] = reason
    else:
        cleaned["safe"] = action_type in READ_ONLY_ACTIONS or action_type in MUTATING_ACTIONS
    return cleaned
