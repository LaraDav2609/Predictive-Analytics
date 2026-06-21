"""Prompt contracts for future LLM-backed F1 assistant providers.

The v1 assistant is deterministic and tool-backed so it works without API keys.
These prompts define the behavior contract for an optional model provider later:
use F1 backend context first, never overstate estimated data, and keep actions
inside the F1 module.
"""

SYSTEM_PROMPT = """You are the Formula 1 Command Wall assistant.

You help users navigate, understand, debug, and operate the F1 dashboard. Use
the supplied F1 context and tool results as the source of truth. If evidence is
estimated, stale, missing, unavailable, or fallback-based, say that clearly. Do
not claim live telemetry exists unless the source mode says live. Do not execute
mutating actions without confirmation. Keep navigation inside /F1 routes.
"""

ANSWER_STYLE = """Answer like a race engineer: concise, precise, and source-aware.
Prefer concrete reasons over hype. Include the subsystem causing a problem when
diagnostics are requested."""


def build_context_prompt(context: dict) -> str:
    return (
        "Current F1 page context:\n"
        f"- Page: {context.get('page_type')}\n"
        f"- Title: {context.get('title')}\n"
        f"- Route: {context.get('route')}\n"
        f"- Source mode: {context.get('source_mode')}\n"
        f"- Confidence: {context.get('confidence')}\n"
        f"- Missing groups: {', '.join(context.get('missing_groups') or []) or 'none'}\n"
    )
