"""Model registry — single dispatch point for "give me the pace head" / "give me the DNF head".

Keeps the simulator decoupled from concrete model implementations. Add a new model
by registering it under a name; the simulator and backtest harness look it up here.
"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a model factory under the given name."""
    def _wrap(factory: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        _REGISTRY[name] = factory
        return factory
    return _wrap


def build(name: str, **kwargs: Any) -> Any:
    """Construct a registered model by name."""
    if name not in _REGISTRY:
        raise KeyError(f"no model registered as '{name}' — available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_registered() -> list[str]:
    return sorted(_REGISTRY)
