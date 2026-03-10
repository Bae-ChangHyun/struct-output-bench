from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.frameworks.base import BaseFrameworkAdapter

_registry: dict[str, type[BaseFrameworkAdapter]] = {}


class FrameworkRegistry:
    @staticmethod
    def register(name: str):
        def decorator(cls: type[BaseFrameworkAdapter]) -> type[BaseFrameworkAdapter]:
            _registry[name] = cls
            return cls

        return decorator

    @staticmethod
    def get(name: str) -> type[BaseFrameworkAdapter]:
        if name not in _registry:
            available = ", ".join(sorted(_registry.keys()))
            raise KeyError(
                f"Framework '{name}' not found. Available: {available}"
            )
        return _registry[name]

    @staticmethod
    def list_names() -> list[str]:
        return sorted(_registry.keys())

    @staticmethod
    def list_modes(name: str) -> list[str]:
        cls = FrameworkRegistry.get(name)
        return list(cls.supported_modes)
