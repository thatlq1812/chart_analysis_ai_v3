"""
Adapter Registry

Central registry for all pipeline stage adapters.
Supports the config-driven plugin architecture.

Every concrete adapter self-registers via the @register decorator:

    @register("s2_detection", "yolov8")
    class YOLOv8Adapter(BaseDetectionAdapter):
        ...

The PipelineBuilder then resolves adapters by (stage, name):

    cls = AdapterRegistry.resolve("s2_detection", "yolov8")
    instance = cls(config)

Adding a new adapter requires:
  1. Implement the stage's BaseAdapter subclass.
  2. Decorate with @register("stage_key", "adapter_name").
  3. Import the module anywhere before pipeline construction
     (or add to the stage package's __init__.py auto-imports).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)

# Internal registry: stage_key -> {adapter_name -> adapter_class}
_REGISTRY: Dict[str, Dict[str, Type]] = {}


def register(stage: str, name: str):
    """
    Class decorator to register an adapter implementation.

    Args:
        stage: Pipeline stage key, e.g. 's2_detection'.
        name:  Adapter identifier declared in pipeline.yaml, e.g. 'yolov8'.

    Example:
        @register("s2_detection", "yolov8")
        class YOLOv8Adapter(BaseDetectionAdapter):
            ...
    """

    def decorator(cls: Type) -> Type:
        _REGISTRY.setdefault(stage, {})[name] = cls
        logger.debug(f"Adapter registered | stage={stage} | name={name} | class={cls.__name__}")
        return cls

    return decorator


class AdapterRegistry:
    """
    Namespace for adapter resolution utilities.

    Usage:
        cls = AdapterRegistry.resolve("s2_detection", "yolov8")
        adapter = cls(config)
    """

    @staticmethod
    def resolve(stage: str, name: str) -> Type:
        """
        Look up an adapter class by stage and name.

        Args:
            stage: Stage key (e.g. 's2_detection').
            name:  Adapter name from config (e.g. 'yolov8').

        Returns:
            The registered adapter class.

        Raises:
            KeyError: If the (stage, name) pair is not registered.
        """
        stage_adapters = _REGISTRY.get(stage) or {}
        cls = stage_adapters.get(name)
        if cls is None:
            available = list(stage_adapters.keys())
            raise KeyError(
                f"No adapter '{name}' registered for stage '{stage}'. "
                f"Available: {available}"
            )
        return cls

    @staticmethod
    def list_adapters(stage: Optional[str] = None) -> Dict[str, Any]:
        """
        List all registered adapters.

        Args:
            stage: If provided, return only adapters for that stage.

        Returns:
            Dict mapping stage -> {name -> class_name}.
        """
        if stage:
            return {
                name: cls.__name__
                for name, cls in (_REGISTRY.get(stage) or {}).items()
            }
        return {
            s: {n: c.__name__ for n, c in adapters.items()}
            for s, adapters in _REGISTRY.items()
        }

    @staticmethod
    def is_registered(stage: str, name: str) -> bool:
        """Return True if (stage, name) is registered."""
        return name in (_REGISTRY.get(stage) or {})
