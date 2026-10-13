"""Resolve a backend name from configuration to a :class:`Detector` instance."""

from __future__ import annotations

import importlib

from app.detectors.base import Detector, DetectorConfig

# Built-in backends are imported lazily so that only the ML framework that is
# actually used needs to be installed.
BUILTIN_BACKENDS: dict[str, str] = {
    "ultralytics": "app.detectors.ultralytics_backend:UltralyticsDetector",
    "torchvision": "app.detectors.torchvision_backend:TorchvisionDetector",
}


class BackendError(RuntimeError):
    pass


def resolve_backend(name: str) -> type[Detector]:
    """Return the detector class for a built-in name or a "module:Class" path."""
    target = BUILTIN_BACKENDS.get(name.lower(), name)
    if ":" not in target:
        known = ", ".join(sorted(BUILTIN_BACKENDS))
        raise BackendError(
            f"Unknown backend {name!r}. Use one of: {known}, or 'package.module:ClassName'."
        )
    module_name, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendError(
            f"Cannot import backend {name!r} ({exc}). Is its extra installed, "
            f"e.g. pip install '.[{name}]'?"
        ) from exc
    cls = getattr(module, class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, Detector)):
        raise BackendError(f"{target!r} is not a Detector subclass")
    return cls


def create_detector(backend: str, config: DetectorConfig) -> Detector:
    return resolve_backend(backend)(config)
