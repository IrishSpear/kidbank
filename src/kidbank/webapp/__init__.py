"""KidBank web application package with optional dependencies."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, List

_OPTIONAL_IMPORT_ERROR: ModuleNotFoundError | None = None
_IMPL_MODULE: ModuleType | None = None

_OPTIONAL_MODULES = {"fastapi", "starlette", "sqlmodel", "sqlalchemy", "dotenv"}

try:
    from . import persistence as _persistence
except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
    if exc.name in _OPTIONAL_MODULES:
        _OPTIONAL_IMPORT_ERROR = exc
        raise RuntimeError(
            "kidbank.webapp requires optional FastAPI/SQLModel dependencies. "
            "Install them via `pip install -r requirements-web.txt` or the web extra."
        ) from exc
    raise

persistence = _persistence
__all__: List[str] = list(getattr(_persistence, "__all__", ()))


def _load_impl() -> ModuleType:
    global _IMPL_MODULE, _OPTIONAL_IMPORT_ERROR
    if _IMPL_MODULE is not None:
        return _IMPL_MODULE
    try:
        module = import_module(".application", __name__)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        if exc.name in _OPTIONAL_MODULES:
            _OPTIONAL_IMPORT_ERROR = exc
            raise RuntimeError(
                "kidbank.webapp requires optional FastAPI/SQLModel dependencies. "
                "Install them via `pip install -r requirements-web.txt` or the web extra."
            ) from exc
        raise
    _IMPL_MODULE = module
    module_all = getattr(module, "__all__", ())
    __all__.extend(name for name in module_all if name not in __all__)
    return module


def __getattr__(name: str) -> Any:
    if hasattr(_persistence, name):
        return getattr(_persistence, name)
    module = _load_impl()
    return getattr(module, name)


def __dir__() -> List[str]:
    names = set(globals()) | set(__all__)
    try:
        module = _load_impl()
    except RuntimeError:
        return sorted(names)
    return sorted(names | set(dir(module)))

