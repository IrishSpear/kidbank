"""KidBank web application package with optional dependencies."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, List

_OPTIONAL_IMPORT_ERROR: ModuleNotFoundError | None = None
_IMPL_MODULE: ModuleType | None = None

_OPTIONAL_MODULES = {"fastapi", "starlette", "sqlmodel", "sqlalchemy", "dotenv"}


def _load_impl() -> ModuleType:
    global _IMPL_MODULE, _OPTIONAL_IMPORT_ERROR
    if _IMPL_MODULE is not None:
        return _IMPL_MODULE
    try:
        _IMPL_MODULE = import_module(".application", __name__)
        return _IMPL_MODULE
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        if exc.name in _OPTIONAL_MODULES:
            _OPTIONAL_IMPORT_ERROR = exc
            raise RuntimeError(
                "kidbank.webapp requires optional FastAPI/SQLModel dependencies. "
                "Install them via `pip install -r requirements-web.txt` or the web extra."
            ) from exc
        raise


def __getattr__(name: str) -> Any:
    module = _load_impl()
    return getattr(module, name)


def __dir__() -> List[str]:
    try:
        module = _load_impl()
    except RuntimeError:
        return sorted(list(globals().keys()))
    return sorted(set(globals()) | set(dir(module)))

