"""Administrative helpers for KidBank."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Dict, Optional
from uuid import uuid4

from .models import AuditEvent, FeatureFlag


class AuditLog:
    """Collect audit events for admin actions."""

    def __init__(self) -> None:
        self._entries: list[AuditEvent] = []

    def record(
        self,
        actor: str,
        action: str,
        target: str,
        *,
        details: Optional[dict] = None,
        timestamp: Optional[datetime] = None,
    ) -> AuditEvent:
        event = AuditEvent(
            actor=actor,
            action=action,
            target=target,
            timestamp=timestamp or datetime.utcnow(),
            details=dict(details or {}),
        )
        self._entries.append(event)
        return event

    def entries(self, *, action: str | None = None, target: str | None = None) -> tuple[AuditEvent, ...]:
        records = self._entries
        if action is not None:
            records = [entry for entry in records if entry.action == action]
        if target is not None:
            records = [entry for entry in records if entry.target == target]
        return tuple(records)

    def latest(self) -> AuditEvent | None:
        return self._entries[-1] if self._entries else None


class UndoManager:
    """Utility that supports undoing the most recent action within a window."""

    def __init__(self, *, window_seconds: int = 300) -> None:
        self._window = timedelta(seconds=window_seconds)
        self._last_event: tuple[str, datetime, Callable[[], None]] | None = None

    def register(self, undo_callable: Callable[[], None], *, timestamp: Optional[datetime] = None) -> str:
        event_id = str(uuid4())
        self._last_event = (event_id, timestamp or datetime.utcnow(), undo_callable)
        return event_id

    def undo(self, *, at: Optional[datetime] = None) -> str:
        if not self._last_event:
            raise LookupError("No undoable action is available.")
        event_id, event_time, undo_callable = self._last_event
        moment = at or datetime.utcnow()
        if moment - event_time > self._window:
            raise TimeoutError("Undo window has expired.")
        undo_callable()
        self._last_event = None
        return event_id


class FeatureFlagRegistry:
    """Minimal in-memory feature flag store backed by :class:`FeatureFlag`."""

    def __init__(self, initial: Optional[Dict[str, bool]] = None) -> None:
        self._flags: Dict[str, FeatureFlag] = {}
        for key, enabled in (initial or {}).items():
            self._flags[key] = FeatureFlag(key=key, enabled=bool(enabled))

    def set(self, key: str, enabled: bool, *, description: str = "") -> FeatureFlag:
        flag = FeatureFlag(key=key, enabled=enabled, description=description)
        self._flags[key] = flag
        return flag

    def enable(self, key: str) -> FeatureFlag:
        return self.set(key, True, description=self._flags.get(key, FeatureFlag(key, True)).description)

    def disable(self, key: str) -> FeatureFlag:
        return self.set(key, False, description=self._flags.get(key, FeatureFlag(key, False)).description)

    def is_enabled(self, key: str, *, default: bool = False) -> bool:
        flag = self._flags.get(key)
        if flag is None:
            return default
        return flag.enabled

    def describe(self, key: str, description: str) -> FeatureFlag:
        flag = self._flags.get(key)
        if not flag:
            flag = FeatureFlag(key=key, enabled=False, description=description)
        else:
            flag = FeatureFlag(key=flag.key, enabled=flag.enabled, description=description)
        self._flags[key] = flag
        return flag

    def list_flags(self) -> tuple[FeatureFlag, ...]:
        return tuple(sorted(self._flags.values(), key=lambda flag: flag.key))


__all__ = ["AuditLog", "FeatureFlagRegistry", "UndoManager"]
