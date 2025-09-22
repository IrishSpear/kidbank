"""Operational utilities for KidBank."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from .models import BackupMetadata


class BackupManager:
    """Manage rolling backups of the in-memory SQLite database."""

    def __init__(self) -> None:
        self._backups: Dict[str, tuple[BackupMetadata, bytes]] = {}

    def create_backup(self, payload: bytes | str | dict, *, label: str = "") -> BackupMetadata:
        data: bytes
        if isinstance(payload, bytes):
            data = payload
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
        backup_id = str(uuid4())
        metadata = BackupMetadata(
            backup_id=backup_id,
            created_at=datetime.utcnow(),
            label=label or backup_id,
            size_bytes=len(data),
        )
        self._backups[backup_id] = (metadata, data)
        return metadata

    def get_backup(self, backup_id: str) -> tuple[BackupMetadata, bytes]:
        return self._backups[backup_id]

    def list_backups(self) -> tuple[BackupMetadata, ...]:
        return tuple(sorted((meta for meta, _ in self._backups.values()), key=lambda meta: meta.created_at))

    def purge_old(self, retain: int = 7) -> None:
        if retain <= 0:
            self._backups.clear()
            return
        backups = sorted(self._backups.values(), key=lambda item: item[0].created_at, reverse=True)
        for metadata, _ in backups[retain:]:
            self._backups.pop(metadata.backup_id, None)


class HealthMonitor:
    """Aggregate runtime health information for status pages."""

    def __init__(self) -> None:
        self.database_online = True
        self.migrations: list[str] = []
        self.cache_timestamp: Optional[datetime] = None

    def set_cache_timestamp(self, timestamp: datetime) -> None:
        self.cache_timestamp = timestamp

    def add_migration(self, name: str) -> None:
        self.migrations.append(name)

    def status(self) -> dict:
        return {
            "database": "ok" if self.database_online else "down",
            "migrations": list(self.migrations),
            "sp500_cache_age_seconds": self.cache_age_seconds(),
        }

    def cache_age_seconds(self) -> Optional[int]:
        if not self.cache_timestamp:
            return None
        return int((datetime.utcnow() - self.cache_timestamp).total_seconds())


class StructuredLogger:
    """Write JSON lines log entries for admin inspection."""

    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path
        self._entries: list[dict] = []

    def log(self, event_type: str, **fields: object) -> dict:
        entry = {"timestamp": datetime.utcnow().isoformat(), "event": event_type, **fields}
        self._entries.append(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        return entry

    def tail(self, limit: int = 50) -> tuple[dict, ...]:
        return tuple(self._entries[-limit:])


__all__ = ["BackupManager", "HealthMonitor", "StructuredLogger"]
