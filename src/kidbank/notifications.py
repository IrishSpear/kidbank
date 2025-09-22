"""Notification primitives for KidBank."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Sequence

from .models import ScheduledDigest


class NotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"


class NotificationType(str, Enum):
    CHORE_REMINDER = "chore_reminder"
    PAYOUT_POSTED = "payout_posted"
    WEEKLY_DIGEST = "weekly_digest"
    GOAL_MILESTONE = "goal_milestone"
    OTHER = "other"


@dataclass(slots=True)
class Notification:
    """Simple representation of a notification waiting to be delivered."""

    recipient: str
    channel: NotificationChannel
    type: NotificationType
    subject: str
    body: str
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> Dict[str, str]:
        payload = {
            "recipient": self.recipient,
            "channel": self.channel.value,
            "type": self.type.value,
            "subject": self.subject,
            "body": self.body,
            "created_at": self.created_at.isoformat(),
        }
        payload.update(self.metadata)
        return payload


class NotificationCenter:
    """In-memory notification inbox used for tests and integrations."""

    def __init__(self) -> None:
        self._queue: List[Notification] = []
        self._sent: List[Notification] = []
        self._scheduled: List[ScheduledDigest] = []

    def queue(self, notification: Notification) -> None:
        self._queue.append(notification)

    def pending(self, *, notification_type: NotificationType | None = None) -> Sequence[Notification]:
        if notification_type is None:
            return tuple(self._queue)
        return tuple(item for item in self._queue if item.type is notification_type)

    def pop_all(self) -> Sequence[Notification]:
        pending = tuple(self._queue)
        self._queue.clear()
        self._sent.extend(pending)
        return pending

    def mark_sent(self, notification: Notification) -> None:
        self._sent.append(notification)
        try:
            self._queue.remove(notification)
        except ValueError:
            pass

    def history(self) -> Sequence[Notification]:
        return tuple(self._sent)

    # Weekly digest helpers -------------------------------------------------
    def schedule_digest(self, digest: ScheduledDigest) -> None:
        self._scheduled.append(digest)

    def due_digests(self, *, at: datetime | None = None) -> Sequence[ScheduledDigest]:
        moment = at or datetime.utcnow()
        ready = [digest for digest in self._scheduled if digest.send_on <= moment]
        self._scheduled = [digest for digest in self._scheduled if digest.send_on > moment]
        return tuple(ready)

    def queue_digest_notifications(self, *, at: datetime | None = None) -> Sequence[Notification]:
        notifications: List[Notification] = []
        for digest in self.due_digests(at=at):
            body_lines = ["Weekly digest summary:"]
            for key, value in digest.summary.items():
                body_lines.append(f"- {key}: {value}")
            notification = Notification(
                recipient=digest.recipient,
                channel=NotificationChannel.EMAIL,
                type=NotificationType.WEEKLY_DIGEST,
                subject="KidBank weekly digest",
                body="\n".join(body_lines),
            )
            self.queue(notification)
            notifications.append(notification)
        return tuple(notifications)


__all__ = [
    "Notification",
    "NotificationCenter",
    "NotificationChannel",
    "NotificationType",
]
