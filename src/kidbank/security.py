"""Security helpers used by the expanded KidBank feature set."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from secrets import token_urlsafe
from typing import Deque, Dict, Iterable, Optional
from uuid import uuid4


@dataclass(slots=True)
class Session:
    """Represents an authenticated session with CSRF metadata."""

    id: str
    user_id: str
    expires_at: datetime
    remember_me: bool
    csrf_token: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    def is_expired(self, *, at: Optional[datetime] = None) -> bool:
        moment = at or datetime.utcnow()
        return moment >= self.expires_at

    def refresh_csrf(self) -> str:
        self.csrf_token = token_urlsafe(16)
        return self.csrf_token


class AuthManager:
    """Provide coarse security controls for the in-memory application."""

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        lockout_minutes: int = 15,
        session_minutes: int = 60,
        remember_days: int = 30,
    ) -> None:
        self._max_attempts = max_attempts
        self._lockout_window = timedelta(minutes=lockout_minutes)
        self._session_duration = timedelta(minutes=session_minutes)
        self._remember_duration = timedelta(days=remember_days)
        self._login_attempts: Dict[str, Deque[datetime]] = {}
        self._pin_attempts: Dict[str, Deque[datetime]] = {}
        self._sessions: Dict[str, Session] = {}

    # ------------------------------------------------------------------
    # Rate limiting helpers
    # ------------------------------------------------------------------
    def record_login_attempt(self, user_id: str, *, success: bool, at: Optional[datetime] = None) -> bool:
        """Record a login attempt and return whether authentication should proceed."""

        return self._record_attempt(self._login_attempts, user_id, success, at)

    def record_pin_attempt(self, user_id: str, *, success: bool, at: Optional[datetime] = None) -> bool:
        """Record a PIN verification attempt and return whether verification is allowed."""

        return self._record_attempt(self._pin_attempts, user_id, success, at)

    def is_locked(self, user_id: str, *, at: Optional[datetime] = None) -> bool:
        """Return ``True`` when ``user_id`` is currently locked out."""

        return self._has_exceeded(self._login_attempts, user_id, at) or self._has_exceeded(
            self._pin_attempts, user_id, at
        )

    def _record_attempt(
        self,
        store: Dict[str, Deque[datetime]],
        user_id: str,
        success: bool,
        at: Optional[datetime],
    ) -> bool:
        now = at or datetime.utcnow()
        bucket = store.setdefault(user_id, deque())
        self._prune(bucket, now)
        if success:
            bucket.clear()
            return True
        bucket.append(now)
        return len(bucket) < self._max_attempts

    def _has_exceeded(self, store: Dict[str, Deque[datetime]], user_id: str, at: Optional[datetime]) -> bool:
        now = at or datetime.utcnow()
        bucket = store.get(user_id)
        if not bucket:
            return False
        self._prune(bucket, now)
        return len(bucket) >= self._max_attempts

    def _prune(self, bucket: Deque[datetime], now: datetime) -> None:
        while bucket and now - bucket[0] > self._lockout_window:
            bucket.popleft()

    # ------------------------------------------------------------------
    # Session handling helpers
    # ------------------------------------------------------------------
    def create_session(self, user_id: str, *, remember_me: bool = False, at: Optional[datetime] = None) -> Session:
        """Create a new session while honouring lockout constraints."""

        if self.is_locked(user_id, at=at):
            raise PermissionError("User is locked out due to repeated failed attempts.")
        now = at or datetime.utcnow()
        expiry = now + (self._remember_duration if remember_me else self._session_duration)
        session = Session(id=str(uuid4()), user_id=user_id, expires_at=expiry, remember_me=remember_me, csrf_token="")
        session.refresh_csrf()
        self._sessions[session.id] = session
        return session

    def validate_session(self, session_id: str, *, at: Optional[datetime] = None) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        if session.is_expired(at=at):
            self._sessions.pop(session_id, None)
            return False
        return True

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def expire_sessions(self, *, at: Optional[datetime] = None) -> None:
        now = at or datetime.utcnow()
        expired = [session_id for session_id, session in self._sessions.items() if session.is_expired(at=now)]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def logout(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def rotate_csrf(self, session_id: str) -> str:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Unknown session id '{session_id}'.")
        return session.refresh_csrf()

    def validate_csrf(self, session_id: str, token: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        if session.is_expired():
            self._sessions.pop(session_id, None)
            return False
        return session.csrf_token == token

    def active_sessions(self, user_id: str) -> Iterable[Session]:
        return tuple(session for session in self._sessions.values() if session.user_id == user_id)


__all__ = ["AuthManager", "Session"]
