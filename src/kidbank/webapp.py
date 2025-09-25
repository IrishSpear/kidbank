"""FastAPI frontend for the KidBank playground project.

This module bundles a self-contained web UI that exposes the rich KidBank
feature set (chores, allowances, investing simulator, prizes, ledgers, etc.)
using SQLite for persistence.  It mirrors the functional requirements from the
reference snippet provided by the user while fitting into the existing Python
package so that it can be deployed in a Debian-based Proxmox LXC container with
minimal effort.

The implementation intentionally keeps the code in a single module so it can be
Easily served with ``uvicorn kidbank.webapp:app``.  For larger deployments the
components could be split into packages, but a single module keeps installation
straight-forward for the target environment.
"""

from __future__ import annotations

import base64
import calendar
import csv
import hashlib
import hmac
import io
import json
import math
import os
import re
import sqlite3
import statistics
import textwrap
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, timedelta
from html import escape as html_escape
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request as URLRequest, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse, Response
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlmodel import Field, Session, SQLModel, create_engine, desc, select
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOM_PIN = os.environ.get("MOM_PIN", "1022")
DAD_PIN = os.environ.get("DAD_PIN", "2097")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "change-this-session-secret")
SQLITE_FILE_NAME = os.environ.get("KIDBANK_SQLITE", "kidbank.db")
DEFAULT_PARENT_ROLES: Tuple[str, ...] = ("mom", "dad")
DEFAULT_PARENT_LABELS: Dict[str, str] = {"mom": "Mom", "dad": "Dad"}
EXTRA_PARENT_ADMINS_KEY = "parent_admins"
GLOBAL_CHORE_TYPES: Tuple[str, ...] = ("daily", "weekly", "monthly")
DEFAULT_GLOBAL_CHORE_TYPE = "daily"
PWA_CACHE_NAME = "kidbank-shell-v1"
PWA_SHELL_PATHS: Tuple[str, ...] = (
    "/",
    "/kid",
    "/admin",
    "/admin/login",
    "/manifest.webmanifest",
    "/pwa-icon.svg",
    "/offline",
)
PWA_ICON_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 128 128'>
  <defs>
    <linearGradient id='grad' x1='0' x2='1' y1='0' y2='1'>
      <stop offset='0%' stop-color='#2563eb'/>
      <stop offset='100%' stop-color='#1d4ed8'/>
    </linearGradient>
  </defs>
  <rect width='128' height='128' rx='28' ry='28' fill='url(#grad)'/>
  <text x='64' y='82' font-size='64' font-family='Roboto, Arial, sans-serif' font-weight='700' fill='#f8fafc' text-anchor='middle'>K</text>
</svg>
"""
SERVICE_WORKER_JS = (
    textwrap.dedent(
        """
        const CACHE_NAME = '__CACHE_NAME__';
        const OFFLINE_URLS = __OFFLINE_URLS__;

        self.addEventListener('install', event => {
          event.waitUntil(
            caches.open(CACHE_NAME).then(cache => cache.addAll(OFFLINE_URLS)).then(() => self.skipWaiting())
          );
        });

        self.addEventListener('activate', event => {
          event.waitUntil(
            caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))))
          );
          self.clients.claim();
        });

        async function fetchAndCache(request) {
          try {
            const response = await fetch(request);
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
            return response;
          } catch (error) {
            throw error;
          }
        }

        self.addEventListener('fetch', event => {
          if (event.request.method !== 'GET') {
            return;
          }
          const request = event.request;
          const url = new URL(request.url);
          if (request.mode === 'navigate') {
            event.respondWith(
              fetchAndCache(request).catch(() => caches.match(request).then(resp => resp || caches.match('/offline')))
            );
            return;
          }
          if (OFFLINE_URLS.includes(url.pathname)) {
            event.respondWith(
              caches.match(request).then(resp => resp || fetchAndCache(request))
            );
            return;
          }
          event.respondWith(
            fetchAndCache(request).catch(() => caches.match(request))
          );
        });
        """
    )
    .strip()
    .replace("__CACHE_NAME__", PWA_CACHE_NAME)
    .replace("__OFFLINE_URLS__", json.dumps(list(PWA_SHELL_PATHS)))
)


REMEMBER_COOKIE_NAME = "kid_remember"
REMEMBER_NAME_COOKIE = "kid_remember_name"
REMEMBER_COOKIE_LIFETIME = timedelta(days=30)
REMEMBER_COOKIE_MAX_AGE = int(REMEMBER_COOKIE_LIFETIME.total_seconds())


# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Kid Bank")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    max_age=None,
)


_time_provider: Callable[[], datetime] = datetime.now


def now_local() -> datetime:
    """Return naive local time using the configured provider."""

    return _time_provider()


# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------
engine = create_engine(
    f"sqlite:///{SQLITE_FILE_NAME}",
    echo=False,
    connect_args={"check_same_thread": False},
)

# Ensure fresh metadata when re-importing in test contexts.
SQLModel.metadata.clear()


if getattr(Session.__init__, "__name__", "") != "_session_init_no_expire":
    _SESSION_INIT = Session.__init__

    def _session_init_no_expire(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple wrapper
        if "expire_on_commit" not in kwargs:
            kwargs["expire_on_commit"] = False
        _SESSION_INIT(self, *args, **kwargs)

    Session.__init__ = _session_init_no_expire  # type: ignore[assignment]


class Child(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kid_id: str
    name: str
    balance_cents: int = 0
    kid_pin: Optional[str] = ""
    allowance_cents: int = 0
    notes: Optional[str] = None
    streak_days: int = 0
    badges: Optional[str] = ""
    level: int = 1
    total_points: int = 0
    last_chore_date: Optional[date] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    child_id: str
    change_cents: int
    reason: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MoneyRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    from_kid_id: str
    to_kid_id: str
    amount_cents: int
    reason: str = ""
    status: str = "pending"  # pending|accepted|declined
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


class Prize(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    cost_cents: int
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


ChoreType = Literal["daily", "weekly", "special", "global"]


class Chore(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kid_id: str
    name: str
    type: str  # daily|weekly|special|global
    award_cents: int
    notes: Optional[str] = None
    active: bool = True
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    max_claimants: int = 1
    weekdays: Optional[str] = None
    specific_dates: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChoreInstance(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chore_id: int
    period_key: str  # daily: YYYY-MM-DD; weekly: <SundayISO>-WEEK; special: SPECIAL
    status: str = "available"  # available|pending|pending_marketplace|paid
    completed_at: Optional[datetime] = None
    paid_event_id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


CHORE_STATUS_PENDING_MARKETPLACE = "pending_marketplace"


MARKETPLACE_STATUS_OPEN = "open"
MARKETPLACE_STATUS_CLAIMED = "claimed"
MARKETPLACE_STATUS_SUBMITTED = "submitted"
MARKETPLACE_STATUS_COMPLETED = "completed"
MARKETPLACE_STATUS_CANCELLED = "cancelled"
MARKETPLACE_STATUS_REJECTED = "rejected"



class MarketplaceListing(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_kid_id: str
    chore_id: int
    chore_name: str
    chore_award_cents: int
    offer_cents: int
    status: str = Field(default=MARKETPLACE_STATUS_OPEN)
    claimed_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    claimed_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    final_payout_cents: Optional[int] = None
    payout_note: Optional[str] = None
    resolved_by: Optional[str] = None
    payout_event_id: Optional[int] = None



class MetaKV(SQLModel, table=True):
    k: str = Field(primary_key=True)
    v: str


class Goal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kid_id: str
    name: str
    target_cents: int
    saved_cents: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    achieved_at: Optional[datetime] = None


class Certificate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kid_id: str
    principal_cents: int
    rate_bps: int
    term_months: int
    term_days: int = 0
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    matured_at: Optional[datetime] = None
    penalty_days: int = 0


class Investment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kid_id: str
    fund: str = "SP500"
    shares: float = 0.0


class InvestmentTx(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kid_id: str
    fund: str = "SP500"
    ts: datetime = Field(default_factory=datetime.utcnow)
    tx_type: str  # "buy" | "sell"
    shares: float
    price_cents: int
    amount_cents: int
    realized_pl_cents: int = 0


class MarketInstrument(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str
    name: str
    kind: str = "stock"  # stock|crypto
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KidMarketInstrument(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kid_id: str
    symbol: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GlobalChoreClaim(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chore_id: int
    kid_id: str
    period_key: str
    status: str = "pending"  # pending|approved|rejected
    award_cents: int = 0
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    notes: Optional[str] = None


class Lesson(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content_md: str
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Quiz(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    lesson_id: int
    payload: str
    reward_cents: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QuizAttempt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    child_id: str
    quiz_id: int
    score: int = 0
    max_score: int = 0
    responses: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


GLOBAL_CHORE_KID_ID = "__GLOBAL__"
GLOBAL_CHORE_STATUS_PENDING = "pending"
GLOBAL_CHORE_STATUS_APPROVED = "approved"
GLOBAL_CHORE_STATUS_REJECTED = "rejected"

INSTRUMENT_KIND_STOCK = "stock"
INSTRUMENT_KIND_CRYPTO = "crypto"
DEFAULT_MARKET_SYMBOL = "SP500"

TIME_MODE_AUTO = "auto"
TIME_MODE_MANUAL = "manual"
TIME_META_MODE_KEY = "time_mode"
TIME_META_MANUAL_KEY = "time_manual_iso"
TIME_META_OFFSET_KEY = "time_offset_minutes"
TIME_META_MANUAL_REF_KEY = "time_manual_reference"


# ---------------------------------------------------------------------------
# Database initialisation & migrations
# ---------------------------------------------------------------------------
def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return any(row[1] == column for row in cur.fetchall())


def _marketplace_error_needs_migration(exc: Exception) -> bool:
    message = str(exc).lower()
    return "marketplacelisting" in message and (
        "no such table" in message
        or "no such column" in message
        or "has no column" in message
    )


def _safe_marketplace_list(session: Session, query) -> List[MarketplaceListing]:
    try:
        return session.exec(query).all()
    except (sqlite3.OperationalError, OperationalError, ProgrammingError) as exc:
        if _marketplace_error_needs_migration(exc):
            run_migrations()
            try:
                return session.exec(query).all()
            except (sqlite3.OperationalError, OperationalError, ProgrammingError) as retry_exc:
                if _marketplace_error_needs_migration(retry_exc):
                    return []
                raise
        raise


def _safe_marketplace_first(session: Session, query) -> Optional[MarketplaceListing]:
    try:
        return session.exec(query).first()
    except (sqlite3.OperationalError, OperationalError, ProgrammingError) as exc:
        if _marketplace_error_needs_migration(exc):
            run_migrations()
            try:
                return session.exec(query).first()
            except (sqlite3.OperationalError, OperationalError, ProgrammingError) as retry_exc:
                if _marketplace_error_needs_migration(retry_exc):
                    return None
                raise
        raise


def run_migrations() -> None:
    raw = sqlite3.connect(SQLITE_FILE_NAME)
    try:
        if not _column_exists(raw, "child", "kid_pin"):
            raw.execute("ALTER TABLE child ADD COLUMN kid_pin TEXT DEFAULT '';")
        if not _column_exists(raw, "child", "allowance_cents"):
            raw.execute("ALTER TABLE child ADD COLUMN allowance_cents INTEGER DEFAULT 0;")
        if not _column_exists(raw, "child", "streak_days"):
            raw.execute("ALTER TABLE child ADD COLUMN streak_days INTEGER DEFAULT 0;")
        if not _column_exists(raw, "child", "badges"):
            raw.execute("ALTER TABLE child ADD COLUMN badges TEXT DEFAULT '';")
        if not _column_exists(raw, "child", "level"):
            raw.execute("ALTER TABLE child ADD COLUMN level INTEGER DEFAULT 1;")
        if not _column_exists(raw, "child", "total_points"):
            raw.execute("ALTER TABLE child ADD COLUMN total_points INTEGER DEFAULT 0;")
        if not _column_exists(raw, "child", "last_chore_date"):
            raw.execute("ALTER TABLE child ADD COLUMN last_chore_date TEXT;")
        if not _column_exists(raw, "chore", "start_date"):
            raw.execute("ALTER TABLE chore ADD COLUMN start_date TEXT;")
        if not _column_exists(raw, "chore", "end_date"):
            raw.execute("ALTER TABLE chore ADD COLUMN end_date TEXT;")
        if not _column_exists(raw, "choreinstance", "paid_event_id"):
            raw.execute("ALTER TABLE choreinstance ADD COLUMN paid_event_id INTEGER;")
        if not _column_exists(raw, "choreinstance", "created_at"):
            raw.execute("ALTER TABLE choreinstance ADD COLUMN created_at TEXT;")
        if not _column_exists(raw, "goal", "achieved_at"):
            raw.execute("ALTER TABLE goal ADD COLUMN achieved_at TEXT;")
        if not _column_exists(raw, "chore", "max_claimants"):
            raw.execute("ALTER TABLE chore ADD COLUMN max_claimants INTEGER DEFAULT 1;")
        if not _column_exists(raw, "chore", "weekdays"):
            raw.execute("ALTER TABLE chore ADD COLUMN weekdays TEXT;")
        if not _column_exists(raw, "chore", "specific_dates"):
            raw.execute("ALTER TABLE chore ADD COLUMN specific_dates TEXT;")
        if not _column_exists(raw, "certificate", "term_days"):
            raw.execute("ALTER TABLE certificate ADD COLUMN term_days INTEGER DEFAULT 0;")
            raw.execute(
                """
                UPDATE certificate
                SET term_days = (
                    CASE
                        WHEN term_months IS NULL OR term_months < 0 THEN 0
                        ELSE term_months * 30
                    END
                )
                WHERE IFNULL(term_days, 0) = 0;
                """
            )
        if not _column_exists(raw, "certificate", "penalty_days"):
            raw.execute("ALTER TABLE certificate ADD COLUMN penalty_days INTEGER DEFAULT 0;")
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS investment (
                id INTEGER PRIMARY KEY,
                kid_id TEXT,
                fund TEXT,
                shares REAL
            );
            """
        )
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS certificate (
                id INTEGER PRIMARY KEY,
                kid_id TEXT,
                principal_cents INTEGER,
                rate_bps INTEGER,
                term_months INTEGER,
                term_days INTEGER DEFAULT 0,
                opened_at TEXT,
                matured_at TEXT,
                penalty_days INTEGER DEFAULT 0
            );
            """
        )
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS investmenttx (
                id INTEGER PRIMARY KEY,
                kid_id TEXT,
                fund TEXT,
                ts TEXT,
                tx_type TEXT,
                shares REAL,
                price_cents INTEGER,
                amount_cents INTEGER,
                realized_pl_cents INTEGER
            );
            """
        )
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS marketinstrument (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                kind TEXT,
                created_at TEXT
            );
            """
        )
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS kidmarketinstrument (
                id INTEGER PRIMARY KEY,
                kid_id TEXT,
                symbol TEXT,
                created_at TEXT
            );
            """
        )
        raw.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kidmarketinstrument_kid_symbol
            ON kidmarketinstrument(kid_id, symbol);
            """
        )
        raw.execute(
            """
            INSERT OR IGNORE INTO kidmarketinstrument (kid_id, symbol, created_at)
            SELECT DISTINCT kid_id, UPPER(fund), CURRENT_TIMESTAMP
            FROM investment
            WHERE IFNULL(kid_id, '') != '' AND IFNULL(fund, '') != '';
            """
        )
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS globalchoreclaim (
                id INTEGER PRIMARY KEY,
                chore_id INTEGER,
                kid_id TEXT,
                period_key TEXT,
                status TEXT,
                award_cents INTEGER,
                submitted_at TEXT,
                approved_at TEXT,
                approved_by TEXT,
                notes TEXT
            );
            """
        )
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS lesson (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content_md TEXT,
                summary TEXT,
                created_at TEXT
            );
            """
        )
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz (
                id INTEGER PRIMARY KEY,
                lesson_id INTEGER,
                payload TEXT,
                reward_cents INTEGER DEFAULT 0,
                created_at TEXT
            );
            """
        )
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS quizattempt (
                id INTEGER PRIMARY KEY,
                child_id TEXT,
                quiz_id INTEGER,
                score INTEGER,
                max_score INTEGER,
                responses TEXT,
                created_at TEXT
            );
            """
        )
        raw.execute(
            """
            CREATE TABLE IF NOT EXISTS marketplacelisting (
                id INTEGER PRIMARY KEY,
                owner_kid_id TEXT,
                chore_id INTEGER,
                chore_name TEXT,
                chore_award_cents INTEGER,
                offer_cents INTEGER,
                status TEXT,
                claimed_by TEXT,
                created_at TEXT,
                claimed_at TEXT,
                submitted_at TEXT,
                completed_at TEXT,
                cancelled_at TEXT,
                final_payout_cents INTEGER,
                payout_note TEXT,
                resolved_by TEXT,
                payout_event_id INTEGER

            );
            """
        )
        raw.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_marketplace_owner_status
            ON marketplacelisting(owner_kid_id, status);
            """
        )
        raw.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_marketplace_status
            ON marketplacelisting(status);
            """
        )

        if not _column_exists(raw, "marketplacelisting", "submitted_at"):
            raw.execute(
                "ALTER TABLE marketplacelisting ADD COLUMN submitted_at TEXT;"
            )
        if not _column_exists(raw, "marketplacelisting", "final_payout_cents"):
            raw.execute(
                "ALTER TABLE marketplacelisting ADD COLUMN final_payout_cents INTEGER;"
            )
        if not _column_exists(raw, "marketplacelisting", "payout_note"):
            raw.execute(
                "ALTER TABLE marketplacelisting ADD COLUMN payout_note TEXT;"
            )
        if not _column_exists(raw, "marketplacelisting", "resolved_by"):
            raw.execute(
                "ALTER TABLE marketplacelisting ADD COLUMN resolved_by TEXT;"
            )
        if not _column_exists(raw, "marketplacelisting", "payout_event_id"):
            raw.execute(
                "ALTER TABLE marketplacelisting ADD COLUMN payout_event_id INTEGER;"
            )

        raw.commit()
    finally:
        raw.close()


create_db_and_tables()
run_migrations()


# ---------------------------------------------------------------------------
# Money helpers
# ---------------------------------------------------------------------------
CD_RATE_KEY = "cd_rate_bps"
CD_PENALTY_DAYS_KEY = "cd_penalty_days"
DEFAULT_CD_RATE_BPS = 250
DEFAULT_CD_PENALTY_DAYS = 0
DEFAULT_CD_TERM_CODE = "12m"
CD_TERM_OPTIONS: List[Tuple[str, str, int]] = [
    ("7d", "1 week", 7),
    ("3m", "3 months", 90),
    ("6m", "6 months", 180),
    ("12m", "12 months", 360),
]
CD_TERM_LOOKUP: Dict[str, Tuple[str, int]] = {
    code: (label, days) for code, label, days in CD_TERM_OPTIONS
}


def _cd_rate_meta_key(term_code: str) -> str:
    return f"{CD_RATE_KEY}_{term_code.strip().lower()}"


def _cd_penalty_meta_key(term_code: str) -> str:
    return f"{CD_PENALTY_DAYS_KEY}_{term_code.strip().lower()}"


def usd(cents: int) -> str:
    try:
        return f"${(cents or 0) / 100:.2f}"
    except Exception:  # pragma: no cover - defensive fallback
        return "$0.00"


def dollars_value(cents: int) -> str:
    try:
        return f"{(cents or 0) / 100:.2f}"
    except Exception:  # pragma: no cover
        return "0.00"


def to_cents_from_dollars_str(raw: str, default: int = 0) -> int:
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        return int(round(float(raw) * 100))
    except Exception:
        return default


def percent_complete(saved: int, target: int) -> Optional[float]:
    if target <= 0:
        return None
    pct = (saved / target) * 100 if target else 0
    if pct < 0:
        pct = 0
    if pct > 100:
        pct = 100
    return pct


def format_percent(pct: Optional[float]) -> str:
    if pct is None:
        return "—"
    if pct <= 0:
        return "0%"
    if pct < 1:
        return f"{pct:.1f}%"
    if pct < 100:
        return f"{pct:.0f}%"
    return "100%"


def filter_events(
    events: Sequence[Event],
    *,
    search: str = "",
    direction: str = "all",
    kid_lookup: Optional[Dict[str, Child]] = None,
    kid_filter: str = "",
) -> List[Event]:
    normalized_direction = (direction or "all").lower()
    if normalized_direction not in {"all", "credit", "debit", "zero"}:
        normalized_direction = "all"
    normalized_kid = (kid_filter or "").strip().lower()
    raw_search = (search or "").strip().lower()
    tokens = [token for token in re.split(r"\s+", raw_search) if token] if raw_search else []
    amount_filters: List[Tuple[str, int]] = []
    text_tokens: List[str] = []
    amount_pattern = re.compile(r"^(>=|<=|>|<|=)?\$?(-?\d+(?:\.\d{1,2})?)$")
    for token in tokens:
        match = amount_pattern.match(token)
        if match:
            op = match.group(1) or "="
            try:
                cents_value = int(round(float(match.group(2)) * 100))
            except (TypeError, ValueError):
                text_tokens.append(token)
                continue
            amount_filters.append((op, cents_value))
        else:
            text_tokens.append(token)
    filtered: List[Event] = []
    for event in events:
        if normalized_kid and (event.child_id or "").lower() != normalized_kid:
            continue
        if normalized_direction == "credit" and event.change_cents <= 0:
            continue
        if normalized_direction == "debit" and event.change_cents >= 0:
            continue
        if normalized_direction == "zero" and event.change_cents != 0:
            continue
        if amount_filters:
            amount_match = True
            for op, cents in amount_filters:
                value = event.change_cents
                if op == ">" and not value > cents:
                    amount_match = False
                    break
                if op == "<" and not value < cents:
                    amount_match = False
                    break
                if op == ">=" and not value >= cents:
                    amount_match = False
                    break
                if op == "<=" and not value <= cents:
                    amount_match = False
                    break
                if op == "=" and value != cents:
                    amount_match = False
                    break
            if not amount_match:
                continue
        if text_tokens:
            haystack = [
                (event.reason or "").lower(),
                (event.reason or "").replace("_", " ").lower(),
                usd(event.change_cents).lower(),
                f"{event.change_cents/100:.2f}",
                event.timestamp.strftime("%Y-%m-%d %H:%M").lower(),
                event.timestamp.strftime("%Y-%m-%d").lower(),
            ]
            if event.child_id:
                haystack.append(event.child_id.lower())
                if kid_lookup and event.child_id in kid_lookup:
                    haystack.append(kid_lookup[event.child_id].name.lower())
            if not all(any(token in text for text in haystack) for token in text_tokens):
                continue
        filtered.append(event)
    return filtered


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def admin_role(request: Request) -> Optional[str]:
    return request.session.get("admin_role")


def admin_authed(request: Request) -> bool:
    return bool(admin_role(request))


def require_admin(request: Request) -> Optional[RedirectResponse]:
    if not admin_authed(request):
        return RedirectResponse("/admin/login", status_code=302)
    return None


def _remember_signature(payload: str) -> str:
    secret = SESSION_SECRET.encode("utf-8")
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _encode_remember_token(kid_id: str, expires_at: datetime) -> str:
    payload = f"{kid_id}:{int(expires_at.timestamp())}"
    signature = _remember_signature(payload)
    token_raw = f"{payload}:{signature}".encode("utf-8")
    return base64.urlsafe_b64encode(token_raw).decode("utf-8")


def _decode_remember_token(token: str) -> Optional[Tuple[str, datetime]]:
    try:
        decoded = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        kid_id, expires_raw, signature = decoded.split(":")
        payload = f"{kid_id}:{expires_raw}"
        expected = _remember_signature(payload)
        if not hmac.compare_digest(signature, expected):
            return None
        expires_at = datetime.fromtimestamp(int(expires_raw))
        if expires_at < datetime.utcnow():
            return None
        return kid_id, expires_at
    except Exception:
        return None


def kid_authed(request: Request) -> Optional[str]:
    cached = getattr(request.state, "_kid_authed_cache", None)
    if cached:
        return cached
    kid_id = request.session.get("kid_authed")
    if kid_id:
        request.state._kid_authed_cache = kid_id  # type: ignore[attr-defined]
        return kid_id
    token = request.cookies.get(REMEMBER_COOKIE_NAME)
    if not token:
        return None
    decoded = _decode_remember_token(token)
    if not decoded:
        return None
    remember_kid_id, _ = decoded
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == remember_kid_id)).first()
        if not child:
            return None
    request.session["kid_authed"] = remember_kid_id
    request.state._kid_authed_cache = remember_kid_id  # type: ignore[attr-defined]
    return remember_kid_id


def require_kid(request: Request) -> Optional[RedirectResponse]:
    if not kid_authed(request):
        return RedirectResponse("/", status_code=302)
    return None


def set_kid_notice(request: Request, message: str, kind: str = "info") -> None:
    request.session["kid_notice"] = message
    request.session["kid_notice_kind"] = kind


def pop_kid_notice(request: Request) -> Tuple[Optional[str], str]:
    message = request.session.pop("kid_notice", None)
    kind = request.session.pop("kid_notice_kind", "info")
    return message, kind


def set_admin_notice(request: Request, message: str, kind: str = "info") -> None:
    request.session["admin_notice"] = message
    request.session["admin_notice_kind"] = kind


def pop_admin_notice(request: Request) -> Tuple[Optional[str], str]:
    message = request.session.pop("admin_notice", None)
    kind = request.session.pop("admin_notice_kind", "info")
    return message, kind


def current_route_with_query(request: Request) -> str:
    query = request.url.query
    return f"{request.url.path}?{query}" if query else request.url.path


@app.post("/ui/preferences")
def set_ui_preferences(
    request: Request,
    font: str = Form("default"),
    contrast: str = Form("standard"),
    redirect_to: str = Form("/"),
) -> RedirectResponse:
    font_pref = font if font in {"default", "dyslexic"} else "default"
    contrast_pref = contrast if contrast in {"standard", "high"} else "standard"
    request.session["ui_font"] = font_pref
    request.session["ui_contrast"] = contrast_pref
    target = redirect_to if redirect_to.startswith("/") else "/"
    return RedirectResponse(target, status_code=302)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------
def body_pref_attrs(request: Optional[Request] = None) -> str:
    classes = ["touch-friendly"]
    font_pref = "default"
    contrast_pref = "standard"
    if request is not None:
        font_pref = (request.session.get("ui_font") or "default").strip().lower()
        contrast_pref = (request.session.get("ui_contrast") or "standard").strip().lower()
    if font_pref == "dyslexic":
        classes.append("font-dyslexic")
    if contrast_pref == "high":
        classes.append("theme-high-contrast")
    attrs = [f"class='{' '.join(classes)}'", f"data-font='{font_pref}'", f"data-contrast='{contrast_pref}'", "data-shell='kidbank'"]
    return " ".join(attrs)


def preference_controls_html(request: Request) -> str:
    font_pref = (request.session.get("ui_font") or "default").strip().lower()
    contrast_pref = (request.session.get("ui_contrast") or "standard").strip().lower()
    redirect_target = html_escape(current_route_with_query(request))

    def pref_form(font_value: str, contrast_value: str, label: str, active: bool) -> str:
        return (
            "<form method='post' action='/ui/preferences' class='ui-pref-form'>"
            + f"<input type='hidden' name='redirect_to' value='{redirect_target}'>"
            + f"<input type='hidden' name='font' value='{font_value}'>"
            + f"<input type='hidden' name='contrast' value='{contrast_value}'>"
            + f"<button type='submit' aria-pressed='{str(active).lower()}'>{label}</button>"
            + "</form>"
        )

    controls = ["<div class='ui-pref-group' role='group' aria-label='Display preferences'>"]
    controls.append("<span class='muted' style='font-size:13px;'>Font:</span>")
    controls.append(
        pref_form("default", contrast_pref, "Standard", active=font_pref != "dyslexic")
    )
    controls.append(
        pref_form("dyslexic", contrast_pref, "Dyslexic", active=font_pref == "dyslexic")
    )
    controls.append("<span class='muted' style='font-size:13px;'>Contrast:</span>")
    controls.append(
        pref_form(font_pref, "standard", "Standard", active=contrast_pref != "high")
    )
    controls.append(
        pref_form(font_pref, "high", "High contrast", active=contrast_pref == "high")
    )
    controls.append("</div>")
    return "".join(controls)


def service_worker_registration() -> str:
    return """
    <script>
      if ('serviceWorker' in navigator) {
        window.addEventListener('load', function() {
          navigator.serviceWorker.register('/service-worker.js').catch(function() {
            console.warn('Service worker registration failed.');
          });
        });
      }
    </script>
    """


def base_styles() -> str:
    return """
    <style>
      :root{
        --bg:#0b1220; --card:#111827; --muted:#9aa4b2; --accent:#2563eb;
        --good:#16a34a; --bad:#dc2626; --text:#e5e7eb;
      }
      @media (prefers-color-scheme: light){
        :root{ --bg:#f7fafc; --card:#ffffff; --muted:#64748b; --accent:#2563eb; --text:#0f172a; }
      }
      html, body { overflow-x: hidden; }
      th, td, button, a, input { overflow-wrap:anywhere; word-break:break-word; }
      body{
        font-family: system-ui,-apple-system,Segoe UI,Roboto,Arial;
        background:var(--bg); color:var(--text);
        max-width:1320px;
        margin:24px auto;
        padding:0 16px;
      }
      .layout{display:grid; grid-template-columns:220px 1fr; gap:16px; align-items:flex-start;}
      .layout .content{min-width:0;}
      .sidebar{display:flex; flex-direction:column; gap:6px; position:sticky; top:24px;}
      .sidebar a{display:block; padding:10px 12px; border-radius:10px; text-decoration:none; color:var(--text); background:rgba(255,255,255,0.04); transition:filter .15s ease, transform .15s ease;}
      .sidebar a:hover{filter:brightness(1.05); transform:translateX(2px);}
      .sidebar a.active{background:var(--accent); color:#fff; box-shadow:0 10px 24px rgba(37,99,235,0.25);}
      .button-link{display:inline-flex; align-items:center; justify-content:center; padding:10px 14px; border-radius:10px; text-decoration:none; font-weight:600; background:rgba(255,255,255,0.06); color:var(--text);} 
      .button-link.secondary{background:rgba(148,163,184,0.16); color:var(--text);} 
      .button-link.danger{background:var(--bad); color:#fff;}
      .button-link:hover{filter:brightness(1.05);}
      .grid{
        display:grid;
        grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
        gap:16px;
      }
      .card{
        background:var(--card);
        border-radius:12px;
        padding:16px;
        box-shadow:0 8px 20px rgba(0,0,0,.08);
        margin:12px 0;
      }
      input,textarea,select{
        width:100%; padding:12px; border:1px solid #2b3545; border-radius:10px;
        background:transparent; color:var(--text); box-sizing:border-box; font-size:16px;
      }
      input[type=checkbox], input[type=radio]{
        width:auto; min-width:0; padding:0; margin:0 6px 0 0;
      }
      input::placeholder{color:var(--muted)}
      button{
        padding:12px 14px; border-radius:10px; border:0; background:var(--accent);
        color:#fff; cursor:pointer; min-height:44px;
      }
      button:hover{filter:brightness(1.05)}
      .danger{ background:var(--bad); }
      .stat-value{font-size:28px; font-weight:700;}
      form.inline{ display:grid; grid-template-columns: 1fr auto; gap:8px; align-items:end; }
      @media (min-width: 900px){ .card form.inline{ grid-template-columns: 1fr 1fr auto; } }
      .stacked-form{display:flex; flex-direction:column; gap:8px;}
      .stacked-form label{font-weight:600;}
      .stacked-form .actions{justify-content:flex-end;}
      .actions{display:flex; gap:8px; flex-wrap:wrap; align-items:center;}
      table{width:100%; border-collapse:collapse}
      th,td{padding:10px; border-bottom:1px solid #243041; text-align:left; vertical-align:top}
      .right{text-align:right}
      .muted{color:var(--muted)}
      .topbar{display:flex; justify-content:space-between; align-items:center; margin-bottom:8px}
      .pill{display:inline-block; padding:4px 8px; border-radius:999px; background:#1f2937; color:#cbd5e1; font-size:12px}
      .kiosk{display:flex; gap:16px; align-items:center; justify-content:space-between}
      .kiosk .balance{font-size:52px; font-weight:900}
      .hero-card{display:flex; gap:24px; align-items:flex-start; justify-content:space-between;}
      .hero-card__intro{display:flex; flex-direction:column; gap:8px; max-width:60%;}
      .hero-card__greeting{font-size:28px; font-weight:700;}
      .hero-card__meta{color:var(--muted); font-size:14px;}
      .hero-card__badges .pill{margin-right:6px; margin-top:4px;}
      .hero-card__progress{margin-top:4px;}
      .hero-card__progress-label{font-weight:600; font-size:14px;}
      .progress-bar{width:100%; height:8px; border-radius:999px; background:rgba(148,163,184,0.2); overflow:hidden; margin-top:6px;}
      .progress-bar__fill{height:100%; background:var(--accent);}
      .hero-card__balance{text-align:right; min-width:200px;}
      .hero-card__amount{font-size:42px; font-weight:800;}
      .hero-card__allowance{margin-top:6px; font-size:14px;}
      .hero-card__pill{background:rgba(37,99,235,0.18); color:var(--text);}
      .overview-stats-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:16px; margin:16px 0;}
      .stat-card{background:rgba(15,23,42,0.6); border-radius:12px; padding:16px; display:flex; flex-direction:column; gap:6px; box-shadow:inset 0 0 0 1px rgba(148,163,184,0.08);}
      .stat-card__label{font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em;}
      .stat-card__value{font-size:26px; font-weight:700;}
      .stat-card__meta{font-size:13px; color:var(--muted);}
      .stat-card__action{align-self:flex-start; margin-top:8px; padding:8px 14px; border-radius:10px; background:rgba(148,163,184,0.16); color:var(--text); text-decoration:none; font-weight:600;}
      .stat-card__action:hover{filter:brightness(1.08);}
      .analytics-card{margin-top:18px;}
      .insight-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin-top:12px;}
      .insight-label{font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em;}
      .insight-value{font-size:24px; font-weight:700;}
      .insight-meta{font-size:13px; color:var(--muted); margin-top:4px;}
      .insight-note{margin-top:12px; font-size:13px;}
      .insight-grid--tight .insight-value{font-size:20px;}
      .insight-link{display:inline-block; margin-top:12px;}
      .text-good{color:#22c55e;}
      .text-bad{color:#f97316;}
      .distribution-bar{display:flex; height:12px; border-radius:999px; overflow:hidden; background:rgba(148,163,184,0.18); margin-top:12px;}
      .distribution-bar__segment{height:100%;}
      .distribution-bar__segment--market{background:#38bdf8;}
      .distribution-bar__segment--cd{background:#facc15;}
      .distribution-legend{display:flex; justify-content:space-between; color:var(--muted); font-size:13px; margin-top:6px;}
      .chore-dashboard{display:flex; flex-direction:column; gap:20px;}
      .chore-header{display:flex; justify-content:space-between; align-items:flex-start; gap:16px;}
      .chore-header__date{text-align:right;}
      .chore-header__label{font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em;}
      .chore-header__value{font-size:18px; font-weight:700;}
      .chore-columns{display:grid; grid-template-columns:2fr 1fr; gap:20px;}
      .chore-column{display:flex; flex-direction:column; gap:12px;}
      .chore-column__headline{font-weight:600; font-size:14px; color:var(--muted);}
      .chore-intro{font-size:14px;}
      .chore-item{display:flex; justify-content:space-between; gap:16px; padding:12px 0; border-bottom:1px solid #243041;}
      .chore-item:last-child{border-bottom:none;}
      .chore-item__info{flex:1; min-width:0;}
      .chore-item__title{display:flex; align-items:center; gap:8px;}
      .chore-item__type{background:rgba(37,99,235,0.18); color:var(--text);}
      .chore-item__meta{font-size:13px;}
      .chore-item__schedule{font-size:12px; margin-top:4px;}
      .chore-item__action{display:flex; align-items:center;}
      .chore-table .chore-schedule{display:flex; flex-direction:column; gap:8px;}
      .chore-schedule__dates{display:flex; flex-wrap:wrap; gap:8px;}
      .chore-schedule__dates input{flex:1 1 140px; min-width:140px;}
      .chore-schedule__weekdays{display:flex; flex-wrap:wrap; gap:8px; font-size:14px;}
      .chore-schedule__weekday{display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border-radius:8px; background:rgba(148,163,184,0.12);}
      .chore-schedule__weekday input{width:auto; margin:0;}
      .chore-row__actions{display:flex; flex-direction:column; gap:8px; align-items:flex-end;}
      .chore-row__actions form{margin:0; display:flex; justify-content:flex-end; width:100%;}
      .chore-row__actions button{width:auto;}
      .chore-field--compact{max-width:160px;}
      .chore-empty{padding:14px; border-radius:10px; background:rgba(148,163,184,0.08);}
      .calendar-nav{display:flex; justify-content:space-between; align-items:center; gap:10px;}
      .calendar-nav__btn{padding:6px 12px; border-radius:999px; background:rgba(148,163,184,0.16); color:var(--text); text-decoration:none; font-size:13px;}
      .calendar-nav__btn:hover{filter:brightness(1.1);}
      .calendar-nav__title{font-weight:600;}
      .calendar-nav__today{font-size:13px; color:var(--accent); text-decoration:none; margin-top:4px; display:inline-block;}
      .calendar-table{width:100%; border-collapse:collapse;}
      .calendar-table th{padding:6px 4px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em;}
      .calendar-table td{padding:6px 4px; border:none; text-align:center;}
      .calendar-table a{display:flex; align-items:center; justify-content:center; width:36px; height:36px; margin:0 auto; border-radius:999px; text-decoration:none; color:var(--text); background:rgba(148,163,184,0.12); transition:filter .15s ease, transform .15s ease;}
      .calendar-table a:hover{filter:brightness(1.05); transform:scale(1.02);}
      .calendar-cell--selected a{background:var(--accent); color:#fff;}
      .calendar-cell--today a{border:2px solid var(--accent);}
      .calendar-cell--faded a{background:rgba(148,163,184,0.06); color:var(--muted);}
      .pill.status-available{background:rgba(148,163,184,0.18); color:var(--text);}
      .pill.status-pending{background:#f59e0b; color:#78350f;}
      .pill.status-paid{background:#16a34a; color:#dcfce7;}
      .admin-top{ grid-template-columns: repeat(12, minmax(0,1fr)); gap:16px; }
      .admin-top > .card{ grid-column: span 6; }
      @media (min-width:1100px){ .admin-top > .card{ grid-column: span 4; } }
      @media (min-width:1280px){ .admin-top > .card{ grid-column: span 3; } }
      @media (max-width:1150px){ .grid{ grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); } }
      @media (max-width:900px){ .grid{ grid-template-columns: 1fr; } }
      .notice{border-radius:12px; padding:14px 16px; margin:12px 0;}
      .notice.success{background:#dcfce7; border-left:4px solid #86efac; color:#166534;}
      .notice.error{background:#fee2e2; border-left:4px solid #fca5a5; color:#b91c1c;}
      .chart{width:100%; height:auto;}
      .chart--detail{background:rgba(148,163,184,0.08); border-radius:12px; padding:12px;}
      .chart-legend{display:flex; justify-content:space-between; color:var(--muted); font-size:13px; margin-top:6px; gap:8px; flex-wrap:wrap;}
      .chart-toggle{margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; align-items:center; color:var(--muted);}
      .chart-toggle a{padding:6px 10px; border-radius:999px; text-decoration:none; background:rgba(148,163,184,0.16); color:var(--text); font-size:13px;}
      .chart-toggle a.active{background:var(--accent); color:#fff;}
      .investing-grid{grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px;}
      .chart-popout{display:block; margin-top:12px; border-radius:14px; background:rgba(148,163,184,0.08); padding:12px; box-shadow:inset 0 0 0 1px rgba(148,163,184,0.12); transition:transform .18s ease, box-shadow .18s ease;}
      .chart-popout:hover,.chart-popout:focus{transform:scale(1.01); box-shadow:inset 0 0 0 1px rgba(148,163,184,0.24);}
      .chart-popout svg{width:100%; height:auto; display:block;}
      .chart-hint{margin-top:6px; font-size:12px; color:var(--muted);}
      .chart-modal__card{max-width:960px;}
      .chart-modal__body{max-height:75vh; overflow:auto;}
      .chart-modal__body svg{width:100%; height:auto;}
      .modal-overlay{position:fixed; inset:0; background:rgba(15,23,42,0.78); display:none; align-items:center; justify-content:center; padding:16px; z-index:1000;}
      .modal-overlay:target{display:flex;}
      .modal-card{background:var(--card); border-radius:12px; padding:20px; max-width:720px; width:100%; max-height:90vh; overflow:auto; box-shadow:0 28px 48px rgba(0,0,0,0.4);}
      .modal-head{display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px;}
      .modal-card table{margin-top:10px;}
      @media (max-width: 640px){
        .card{padding:12px}
        .layout{grid-template-columns:1fr;}
        .sidebar{flex-direction:row; position:static; overflow-x:auto;}
        .sidebar a{white-space:nowrap;}
        .hero-card{flex-direction:column; align-items:flex-start; gap:16px;}
        .hero-card__intro{max-width:100%;}
        .hero-card__balance{text-align:left;}
        .hero-card__amount{font-size:34px;}
        .overview-stats-grid{grid-template-columns:1fr;}
        .chore-columns{grid-template-columns:1fr;}
        .calendar-nav{flex-wrap:wrap;}
        .calendar-table a{width:32px; height:32px;}
        table, thead, tbody, th, td, tr { display:block; width:100%; }
        thead { display:none; }
        tr { margin-bottom:12px; border:1px solid #243041; border-radius:8px; padding:8px; background:var(--card); }
        td {
          border:none; border-bottom:1px solid #243041;
          position:relative; padding-left:52%; text-align:left !important;
          white-space:normal;
        }
        td:last-child { border-bottom:none; }
        td::before {
          position:absolute; top:8px; left:8px; width:45%;
          white-space:nowrap; font-weight:600; color:var(--muted);
          content: attr(data-label);
        }
        td[data-label="Actions"]{ padding-left:12px; }
        td[data-label="Actions"]::before{ position:static; display:block; margin-bottom:6px; width:auto; }
        td[data-label="Actions"] .right{ text-align:left; }
        td[data-label="Actions"] form,
        td[data-label="Actions"] a{ display:block; width:100%; margin:6px 0 0 0; }
        td[data-label="Actions"] button{ width:100%; }
        .chore-row__actions{align-items:stretch;}
        .chore-row__actions form{justify-content:stretch;}
        button{ width:100%; }
        .kiosk{flex-direction:column; align-items:flex-start}
        .kiosk .balance{font-size:42px}
        .modal-card{padding:16px; border-radius:10px; max-height:96vh;}
      }
      body.touch-friendly button,
      body.touch-friendly .button-link,
      body.touch-friendly .sidebar a{min-height:52px; padding-top:12px; padding-bottom:12px;}
      body.touch-friendly input,
      body.touch-friendly select,
      body.touch-friendly textarea{min-height:48px;}
      body.font-dyslexic{font-family:'OpenDyslexic','Comic Sans MS','Trebuchet MS',Verdana,sans-serif; letter-spacing:0.03em;}
      body.theme-high-contrast{--bg:#000814; --card:#001d3d; --muted:#e2e8f0; --accent:#ffd60a; --text:#f8fafc; --good:#4ade80; --bad:#f87171;}
      body.theme-high-contrast .card{box-shadow:none; border:2px solid rgba(248,250,252,0.18);}
      .help-icon{display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:999px; background:rgba(148,163,184,0.24); color:var(--text); font-size:12px; text-decoration:none; margin-left:6px;}
      .help-icon:hover{filter:brightness(1.12);}
      .ui-pref-group{display:flex; gap:6px; align-items:center; flex-wrap:wrap;}
      .ui-pref-form{display:inline;}
      .ui-pref-form button{padding:6px 10px; min-height:36px;}
      .ui-pref-form button[aria-pressed='true']{background:var(--accent); color:#fff;}
      .analytics-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin-top:16px;}
      .trend-bars{display:flex; align-items:flex-end; gap:6px; height:120px; margin-top:12px;}
      .trend-bars__bar{flex:1; position:relative; border-radius:8px 8px 0 0; background:rgba(59,130,246,0.32); min-height:6px;}
      .trend-bars__bar--completed{background:rgba(16,185,129,0.38);}
      .trend-bars__bar--approval{background:rgba(249,115,22,0.38);}
      .trend-bars__bar--interest{background:rgba(217,70,239,0.38);}
      .trend-bars__value{position:absolute; top:-22px; left:50%; transform:translateX(-50%); font-size:11px; color:var(--muted); white-space:nowrap;}
      .trend-bars__label{position:absolute; bottom:-18px; left:50%; transform:translateX(-50%); font-size:11px; color:var(--muted);}
      .jar-bar{display:flex; height:16px; border-radius:999px; overflow:hidden; background:rgba(148,163,184,0.16); margin-top:12px;}
      .jar-bar__segment{height:100%; position:relative;}
      .jar-bar__segment--cash{background:#38bdf8;}
      .jar-bar__segment--goals{background:#f97316;}
      .jar-bar__segment--market{background:#22c55e;}
      .jar-bar__segment--cd{background:#a855f7;}
      .jar-legend{display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; font-size:12px; color:var(--muted);}
      .jar-legend__item{display:flex; align-items:center; gap:6px;}
      .jar-legend__swatch{width:12px; height:12px; border-radius:3px;}
      .lesson-list{display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin-top:12px;}
      .lesson-card{padding:14px; border-radius:12px; background:rgba(37,99,235,0.12); display:flex; flex-direction:column; gap:8px;}
      .lesson-card__header{display:flex; justify-content:space-between; align-items:flex-start; gap:8px;}
      .lesson-card__summary{color:var(--muted); font-size:14px;}
      .lesson-card__meta{font-size:13px; color:var(--muted);}
      .quiz-question{margin-top:12px; padding:12px; border-radius:10px; background:rgba(148,163,184,0.12);}
      .quiz-question legend{font-weight:600;}
      .quiz-question label{display:block; margin-top:6px;}
      .goal-projection{margin-top:18px; padding:16px; border-radius:12px; background:rgba(148,163,184,0.12);}
      .goal-projection__metrics{margin-top:8px; font-size:14px;}
      .goal-projection__eta{font-weight:600;}
      input[type=range]{width:100%;}
      .tutorial-list{margin:0; padding-left:20px;}
      .tutorial-list li{margin:6px 0;}
      .badge-earned{background:#22c55e; color:#0b1220;}
    </style>
    """

def money_mask_js() -> str:
    return r"""
    <script>
    (function(){
      function fmt(d){
        d=(d||"").replace(/\D/g,'');
        if(d.length===0) return "0.00";
        if(d.length===1) return "0.0"+d;
        if(d.length===2) return "0."+d;
        return d.slice(0,-2)+"."+d.slice(-2);
      }
      function toDigits(v){ return (v||"").replace(/\D/g,''); }
      function attach(el){
        let init = toDigits(el.value);
        el.dataset.digits = init || "";
        el.value = fmt(el.dataset.digits);
        el.setAttribute('inputmode','numeric');
        el.setAttribute('autocomplete','off');
        el.setAttribute('autocorrect','off');
        el.setAttribute('autocapitalize','off');
        el.setAttribute('spellcheck','false');

        el.addEventListener('keydown', function(e){
          if(e.key==='Backspace'){
            e.preventDefault();
            let d = el.dataset.digits||"";
            d = d.slice(0,-1);
            el.dataset.digits = d;
            el.value = fmt(d);
          }else if(e.key.length===1 && /\d/.test(e.key)){
            e.preventDefault();
            let d = (el.dataset.digits||"") + e.key;
            el.dataset.digits = d;
            el.value = fmt(d);
          }
        });
        el.addEventListener('paste', function(e){
          e.preventDefault();
          const txt=(e.clipboardData||window.clipboardData).getData('text')||'';
          let d=(el.dataset.digits||"")+toDigits(txt);
          el.dataset.digits=d;
          el.value=fmt(d);
        });
        el.addEventListener('input', function(){ el.value=fmt(el.dataset.digits||""); });
        el.addEventListener('blur', function(){ el.value=fmt(el.dataset.digits||""); });
      }
      function ready(fn){ document.readyState!=='loading' ? fn() : document.addEventListener('DOMContentLoaded', fn); }
      ready(function(){
        document.querySelectorAll('input[data-money]').forEach(attach);
        document.querySelectorAll('form').forEach(function(f){
          f.addEventListener('submit', function(){
            f.querySelectorAll('input[data-money]').forEach(function(el){
              const d=(el.dataset.digits||"").replace(/\D/g,'');
              el.value = (d.length? (d.length===1?"0.0"+d:d.length===2?"0."+d:d.slice(0,-2)+"."+d.slice(-2)) : "0.00");
            });
          });
        });
      });
    })();
    </script>
    """


def frame(title: str, inner: str, head_extra: str = "", body_attrs: str = "") -> str:
    manifest_head = (
        "<link rel='manifest' href='/manifest.webmanifest'>"
        "<link rel='apple-touch-icon' href='/pwa-icon.svg'>"
        "<meta name='theme-color' content='#2563eb'>"
    )
    body_attr = f" {body_attrs.strip()}" if body_attrs else ""
    return (
        "<html><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'>{manifest_head}{head_extra}<title>{title}</title>"
        f"{base_styles()}{money_mask_js()}</head><body{body_attr}>{inner}{service_worker_registration()}</body></html>"
    )


def render_page(
    request: Optional[Request],
    title: str,
    inner: str,
    *,
    head_extra: str = "",
    status_code: int = 200,
) -> HTMLResponse:
    html = frame(title, inner, head_extra=head_extra, body_attrs=body_pref_attrs(request))
    return HTMLResponse(html, status_code=status_code)


def simple_markdown_to_html(md: str) -> str:
    blocks: List[str] = []
    for raw_block in md.split("\n\n"):
        block = raw_block.strip()
        if not block:
            continue
        if block.startswith("## "):
            blocks.append(f"<h3>{html_escape(block[3:].strip())}</h3>")
            continue
        if block.startswith("* "):
            items = []
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("* "):
                    items.append(f"<li>{html_escape(line[2:].strip())}</li>")
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue
        safe = html_escape(block)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\\1</strong>", safe)
        safe = re.sub(r"\*(.+?)\*", r"<em>\\1</em>", safe)
        blocks.append(f"<p>{safe}</p>")
    return "".join(blocks)


# ---------------------------------------------------------------------------
# Chore helpers
# ---------------------------------------------------------------------------
def _chore_weekday_set(raw: Optional[str]) -> Set[int]:
    values: Set[int] = set()
    if not raw:
        return values
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token) % 7)
        except ValueError:
            continue
    return values


def _chore_specific_dates(raw: Optional[str]) -> Set[date]:
    dates: Set[date] = set()
    if not raw:
        return dates
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            dates.add(date.fromisoformat(token))
        except ValueError:
            continue
    return dates


def chore_weekdays(chore: Chore) -> Set[int]:
    return _chore_weekday_set(chore.weekdays)


def chore_specific_dates(chore: Chore) -> Set[date]:
    return _chore_specific_dates(chore.specific_dates)


WEEKDAY_OPTIONS: Tuple[Tuple[int, str], ...] = (
    (0, "Mon"),
    (1, "Tue"),
    (2, "Wed"),
    (3, "Thu"),
    (4, "Fri"),
    (5, "Sat"),
    (6, "Sun"),
)


def serialize_weekday_selection(values: Iterable[str]) -> Optional[str]:
    days: Set[int] = set()
    for raw in values:
        try:
            day = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= day <= 6:
            days.add(day)
    if not days:
        return None
    return ",".join(str(day) for day in sorted(days))


def serialize_specific_dates(raw: str) -> Optional[str]:
    cleaned = []
    for part in (raw or "").split(","):
        value = part.strip()
        if not value:
            continue
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            continue
        cleaned.append(parsed.isoformat())
    if not cleaned:
        return None
    return ",".join(sorted(set(cleaned)))


def format_weekdays(days: Set[int]) -> str:
    labels = [label for value, label in WEEKDAY_OPTIONS if value in days]
    return ", ".join(labels)


def normalize_chore_type(value: str, *, is_global: bool = False) -> str:
    normalized = (value or "").lower()
    if normalized == "global":
        normalized = DEFAULT_GLOBAL_CHORE_TYPE
    if is_global:
        return normalized if normalized in GLOBAL_CHORE_TYPES else DEFAULT_GLOBAL_CHORE_TYPE
    valid = {"daily", "weekly", "monthly", "special"}
    return normalized if normalized in valid else "daily"


def period_key_for(chore_type: str, moment: datetime) -> str:
    if chore_type == "daily":
        return moment.strftime("%Y-%m-%d")
    if chore_type == "weekly":
        days_since_sunday = (moment.weekday() + 1) % 7
        sunday = (moment - timedelta(days=days_since_sunday)).date()
        return f"{sunday.isoformat()}-WEEK"
    if chore_type == "monthly":
        return moment.strftime("%Y-%m")
    return "SPECIAL"


def is_chore_in_window(chore: Chore, today: date) -> bool:
    if chore.start_date and today < chore.start_date:
        return False
    if chore.end_date and today > chore.end_date:
        return False
    if not chore.active:
        return False
    weekdays = chore_weekdays(chore)
    if weekdays and today.weekday() not in weekdays:
        return False
    specific_dates = chore_specific_dates(chore)
    if specific_dates and today not in specific_dates:
        return False
    return True


def global_chore_period_key(moment: datetime, chore: Optional[Chore] = None) -> str:
    if chore is not None:
        chore_type = normalize_chore_type(chore.type, is_global=True)
        if chore_type in {"weekly", "daily", "monthly"}:
            return period_key_for(chore_type, moment)
    return moment.strftime("%Y-%m-%d")


def count_global_claims(session: Session, chore_id: int, period_key: str, *, include_pending: bool = True) -> int:
    query = select(GlobalChoreClaim).where(
        GlobalChoreClaim.chore_id == chore_id,
        GlobalChoreClaim.period_key == period_key,
    )
    if not include_pending:
        query = query.where(GlobalChoreClaim.status == GLOBAL_CHORE_STATUS_APPROVED)
    else:
        query = query.where(
            GlobalChoreClaim.status.in_([GLOBAL_CHORE_STATUS_PENDING, GLOBAL_CHORE_STATUS_APPROVED])
        )
    return len(session.exec(query).all())


def kid_has_global_claim(session: Session, chore_id: int, kid_id: str, period_key: str) -> bool:
    claim = session.exec(
        select(GlobalChoreClaim)
        .where(GlobalChoreClaim.chore_id == chore_id)
        .where(GlobalChoreClaim.kid_id == kid_id)
        .where(GlobalChoreClaim.period_key == period_key)
        .where(GlobalChoreClaim.status.in_([GLOBAL_CHORE_STATUS_PENDING, GLOBAL_CHORE_STATUS_APPROVED]))
    ).first()
    return claim is not None


def get_global_claim(session: Session, chore_id: int, kid_id: str, period_key: str) -> Optional[GlobalChoreClaim]:
    return session.exec(
        select(GlobalChoreClaim)
        .where(GlobalChoreClaim.chore_id == chore_id)
        .where(GlobalChoreClaim.kid_id == kid_id)
        .where(GlobalChoreClaim.period_key == period_key)
        .order_by(desc(GlobalChoreClaim.submitted_at))
    ).first()


def ensure_instances_for_kid(kid_id: str) -> None:
    moment = now_local()
    today = moment.date()
    with Session(engine) as session:
        chores = session.exec(select(Chore).where(Chore.kid_id == kid_id, Chore.active == True)).all()  # noqa: E712
        for chore in chores:
            if not is_chore_in_window(chore, today):
                continue
            chore_type = normalize_chore_type(chore.type)
            if chore_type == "special":
                continue
            pk = period_key_for(chore_type, moment)
            exists = session.exec(
                select(ChoreInstance)
                .where(ChoreInstance.chore_id == chore.id)
                .where(ChoreInstance.period_key == pk)
            ).first()
            if not exists:
                session.add(ChoreInstance(chore_id=chore.id, period_key=pk, status="available"))
        session.commit()


def list_chore_instances_for_kid(
    kid_id: str, target_day: Optional[date] = None
) -> List[Tuple[Chore, Optional[ChoreInstance]]]:
    now_moment = now_local()
    selected_day = target_day or now_moment.date()
    if target_day is None or target_day == now_moment.date():
        ensure_instances_for_kid(kid_id)
    moment = datetime.combine(selected_day, datetime.min.time())
    today = selected_day
    pk_daily = period_key_for("daily", moment)
    pk_weekly = period_key_for("weekly", moment)
    pk_monthly = period_key_for("monthly", moment)
    with Session(engine) as session:
        chores = session.exec(select(Chore).where(Chore.kid_id == kid_id, Chore.active == True)).all()  # noqa: E712
        output: List[Tuple[Chore, Optional[ChoreInstance]]] = []
        for chore in chores:
            if not is_chore_in_window(chore, today):
                continue
            insts = session.exec(
                select(ChoreInstance)
                .where(ChoreInstance.chore_id == chore.id)
                .order_by(desc(ChoreInstance.id))
            ).all()
            current: Optional[ChoreInstance]
            chore_type = normalize_chore_type(chore.type)
            if chore_type == "daily":
                current = next((i for i in insts if i.period_key == pk_daily), None)
            elif chore_type == "weekly":
                current = next((i for i in insts if i.period_key == pk_weekly), None)
            elif chore_type == "monthly":
                current = next((i for i in insts if i.period_key == pk_monthly), None)
            else:
                current = next(
                    (
                        i
                        for i in insts
                        if i.status in {"available", "pending", CHORE_STATUS_PENDING_MARKETPLACE}
                    ),
                    None,
                )
            output.append((chore, current))
        return output


# ---------------------------------------------------------------------------
# Allowance + rules helpers
# ---------------------------------------------------------------------------
class MetaDAO:
    @staticmethod
    def get(session: Session, key: str) -> Optional[str]:
        row = session.get(MetaKV, key)
        return row.v if row else None

    @staticmethod
    def set(session: Session, key: str, value: str) -> None:
        row = session.get(MetaKV, key)
        if row:
            row.v = value
            session.add(row)
        else:
            session.add(MetaKV(k=key, v=value))


_time_settings: Dict[str, Any] = {
    "mode": TIME_MODE_AUTO,
    "offset": 0,
    "manual": None,
    "manual_ref": None,
}


def _compute_now_from_settings() -> datetime:
    settings = _time_settings
    mode = settings.get("mode", TIME_MODE_AUTO)
    try:
        offset_minutes = int(settings.get("offset", 0) or 0)
    except (TypeError, ValueError):
        offset_minutes = 0
    if mode == TIME_MODE_MANUAL:
        manual_raw = settings.get("manual")
        manual_ref_raw = settings.get("manual_ref")
        if manual_raw:
            try:
                manual_dt = datetime.fromisoformat(manual_raw)
                if manual_ref_raw:
                    try:
                        reference_dt = datetime.fromisoformat(manual_ref_raw)
                        delta = datetime.utcnow() - reference_dt
                        return manual_dt + delta
                    except ValueError:
                        pass
                return manual_dt
            except ValueError:
                pass
    return datetime.utcnow() + timedelta(minutes=offset_minutes)


def _load_time_settings(active_session: Session) -> Dict[str, Any]:
    mode = MetaDAO.get(active_session, TIME_META_MODE_KEY) or TIME_MODE_AUTO
    if mode not in {TIME_MODE_AUTO, TIME_MODE_MANUAL}:
        mode = TIME_MODE_AUTO
    raw_offset = MetaDAO.get(active_session, TIME_META_OFFSET_KEY) or "0"
    try:
        offset_minutes = int(raw_offset)
    except ValueError:
        offset_minutes = 0
    manual = MetaDAO.get(active_session, TIME_META_MANUAL_KEY) or None
    manual_ref = MetaDAO.get(active_session, TIME_META_MANUAL_REF_KEY) or None
    return {
        "mode": mode,
        "offset": offset_minutes,
        "manual": manual or None,
        "manual_ref": manual_ref or None,
    }


def refresh_time_settings(session: Session | None = None) -> None:
    if session is None:
        with Session(engine) as new_session:
            settings = _load_time_settings(new_session)
    else:
        settings = _load_time_settings(session)
    _time_settings.update(settings)


def get_time_settings(session: Session | None = None) -> Dict[str, Any]:
    if session is None:
        with Session(engine) as new_session:
            return _load_time_settings(new_session)
    return _load_time_settings(session)


def set_time_settings(mode: str, offset_minutes: int, manual_iso: Optional[str]) -> None:
    normalized_mode = mode if mode in {TIME_MODE_AUTO, TIME_MODE_MANUAL} else TIME_MODE_AUTO
    manual_clean = manual_iso.strip() if manual_iso else None
    manual_reference = None
    if normalized_mode == TIME_MODE_MANUAL and manual_clean:
        try:
            datetime.fromisoformat(manual_clean)
            manual_reference = datetime.utcnow().isoformat()
        except ValueError:
            manual_clean = None
            normalized_mode = TIME_MODE_AUTO
    with Session(engine) as session:
        MetaDAO.set(session, TIME_META_MODE_KEY, normalized_mode)
        MetaDAO.set(session, TIME_META_OFFSET_KEY, str(int(offset_minutes)))
        if manual_clean and normalized_mode == TIME_MODE_MANUAL:
            MetaDAO.set(session, TIME_META_MANUAL_KEY, manual_clean)
            MetaDAO.set(session, TIME_META_MANUAL_REF_KEY, manual_reference or "")
        else:
            MetaDAO.set(session, TIME_META_MANUAL_KEY, "")
            MetaDAO.set(session, TIME_META_MANUAL_REF_KEY, "")
        session.commit()
        refresh_time_settings(session)


_time_provider = _compute_now_from_settings
refresh_time_settings()


def _parent_pin_default(role: str) -> str:
    if role == "mom":
        return MOM_PIN
    if role == "dad":
        return DAD_PIN
    return ""


def _parent_pin_meta_key(role: str) -> str:
    return f"parent_pin_{role}"


def _normalize_parent_role_key(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", (label or "").lower())
    return slug or "admin"


def _load_extra_parent_admins(session: Session) -> List[Dict[str, str]]:
    raw = MetaDAO.get(session, EXTRA_PARENT_ADMINS_KEY) or "[]"
    try:
        data = json.loads(raw)
    except Exception:
        data = []
    cleaned: List[Dict[str, str]] = []
    for entry in data:
        role = str(entry.get("role") or "").strip().lower()
        label = str(entry.get("label") or "").strip()
        if not role:
            continue
        cleaned.append({"role": role, "label": label or role.title()})
    return cleaned


def all_parent_admins(session: Session | None = None) -> List[Dict[str, str]]:
    def load(active: Session) -> List[Dict[str, str]]:
        base = [
            {"role": role, "label": DEFAULT_PARENT_LABELS.get(role, role.title())}
            for role in DEFAULT_PARENT_ROLES
        ]
        extras = _load_extra_parent_admins(active)
        seen: Set[str] = {item["role"] for item in base}
        unique_extras: List[Dict[str, str]] = []
        for entry in extras:
            role = entry["role"]
            if role in seen:
                continue
            seen.add(role)
            unique_extras.append(entry)
        return base + unique_extras

    if session is not None:
        return load(session)
    with Session(engine) as new_session:
        return load(new_session)


def parent_role_label(role: str, session: Session | None = None) -> str:
    normalized = (role or "").lower()
    if not normalized:
        return ""
    admins = all_parent_admins(session)
    for admin in admins:
        if admin["role"] == normalized:
            return admin["label"]
    return normalized.title()


def get_parent_pins(session: Session | None = None) -> Dict[str, str]:
    def load(active_session: Session) -> Dict[str, str]:
        pins: Dict[str, str] = {}
        for admin in all_parent_admins(active_session):
            role = admin["role"]
            override = MetaDAO.get(active_session, _parent_pin_meta_key(role))
            if override:
                pins[role] = override
            else:
                default_pin = _parent_pin_default(role)
                if default_pin:
                    pins[role] = default_pin
        return pins

    if session is not None:
        return load(session)
    with Session(engine) as new_session:
        return load(new_session)


def resolve_admin_role(pin: str, *, session: Session | None = None) -> Optional[str]:
    pins = get_parent_pins(session)
    for role, value in pins.items():
        if pin == value:
            return role
    return None


def set_parent_pin(role: str, new_pin: str) -> None:
    normalized_role = (role or "").lower()
    with Session(engine) as session:
        if normalized_role not in {admin["role"] for admin in all_parent_admins(session)}:
            raise ValueError(f"Unknown parent role: {role}")
        MetaDAO.set(session, _parent_pin_meta_key(normalized_role), new_pin)
        session.commit()


ADMIN_PRIV_META_PREFIX = "admin_privileges:"


@dataclass
class AdminPrivileges:
    role: str
    kid_scope: str = "all"
    kid_ids: List[str] = field(default_factory=list)
    max_credit_cents: Optional[int] = None
    max_debit_cents: Optional[int] = None
    can_manage_payouts: bool = True
    can_manage_chores: bool = True
    can_manage_time: bool = True
    can_manage_allowance: bool = True
    can_manage_prizes: bool = True
    can_create_accounts: bool = True
    can_delete_accounts: bool = True
    can_adjust_balances: bool = True
    can_transfer_funds: bool = True
    can_create_admins: bool = True
    can_delete_admins: bool = True
    can_change_admin_pins: bool = True
    can_manage_investing: bool = True
    _kid_lookup: Set[str] = field(init=False, repr=False, default_factory=set)

    def __post_init__(self) -> None:
        scope = self.kid_scope if self.kid_scope in {"all", "custom"} else "all"
        object.__setattr__(self, "kid_scope", scope)
        cleaned_ids = sorted({kid.strip() for kid in self.kid_ids if kid and kid.strip()})
        object.__setattr__(self, "kid_ids", cleaned_ids)
        object.__setattr__(self, "_kid_lookup", {kid.lower() for kid in cleaned_ids})
        credit_limit = self.max_credit_cents if isinstance(self.max_credit_cents, int) else None
        debit_limit = self.max_debit_cents if isinstance(self.max_debit_cents, int) else None
        if credit_limit is not None and credit_limit < 0:
            credit_limit = None
        if debit_limit is not None and debit_limit < 0:
            debit_limit = None
        object.__setattr__(self, "max_credit_cents", credit_limit)
        object.__setattr__(self, "max_debit_cents", debit_limit)

    @property
    def is_all_kids(self) -> bool:
        return self.kid_scope != "custom"

    def allows_kid(self, kid_id: str) -> bool:
        if not kid_id or self.is_all_kids:
            return True
        if kid_id == GLOBAL_CHORE_KID_ID:
            return True
        return kid_id.lower() in self._kid_lookup

    def credit_allowed(self, amount_cents: int) -> bool:
        if amount_cents <= 0:
            return True
        if self.max_credit_cents is None:
            return True
        return amount_cents <= self.max_credit_cents

    def debit_allowed(self, amount_cents: int) -> bool:
        if amount_cents <= 0:
            return True
        if self.max_debit_cents is None:
            return True
        return amount_cents <= self.max_debit_cents

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kid_scope": self.kid_scope,
            "kid_ids": list(self.kid_ids),
            "max_credit_cents": self.max_credit_cents,
            "max_debit_cents": self.max_debit_cents,
            "can_manage_payouts": self.can_manage_payouts,
            "can_manage_chores": self.can_manage_chores,
            "can_manage_time": self.can_manage_time,
            "can_manage_allowance": self.can_manage_allowance,
            "can_manage_prizes": self.can_manage_prizes,
            "can_create_accounts": self.can_create_accounts,
            "can_delete_accounts": self.can_delete_accounts,
            "can_adjust_balances": self.can_adjust_balances,
            "can_transfer_funds": self.can_transfer_funds,
            "can_create_admins": self.can_create_admins,
            "can_delete_admins": self.can_delete_admins,
            "can_change_admin_pins": self.can_change_admin_pins,
            "can_manage_investing": self.can_manage_investing,
        }

    @classmethod
    def from_dict(cls, role: str, data: Dict[str, Any]) -> "AdminPrivileges":
        if not isinstance(data, dict):
            data = {}

        def _bool(name: str, default: bool) -> bool:
            value = data.get(name, default)
            return bool(value) if not isinstance(value, str) else value.lower() in {"1", "true", "yes", "on"}

        def _int_or_none(name: str) -> Optional[int]:
            value = data.get(name)
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        return cls(
            role=role,
            kid_scope=str(data.get("kid_scope", "all")),
            kid_ids=list(data.get("kid_ids", [])),
            max_credit_cents=_int_or_none("max_credit_cents"),
            max_debit_cents=_int_or_none("max_debit_cents"),
            can_manage_payouts=_bool("can_manage_payouts", True),
            can_manage_chores=_bool("can_manage_chores", True),
            can_manage_time=_bool("can_manage_time", True),
            can_manage_allowance=_bool("can_manage_allowance", True),
            can_manage_prizes=_bool("can_manage_prizes", True),
            can_create_accounts=_bool("can_create_accounts", True),
            can_delete_accounts=_bool("can_delete_accounts", True),
            can_adjust_balances=_bool("can_adjust_balances", True),
            can_transfer_funds=_bool("can_transfer_funds", True),
            can_create_admins=_bool("can_create_admins", True),
            can_delete_admins=_bool("can_delete_admins", True),
            can_change_admin_pins=_bool("can_change_admin_pins", True),
            can_manage_investing=_bool("can_manage_investing", True),
        )

    @classmethod
    def default(cls, role: str) -> "AdminPrivileges":
        return cls(role=role)


def _admin_priv_key(role: str) -> str:
    return f"{ADMIN_PRIV_META_PREFIX}{role}"


def load_admin_privileges(session: Session, role: str) -> AdminPrivileges:
    normalized = (role or "").strip().lower()
    if not normalized:
        return AdminPrivileges.default("admin")
    if normalized == "dad":
        return AdminPrivileges.default(normalized)
    raw = MetaDAO.get(session, _admin_priv_key(normalized))
    data: Dict[str, Any] = {}
    if raw:
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
    return AdminPrivileges.from_dict(normalized, data)


def save_admin_privileges(session: Session, privileges: AdminPrivileges) -> None:
    if privileges.role in {role for role in DEFAULT_PARENT_ROLES}:
        return
    MetaDAO.set(session, _admin_priv_key(privileges.role), json.dumps(privileges.to_dict()))


def delete_admin_privileges(session: Session, role: str) -> None:
    normalized = (role or "").strip().lower()
    if not normalized:
        return
    if normalized in DEFAULT_PARENT_ROLES:
        return
    key = _admin_priv_key(normalized)
    entry = session.get(MetaKV, key)
    if entry:
        session.delete(entry)


def current_admin_privileges(request: Request, session: Session | None = None) -> AdminPrivileges:
    cached = getattr(request.state, "_admin_privileges", None)
    if cached is not None:
        return cached  # type: ignore[return-value]
    role = admin_role(request) or ""
    if session is None:
        with Session(engine) as temp:
            privileges = load_admin_privileges(temp, role)
    else:
        privileges = load_admin_privileges(session, role)
    request.state._admin_privileges = privileges  # type: ignore[attr-defined]
    return privileges


def admin_forbidden(request: Request, message: str, redirect: str = "/admin") -> RedirectResponse:
    set_admin_notice(request, message, "error")
    return RedirectResponse(redirect, status_code=302)


def require_admin_permission(
    request: Request,
    attribute: str,
    *,
    redirect: str = "/admin",
) -> Optional[RedirectResponse]:
    if (redirect_response := require_admin(request)) is not None:
        return redirect_response
    privileges = current_admin_privileges(request)
    allowed = getattr(privileges, attribute, False)
    if not allowed:
        return admin_forbidden(request, "You do not have permission to perform that action.", redirect)
    return None


def ensure_admin_kid_access(
    request: Request,
    kid_id: Optional[str],
    *,
    redirect: str = "/admin",
) -> Optional[RedirectResponse]:
    if not kid_id:
        return None
    privileges = current_admin_privileges(request)
    if privileges.allows_kid(kid_id):
        return None
    return admin_forbidden(request, "You do not have access to that kid.", redirect)


def ensure_admin_amount_within_limits(
    request: Request,
    amount_cents: int,
    kind: str,
    *,
    redirect: str = "/admin",
) -> Optional[RedirectResponse]:
    privileges = current_admin_privileges(request)
    if kind == "credit" and not privileges.credit_allowed(amount_cents):
        limit = privileges.max_credit_cents
        if limit is not None:
            return admin_forbidden(
                request,
                f"Credits above {usd(limit)} require approval from a full administrator.",
                redirect,
            )
    if kind == "debit" and not privileges.debit_allowed(amount_cents):
        limit = privileges.max_debit_cents
        if limit is not None:
            return admin_forbidden(
                request,
                f"Debits above {usd(limit)} require approval from a full administrator.",
                redirect,
            )
    return None


def _parse_rate(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def get_cd_rate_bps(session: Session, term_code: Optional[str] = None) -> int:
    normalized = (term_code or "").strip().lower()
    if normalized:
        specific = _parse_rate(MetaDAO.get(session, _cd_rate_meta_key(normalized)))
        if specific is not None:
            return specific
    default_specific = _parse_rate(MetaDAO.get(session, _cd_rate_meta_key(DEFAULT_CD_TERM_CODE)))
    if default_specific is not None:
        return default_specific
    legacy = _parse_rate(MetaDAO.get(session, CD_RATE_KEY))
    if legacy is not None:
        return legacy
    return DEFAULT_CD_RATE_BPS


def get_all_cd_rate_bps(session: Session) -> Dict[str, int]:
    return {code: get_cd_rate_bps(session, code) for code, _, _ in CD_TERM_OPTIONS}


def get_cd_penalty_days(session: Session, term_code: Optional[str] = None) -> int:
    normalized = (term_code or "").strip().lower()
    if normalized:
        specific_raw = MetaDAO.get(session, _cd_penalty_meta_key(normalized))
        if specific_raw is not None:
            try:
                return max(0, int(specific_raw))
            except ValueError:
                pass
    default_raw = MetaDAO.get(session, _cd_penalty_meta_key(DEFAULT_CD_TERM_CODE))
    if default_raw is not None:
        try:
            return max(0, int(default_raw))
        except ValueError:
            pass
    legacy_raw = MetaDAO.get(session, CD_PENALTY_DAYS_KEY)
    if legacy_raw is not None:
        try:
            return max(0, int(legacy_raw))
        except ValueError:
            pass
    return DEFAULT_CD_PENALTY_DAYS


def get_all_cd_penalty_days(session: Session) -> Dict[str, int]:
    penalties: Dict[str, int] = {}
    for code, _label, _days in CD_TERM_OPTIONS:
        penalties[code] = get_cd_penalty_days(session, code)
    return penalties


def resolve_certificate_term(selection: str) -> Tuple[str, int, int]:
    choice = (selection or "").strip().lower()
    if choice in CD_TERM_LOOKUP:
        _, days = CD_TERM_LOOKUP[choice]
        months = days // 30 if days % 30 == 0 else 0
        return choice, days, months
    if choice.endswith("m") and choice[:-1].isdigit():
        months = max(1, int(choice[:-1]))
        days = months * 30
        return f"{months}m", days, months
    if choice.endswith("w") and choice[:-1].isdigit():
        weeks = max(1, int(choice[:-1]))
        days = weeks * 7
        return f"{weeks}w", days, 0
    if choice.endswith("d") and choice[:-1].isdigit():
        days = max(1, int(choice[:-1]))
        months = days // 30 if days % 30 == 0 else 0
        if months and months * 30 == days:
            return f"{months}m", days, months
        return f"{days}d", days, months
    if choice.isdigit():
        months = max(1, int(choice))
        days = months * 30
        return f"{months}m", days, months
    _, default_days = CD_TERM_LOOKUP[DEFAULT_CD_TERM_CODE]
    default_months = default_days // 30 if default_days % 30 == 0 else 0
    return DEFAULT_CD_TERM_CODE, default_days, default_months


def certificate_term_days(certificate: Certificate) -> int:
    if getattr(certificate, "term_days", 0):
        return max(0, certificate.term_days)
    return max(0, certificate.term_months) * 30


def certificate_term_label(certificate: Certificate) -> str:
    days = certificate_term_days(certificate)
    if days <= 0:
        months = max(0, certificate.term_months)
        if months > 0:
            return f"{months} month{'s' if months != 1 else ''}"
        return "No term"
    if days % 30 == 0:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''}"
    if days % 7 == 0:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks != 1 else ''}"
    return f"{days} day{'s' if days != 1 else ''}"


def certificate_term_code(certificate: Certificate) -> str:
    days = certificate_term_days(certificate)
    if days <= 0:
        months = max(0, certificate.term_months)
        if months > 0:
            return f"{months}m"
        return "0d"
    if days % 30 == 0:
        months = days // 30
        return f"{months}m"
    if days % 7 == 0:
        weeks = days // 7
        return f"{weeks}w"
    return f"{days}d"


def certificate_maturity_date(certificate: Certificate) -> datetime:
    return certificate.opened_at + timedelta(days=certificate_term_days(certificate))


def _decimal_to_cents(value: Decimal) -> int:
    return int((value * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _certificate_compound_value(
    principal_cents: int, rate_bps: int, days: float, total_days: float
) -> Decimal:
    principal = Decimal(principal_cents) / Decimal(100)
    rate = Decimal(max(0, rate_bps)) / Decimal(10000)
    if total_days <= 0 or rate <= Decimal("0") or days <= 0:
        return principal.quantize(Decimal("0.01"))
    clamped_days = max(0.0, min(days, total_days))
    if clamped_days <= 0:
        return principal.quantize(Decimal("0.01"))
    try:
        fraction = clamped_days / total_days
    except ZeroDivisionError:  # pragma: no cover - defensive
        return principal.quantize(Decimal("0.01"))
    try:
        growth = math.exp(math.log1p(float(rate)) * fraction)
    except ValueError:
        growth = 1.0
    try:
        return (principal * Decimal(str(growth))).quantize(Decimal("0.01"))
    except (ValueError, ArithmeticError):  # pragma: no cover - defensive
        return principal.quantize(Decimal("0.01"))


def _certificate_elapsed_days(
    certificate: Certificate, *, at: Optional[datetime] = None
) -> float:
    total_days = float(max(0, certificate_term_days(certificate)))
    if total_days <= 0:
        return 0.0
    moment = at or _time_provider()
    if certificate.matured_at:
        moment = min(moment, certificate.matured_at)
    elapsed_seconds = (moment - certificate.opened_at).total_seconds()
    elapsed_days = elapsed_seconds / 86400.0
    return max(0.0, min(elapsed_days, total_days))


def certificate_maturity_value_cents(certificate: Certificate) -> int:
    total_days = float(certificate_term_days(certificate))
    return _decimal_to_cents(
        _certificate_compound_value(
            certificate.principal_cents,
            certificate.rate_bps,
            total_days,
            total_days,
        )
    )


def _certificate_elapsed_fraction(certificate: Certificate, *, at: Optional[datetime] = None) -> float:
    total_days = certificate_term_days(certificate)
    if total_days <= 0:
        return 1.0
    elapsed_days = _certificate_elapsed_days(certificate, at=at)
    return min(1.0, max(0.0, elapsed_days / total_days))


def certificate_value_cents(certificate: Certificate, *, at: Optional[datetime] = None) -> int:
    elapsed_days = _certificate_elapsed_days(certificate, at=at)
    total_days = float(certificate_term_days(certificate))
    return _decimal_to_cents(
        _certificate_compound_value(
            certificate.principal_cents,
            certificate.rate_bps,
            elapsed_days,
            total_days,
        )
    )


def certificate_penalty_cents(certificate: Certificate, *, at: Optional[datetime] = None) -> int:
    if certificate.penalty_days <= 0:
        return 0
    if certificate.matured_at is not None:
        return 0
    moment = at or _time_provider()
    if moment >= certificate_maturity_date(certificate):
        return 0
    elapsed_days = _certificate_elapsed_days(certificate, at=moment)
    if elapsed_days <= 0:
        return 0
    penalty_window = min(float(certificate.penalty_days), elapsed_days)
    if penalty_window <= 0:
        return 0
    total_days = float(certificate_term_days(certificate))
    current_value = _certificate_compound_value(
        certificate.principal_cents,
        certificate.rate_bps,
        elapsed_days,
        total_days,
    )
    prior_value = _certificate_compound_value(
        certificate.principal_cents,
        certificate.rate_bps,
        elapsed_days - penalty_window,
        total_days,
    )
    penalty_value = max(Decimal("0.00"), current_value - prior_value)
    principal_value = Decimal(certificate.principal_cents) / Decimal(100)
    accrued = max(Decimal("0.00"), current_value - principal_value)
    penalty_value = min(penalty_value, accrued)
    return _decimal_to_cents(penalty_value)


def certificate_sale_breakdown_cents(
    certificate: Certificate, *, at: Optional[datetime] = None
) -> Tuple[int, int, int]:
    gross = certificate_value_cents(certificate, at=at)
    penalty = certificate_penalty_cents(certificate, at=at)
    net = max(0, gross - penalty)
    return gross, penalty, net


def certificate_progress_percent(certificate: Certificate, *, at: Optional[datetime] = None) -> float:
    if certificate.matured_at:
        return 100.0
    total_days = certificate_term_days(certificate)
    if total_days <= 0:
        return 0.0
    elapsed_days = _certificate_elapsed_days(certificate, at=at)
    pct = (elapsed_days / total_days) * 100
    return min(100.0, max(0.0, pct))


def sunday_key(moment: datetime) -> str:
    days_since_sunday = (moment.weekday() + 1) % 7
    sunday = (moment - timedelta(days=days_since_sunday)).date()
    return sunday.isoformat()


def chores_completion_for_week(session: Session, kid_id: str, week_iso: str) -> Tuple[int, int]:
    week_start = date.fromisoformat(week_iso)
    week_end = week_start + timedelta(days=6)
    chores = session.exec(select(Chore).where(Chore.kid_id == kid_id, Chore.active == True)).all()  # noqa: E712
    expected = 0
    paid = 0
    for chore in chores:
        if chore.start_date and week_end < chore.start_date:
            continue
        if chore.end_date and week_start > chore.end_date:
            continue
        if chore.type == "daily":
            for offset in range(7):
                day = week_start + timedelta(days=offset)
                if chore.start_date and day < chore.start_date:
                    continue
                if chore.end_date and day > chore.end_date:
                    continue
                expected += 1
                pk = day.strftime("%Y-%m-%d")
                inst = session.exec(
                    select(ChoreInstance)
                    .where(ChoreInstance.chore_id == chore.id)
                    .where(ChoreInstance.period_key == pk)
                    .where(ChoreInstance.status == "paid")
                ).first()
                if inst:
                    paid += 1
        elif chore.type == "weekly":
            expected += 1
            pk = f"{week_start.isoformat()}-WEEK"
            inst = session.exec(
                select(ChoreInstance)
                .where(ChoreInstance.chore_id == chore.id)
                .where(ChoreInstance.period_key == pk)
                .where(ChoreInstance.status == "paid")
            ).first()
            if inst:
                paid += 1
    return paid, expected


def run_weekly_allowance_if_needed() -> None:
    moment = now_local()
    cache_key = "allowance_last_sunday"
    with Session(engine) as session:
        last = MetaDAO.get(session, cache_key)
        current = sunday_key(moment)
        if last == current:
            return
        bonus_on_all = MetaDAO.get(session, "rule_bonus_all_complete") == "1"
        bonus_cents = int(MetaDAO.get(session, "rule_bonus_cents") or "0")
        penalty_on_miss = MetaDAO.get(session, "rule_penalty_on_miss") == "1"
        penalty_cents = int(MetaDAO.get(session, "rule_penalty_cents") or "0")
        kids = session.exec(select(Child)).all()
        for child in kids:
            if (child.allowance_cents or 0) > 0:
                child.balance_cents += child.allowance_cents
                child.updated_at = datetime.utcnow()
                session.add(child)
                session.add(Event(child_id=child.kid_id, change_cents=child.allowance_cents, reason="weekly_allowance"))
            paid, expected = chores_completion_for_week(session, child.kid_id, current)
            if expected > 0:
                if bonus_on_all and paid == expected and bonus_cents > 0:
                    child.balance_cents += bonus_cents
                    session.add(Event(child_id=child.kid_id, change_cents=bonus_cents, reason="weekly_bonus_all_complete"))
                elif penalty_on_miss and paid < expected and penalty_cents > 0:
                    deduction = min(penalty_cents, child.balance_cents)
                    if deduction > 0:
                        child.balance_cents -= deduction
                        session.add(Event(child_id=child.kid_id, change_cents=-deduction, reason="weekly_penalty_missed"))
                child.updated_at = datetime.utcnow()
                session.add(child)
        MetaDAO.set(session, cache_key, current)
        session.commit()

# ---------------------------------------------------------------------------
# Investing helpers (market instruments, live data with cached fallback)
# ---------------------------------------------------------------------------
def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def ensure_default_instrument() -> None:
    normalized = _normalize_symbol(DEFAULT_MARKET_SYMBOL)
    with Session(engine) as session:
        existing = session.exec(
            select(MarketInstrument).where(MarketInstrument.symbol == normalized)
        ).first()
        if not existing:
            session.add(
                MarketInstrument(
                    symbol=normalized,
                    name="S&P 500 Fund",
                    kind=INSTRUMENT_KIND_STOCK,
                )
            )
            session.commit()


def ensure_default_learning_content() -> None:
    sample_lessons = [
        {
            "title": "Why saving matters",
            "summary": "Set goals, save steadily, and watch your money grow.",
            "content": textwrap.dedent(
                """
                ## Saving builds choices

                Every dollar you set aside gives you more options later. Goals help
                you stay focused and track your progress.

                * Pick something you care about.
                * Break the goal into weekly mini-targets.
                * Celebrate each deposit!

                ## Smart habits

                Try the 50/40/10 idea: 50% for saving goals, 40% for fun spending,
                and 10% for sharing or gifts. Adjust the mix to fit your family rules.
                """
            ).strip(),
            "quiz": {
                "questions": [
                    {
                        "prompt": "What is the first step when saving for something big?",
                        "options": [
                            "Spend money right away",
                            "Choose a goal you care about",
                            "Wait for someone to buy it",
                        ],
                        "answer": 1,
                    },
                    {
                        "prompt": "How can mini-targets help you?",
                        "options": [
                            "They make progress easier to see",
                            "They cost more money",
                            "They replace your goal",
                        ],
                        "answer": 0,
                    },
                ],
                "passing_score": 2,
                "reward_cents": 50,
            },
        },
        {
            "title": "Understanding interest",
            "summary": "Interest is a small thank-you for letting money rest.",
            "content": textwrap.dedent(
                """
                ## What is interest?

                Banks and parents sometimes pay extra money when you leave your
                savings alone. This extra is called *interest*.

                ## How to earn it

                * Deposit money into a savings jar or certificate.
                * Wait for the agreed time.
                * Collect the bonus interest on top of your savings.

                The longer your money stays put, the more interest you may earn.
                """
            ).strip(),
            "quiz": {
                "questions": [
                    {
                        "prompt": "What do you call extra money paid for saving?",
                        "options": ["Allowance", "Interest", "Debt"],
                        "answer": 1,
                    },
                    {
                        "prompt": "How do you earn more interest?",
                        "options": [
                            "Spend the money quickly",
                            "Leave the money saved longer",
                            "Hide it under a pillow",
                        ],
                        "answer": 1,
                    },
                ],
                "passing_score": 2,
                "reward_cents": 75,
            },
        },
    ]
    with Session(engine) as session:
        exists = session.exec(select(Lesson.id).limit(1)).first()
        if exists:
            return
        for entry in sample_lessons:
            lesson = Lesson(
                title=entry["title"],
                content_md=entry["content"],
                summary=entry["summary"],
            )
            session.add(lesson)
            session.commit()
            session.refresh(lesson)
            quiz_payload = entry.get("quiz")
            if quiz_payload:
                reward = int(quiz_payload.get("reward_cents", 0))
                payload = json.dumps(quiz_payload)
                session.add(
                    Quiz(
                        lesson_id=lesson.id,
                        payload=payload,
                        reward_cents=reward,
                    )
                )
                session.commit()


def list_market_instruments(session: Session | None = None) -> List[MarketInstrument]:
    def _load(active: Session) -> List[MarketInstrument]:
        return (
            active.exec(select(MarketInstrument).order_by(MarketInstrument.symbol)).all()
        )

    if session is not None:
        return _load(session)
    with Session(engine) as new_session:
        return _load(new_session)


def list_kid_market_symbols(
    kid_id: str, session: Session | None = None
) -> List[str]:
    normalized_default = _normalize_symbol(DEFAULT_MARKET_SYMBOL)

    def _load(active: Session) -> List[str]:
        rows = active.exec(
            select(KidMarketInstrument)
            .where(KidMarketInstrument.kid_id == kid_id)
            .order_by(KidMarketInstrument.created_at)
        ).all()
        symbols: List[str] = []
        for row in rows:
            normalized = _normalize_symbol(row.symbol)
            if normalized and normalized not in symbols:
                symbols.append(normalized)
        if normalized_default not in symbols:
            return [normalized_default] + symbols
        ordered = [normalized_default]
        ordered.extend(sym for sym in symbols if sym != normalized_default)
        return ordered

    if session is not None:
        return _load(session)
    with Session(engine) as new_session:
        return _load(new_session)


def list_market_instruments_for_kid(
    kid_id: str, session: Session | None = None
) -> List[MarketInstrument]:
    def _load(active: Session) -> List[MarketInstrument]:
        symbols = list_kid_market_symbols(kid_id, session=active)
        if not symbols:
            symbols = [_normalize_symbol(DEFAULT_MARKET_SYMBOL)]
        fetched = active.exec(
            select(MarketInstrument).where(
                MarketInstrument.symbol.in_(symbols)
            )
        ).all()
        lookup = {
            _normalize_symbol(inst.symbol): inst
            for inst in fetched
        }
        instruments: List[MarketInstrument] = []
        for symbol in symbols:
            normalized = _normalize_symbol(symbol)
            inst = lookup.get(normalized)
            if not inst:
                inst = MarketInstrument(symbol=normalized, name=normalized, kind=INSTRUMENT_KIND_STOCK)
            instruments.append(inst)
        return instruments

    if session is not None:
        return _load(session)
    with Session(engine) as new_session:
        return _load(new_session)


def get_market_instrument(symbol: str, session: Session | None = None) -> Optional[MarketInstrument]:
    normalized = _normalize_symbol(symbol)

    def _load(active: Session) -> Optional[MarketInstrument]:
        return active.exec(
            select(MarketInstrument).where(MarketInstrument.symbol == normalized)
        ).first()

    if session is not None:
        return _load(session)
    with Session(engine) as new_session:
        return _load(new_session)


def add_market_instrument(symbol: str, name: str, kind: str) -> Optional[MarketInstrument]:
    normalized_symbol = _normalize_symbol(symbol)
    if not normalized_symbol:
        return None
    normalized_kind = kind if kind in {INSTRUMENT_KIND_STOCK, INSTRUMENT_KIND_CRYPTO} else INSTRUMENT_KIND_STOCK
    friendly_name = name.strip() if name else normalized_symbol
    with Session(engine) as session:
        existing = session.exec(
            select(MarketInstrument).where(MarketInstrument.symbol == normalized_symbol)
        ).first()
        if existing:
            existing.name = friendly_name
            existing.kind = normalized_kind
            session.add(existing)
            session.commit()
            session.refresh(existing)
            return existing
        instrument = MarketInstrument(
            symbol=normalized_symbol,
            name=friendly_name,
            kind=normalized_kind,
        )
        session.add(instrument)
        session.commit()
        session.refresh(instrument)
        return instrument


def search_market_symbols(query: str, *, limit: int = 6) -> List[Dict[str, str]]:
    cleaned = (query or "").strip()
    if not cleaned:
        return []
    encoded = quote(cleaned, safe="")
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={encoded}&quotesCount={limit}&newsCount=0"
    try:
        req = URLRequest(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            payload = resp.read().decode("utf-8")
        data = json.loads(payload)
    except (URLError, HTTPError, TimeoutError, ValueError, KeyError):
        return []
    matches: List[Dict[str, str]] = []
    for item in data.get("quotes", [])[:limit]:
        symbol = item.get("symbol")
        if not symbol:
            continue
        name = item.get("shortname") or item.get("longname") or symbol
        quote_type = (item.get("quoteType") or "").lower()
        kind = INSTRUMENT_KIND_CRYPTO if "crypto" in quote_type else INSTRUMENT_KIND_STOCK
        matches.append({"symbol": symbol, "name": name, "kind": kind})
    return matches


def lookup_symbol_profile(symbol: str) -> Optional[Dict[str, str]]:
    normalized = _normalize_symbol(symbol)
    if not normalized:
        return None
    encoded = quote(normalized, safe="")
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={encoded}"
    try:
        req = URLRequest(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            payload = resp.read().decode("utf-8")
        data = json.loads(payload)
        result = data.get("quoteResponse", {}).get("result", [])
        if not result:
            return None
        entry = result[0]
    except (URLError, HTTPError, TimeoutError, ValueError, KeyError, IndexError):
        return None
    name = entry.get("longName") or entry.get("shortName") or normalized
    quote_type = (entry.get("quoteType") or "").lower()
    kind = INSTRUMENT_KIND_CRYPTO if "crypto" in quote_type else INSTRUMENT_KIND_STOCK
    return {"symbol": normalized, "name": name, "kind": kind}


def delete_market_instrument(instrument_id: int) -> None:
    with Session(engine) as session:
        instrument = session.get(MarketInstrument, instrument_id)
        if not instrument:
            return
        if _normalize_symbol(instrument.symbol) == _normalize_symbol(DEFAULT_MARKET_SYMBOL):
            return
        session.delete(instrument)
        session.commit()


def instrument_yahoo_symbol(symbol: str) -> str:
    normalized = _normalize_symbol(symbol)
    if normalized == _normalize_symbol(DEFAULT_MARKET_SYMBOL):
        return "^GSPC"
    return normalized


def _price_cache_key(symbol: str) -> str:
    return f"market_price_{_normalize_symbol(symbol)}"


def _price_ts_cache_key(symbol: str) -> str:
    return f"market_price_ts_{_normalize_symbol(symbol)}"


def _price_history_key(symbol: str) -> str:
    return f"market_price_hist_{_normalize_symbol(symbol)}"


PRICE_HISTORY_RANGES: Dict[str, Dict[str, Any]] = {
    "1d": {"label": "24hr", "range": "1d", "interval": "5m", "ttl": 15},
    "1w": {"label": "1wk", "range": "5d", "interval": "30m", "ttl": 60},
    "1m": {"label": "1mo", "range": "1mo", "interval": "1d", "ttl": 180},
    "3m": {"label": "3mo", "range": "3mo", "interval": "1d", "ttl": 180},
    "6m": {"label": "6mo", "range": "6mo", "interval": "1d", "ttl": 180},
    "1y": {"label": "1yr", "range": "1y", "interval": "1wk", "ttl": 240},
    "5y": {"label": "5yr", "range": "5y", "interval": "1mo", "ttl": 1440},
}
DEFAULT_PRICE_RANGE = "1m"
CHART_VIEW_COMPACT = "compact"
CHART_VIEW_DETAIL = "detail"
DEFAULT_CHART_VIEW = CHART_VIEW_COMPACT
CHART_VIEWS = {CHART_VIEW_COMPACT, CHART_VIEW_DETAIL}


def normalize_history_range(value: str) -> str:
    normalized = (value or "").lower()
    return normalized if normalized in PRICE_HISTORY_RANGES else DEFAULT_PRICE_RANGE


def normalize_chart_view(value: str) -> str:
    normalized = (value or "").strip().lower()
    return normalized if normalized in CHART_VIEWS else DEFAULT_CHART_VIEW


def _price_history_range_cache_key(symbol: str, range_code: str) -> str:
    return f"market_price_hist_{_normalize_symbol(symbol)}_{range_code}"


def _cache_set_price(session: Session, symbol: str, cents: int) -> None:
    MetaDAO.set(session, _price_cache_key(symbol), str(int(cents)))
    MetaDAO.set(session, _price_ts_cache_key(symbol), datetime.utcnow().isoformat())


def _cache_get_price(session: Session, symbol: str) -> tuple[int, Optional[datetime]]:
    price_raw = MetaDAO.get(session, _price_cache_key(symbol))
    ts_raw = MetaDAO.get(session, _price_ts_cache_key(symbol))
    try:
        price_c = int(price_raw) if price_raw is not None else 0
    except (TypeError, ValueError):
        price_c = 0
    try:
        last = datetime.fromisoformat(ts_raw) if ts_raw else None
    except Exception:
        last = None
    return price_c, last


def _should_refresh(last: Optional[datetime]) -> bool:
    if not last:
        return True
    return (datetime.utcnow() - last) > timedelta(minutes=5)


def _download_price_history(symbol: str, range_param: str, interval: str) -> List[Dict[str, Any]]:
    yahoo_symbol = instrument_yahoo_symbol(symbol)
    encoded = quote(yahoo_symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?range={range_param}&interval={interval}"
    )
    try:
        req = URLRequest(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            payload = resp.read().decode("utf-8")
        data = json.loads(payload)
        result = data.get("chart", {}).get("result")
        if not result:
            return []
        chart = result[0]
        timestamps = chart.get("timestamp") or []
        quote_info = chart.get("indicators", {}).get("quote", [{}])[0]
        closes = quote_info.get("close") or []
        points: List[Dict[str, Any]] = []
        for ts, close in zip(timestamps, closes):
            if close is None:
                continue
            try:
                price_c = int(round(float(close) * 100))
            except (TypeError, ValueError):
                continue
            dt = datetime.fromtimestamp(ts)
            points.append({"t": dt.isoformat(), "p": price_c})
        return points
    except (URLError, HTTPError, TimeoutError, ValueError, KeyError):
        return []


def fetch_price_history_range(symbol: str, range_code: str) -> List[Dict[str, Any]]:
    normalized_range = normalize_history_range(range_code)
    config = PRICE_HISTORY_RANGES[normalized_range]
    cache_key = _price_history_range_cache_key(symbol, normalized_range)
    ttl_minutes = int(config.get("ttl", 60) or 0)
    with Session(engine) as session:
        cached_raw = MetaDAO.get(session, cache_key)
        if cached_raw:
            try:
                cached_data = json.loads(cached_raw)
                updated_raw = cached_data.get("updated")
                cached_points = cached_data.get("points", [])
            except Exception:
                cached_data = None
            else:
                try:
                    updated_ts = datetime.fromisoformat(updated_raw) if updated_raw else None
                except Exception:
                    updated_ts = None
                if updated_ts and (datetime.utcnow() - updated_ts) <= timedelta(minutes=ttl_minutes):
                    return cached_points
        points = _download_price_history(symbol, config["range"], config["interval"])
        if points:
            MetaDAO.set(
                session,
                cache_key,
                json.dumps({"updated": datetime.utcnow().isoformat(), "points": points}),
            )
            session.commit()
            return points
        if cached_raw:
            try:
                cached_data = json.loads(cached_raw)
                return cached_data.get("points", [])
            except Exception:
                return []
        return []


def _fetch_price_from_yahoo(symbol: str) -> Optional[int]:
    yahoo_symbol = instrument_yahoo_symbol(symbol)
    encoded = quote(yahoo_symbol, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?range=1d&interval=5m"
    try:
        req = URLRequest(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            data = json.load(resp)
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        meta = result[0].get("meta", {})
        last_price = meta.get("regularMarketPrice")
        if last_price is None:
            closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            closes = [x for x in closes if isinstance(x, (int, float))]
            last_price = closes[-1] if closes else None
        if last_price is None:
            return None
        return int(round(float(last_price) * 100))
    except (URLError, HTTPError, TimeoutError, ValueError, KeyError):
        return None


def _append_price_history(session: Session, symbol: str, price_c: int, max_len: int = 2016) -> None:
    try:
        raw = MetaDAO.get(session, _price_history_key(symbol)) or "[]"
        history = json.loads(raw)
    except Exception:
        history = []
    now_utc = datetime.utcnow()
    if history:
        try:
            last_ts = datetime.fromisoformat(history[-1]["t"])
            if (now_utc - last_ts) < timedelta(minutes=5):
                return
        except Exception:
            pass
    history.append({"t": now_utc.isoformat(), "p": int(price_c)})
    if len(history) > max_len:
        history = history[-max_len:]
    MetaDAO.set(session, _price_history_key(symbol), json.dumps(history))


def get_price_history(symbol: str) -> list[dict]:
    with Session(engine) as session:
        try:
            raw = MetaDAO.get(session, _price_history_key(symbol)) or "[]"
            return json.loads(raw)
        except Exception:
            return []


def _simulate_sp500_to_today() -> int:
    from random import Random

    today = now_local().date()
    with Session(engine) as session:
        last_date_raw = MetaDAO.get(session, "sim_sp500_last")
        price_c = int(MetaDAO.get(session, "sim_sp500_price_cents") or "0")
        seed = int(MetaDAO.get(session, "sim_sp500_seed") or "12345")
        if not last_date_raw or price_c <= 0:
            price_c = 40000
            MetaDAO.set(session, "sim_sp500_price_cents", str(price_c))
            MetaDAO.set(session, "sim_sp500_last", today.isoformat())
            MetaDAO.set(session, "sim_sp500_seed", str(seed))
            session.commit()
            return price_c
        last_date = date.fromisoformat(last_date_raw)
        if last_date >= today:
            return price_c
        current = last_date
        price = price_c / 100.0
        while current < today:
            current += timedelta(days=1)
            day_rng = Random(seed + current.toordinal())
            mu, sigma = 0.0003, 0.01
            r = day_rng.gauss(mu, sigma)
            price = max(1.0, price * (1.0 + r))
        price_c = int(round(price * 100))
        MetaDAO.set(session, "sim_sp500_price_cents", str(price_c))
        MetaDAO.set(session, "sim_sp500_last", today.isoformat())
        session.commit()
        return price_c


def real_market_price_cents(symbol: str) -> Optional[int]:
    with Session(engine) as session:
        cached, last = _cache_get_price(session, symbol)
        if cached and not _should_refresh(last):
            return cached
        live = _fetch_price_from_yahoo(symbol)
        if live and live > 0:
            _cache_set_price(session, symbol, live)
            _append_price_history(session, symbol, live)
            session.commit()
            return live
        if cached > 0:
            return cached
        return None


def market_price_cents(symbol: str) -> int:
    live = real_market_price_cents(symbol)
    if isinstance(live, int) and live > 0:
        return live
    if _normalize_symbol(symbol) == _normalize_symbol(DEFAULT_MARKET_SYMBOL):
        return _simulate_sp500_to_today()
    return max(live or 0, 0)


def compute_holdings_metrics(kid_id: str, symbol: str) -> dict:
    normalized_symbol = _normalize_symbol(symbol)
    price_c = market_price_cents(normalized_symbol)
    with Session(engine) as session:
        holding = session.exec(
            select(Investment).where(Investment.kid_id == kid_id, Investment.fund == normalized_symbol)
        ).first()
        txs = session.exec(
            select(InvestmentTx)
            .where(InvestmentTx.kid_id == kid_id, InvestmentTx.fund == normalized_symbol)
            .order_by(InvestmentTx.ts)
        ).all()
    shares = holding.shares if holding else 0.0
    avg_cost_c = 0
    invested_cost_c = 0
    realized_pl_c = 0
    running_shares = 0.0
    running_cost_c = 0
    for tx in txs:
        if tx.tx_type == "buy":
            running_cost_c += int(round(tx.shares * tx.price_cents))
            running_shares += tx.shares
        else:
            if running_shares > 1e-12:
                avg_cost = running_cost_c / running_shares
                sold_cost = int(round(tx.shares * avg_cost))
                proceeds = int(round(tx.shares * tx.price_cents))
                realized_pl_c += proceeds - sold_cost
                running_cost_c -= sold_cost
                running_shares -= tx.shares
            realized_pl_c += tx.realized_pl_cents
    if running_shares > 1e-12:
        avg_cost_c = int(round(running_cost_c / running_shares))
        invested_cost_c = int(round(running_cost_c))
    market_value_c = int(round(shares * price_c))
    unrealized_pl_c = market_value_c - invested_cost_c
    return {
        "shares": shares,
        "avg_cost_c": int(avg_cost_c),
        "invested_cost_c": int(invested_cost_c),
        "realized_pl_c": int(realized_pl_c),
        "market_value_c": int(market_value_c),
        "unrealized_pl_c": int(unrealized_pl_c),
        "price_c": int(price_c),
    }


def sparkline_svg_from_history(hist: Iterable[dict], width: int = 320, height: int = 64, pad: int = 6) -> str:
    prices = [point.get("p") for point in hist if isinstance(point.get("p"), int)]
    if len(prices) < 2:
        return f"<svg width='{width}' height='{height}'></svg>"
    pmin, pmax = min(prices), max(prices)
    rng = max(1, pmax - pmin)
    xs: List[float] = []
    ys: List[float] = []
    total = len(prices)
    for idx, price in enumerate(prices):
        xs.append(pad + (idx * (width - 2 * pad) / (total - 1)))
        ys.append(pad + (height - 2 * pad) * (1 - (price - pmin) / rng))
    path = "M {:.2f} {:.2f} ".format(xs[0], ys[0]) + " ".join(
        f"L {x:.2f} {y:.2f}" for x, y in zip(xs[1:], ys[1:])
    )
    color = "#16a34a" if prices[-1] >= prices[0] else "#dc2626"
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        f"xmlns='http://www.w3.org/2000/svg' role='img' aria-label='7-day price sparkline'>"
        f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/></svg>"
    )


def detailed_history_chart_svg(
    hist: Iterable[dict], *, width: int = 640, height: int = 240
) -> str:
    points: List[Tuple[datetime, int]] = []
    for entry in hist:
        price_raw = entry.get("p")
        timestamp_raw = entry.get("t")
        if price_raw is None or timestamp_raw is None:
            continue
        try:
            price_c = int(price_raw)
        except (TypeError, ValueError):
            continue
        try:
            moment = (
                datetime.fromisoformat(timestamp_raw)
                if isinstance(timestamp_raw, str)
                else datetime.fromtimestamp(int(timestamp_raw))
            )
        except (TypeError, ValueError, OSError):
            continue
        points.append((moment, price_c))
    if len(points) < 2:
        return (
            f"<svg class='chart chart--detail' width='{width}' height='{height}'></svg>"
        )
    points.sort(key=lambda item: item[0])
    pad_left = 60.0
    pad_right = 16.0
    pad_top = 24.0
    pad_bottom = 36.0
    inner_width = width - pad_left - pad_right
    inner_height = height - pad_top - pad_bottom
    if inner_width <= 0 or inner_height <= 0:
        return (
            f"<svg class='chart chart--detail' width='{width}' height='{height}'></svg>"
        )
    start_time = points[0][0]
    offsets = [(point[0] - start_time).total_seconds() for point in points]
    span = offsets[-1]
    if span <= 0:
        x_positions = [
            pad_left + (inner_width * idx / (len(points) - 1))
            for idx in range(len(points))
        ]
    else:
        x_positions = [pad_left + (offset / span) * inner_width for offset in offsets]
    prices = [price for _, price in points]
    min_price = min(prices)
    max_price = max(prices)
    price_range = max(1, max_price - min_price)
    y_positions = [
        pad_top + (inner_height * (1 - (price - min_price) / price_range))
        for price in prices
    ]
    baseline = pad_top + inner_height
    color = "#16a34a" if prices[-1] >= prices[0] else "#dc2626"
    axis_color = "#94a3b8"
    path_parts = [f"M {x_positions[0]:.2f} {y_positions[0]:.2f}"]
    for x, y in zip(x_positions[1:], y_positions[1:]):
        path_parts.append(f"L {x:.2f} {y:.2f}")
    path_data = " ".join(path_parts)
    fill_data = (
        path_data
        + f" L {pad_left + inner_width:.2f} {baseline:.2f}"
        + f" L {pad_left:.2f} {baseline:.2f} Z"
    )
    y_ticks = [min_price, (min_price + max_price) / 2, max_price]
    y_labels = []
    for value in y_ticks:
        y = pad_top + (inner_height * (1 - (float(value) - min_price) / price_range))
        y_labels.append(
            f"<text x='{pad_left - 10:.2f}' y='{y:.2f}' fill='{axis_color}' "
            "font-size='12' text-anchor='end' dominant-baseline='middle'>"
            + usd(int(round(value)))
            + "</text>"
        )
    total_seconds = points[-1][0] - points[0][0]
    span_seconds = max(1.0, total_seconds.total_seconds())
    if span_seconds <= 36_000:  # ~10 hours
        tick_fmt = "%b %d %H:%M"
    elif span_seconds <= 86_400 * 90:
        tick_fmt = "%b %d"
    elif span_seconds <= 86_400 * 540:
        tick_fmt = "%b %Y"
    else:
        tick_fmt = "%Y"
    tick_candidates = [0, len(points) // 3, (2 * len(points)) // 3, len(points) - 1]
    ticks: List[int] = []
    for idx in tick_candidates:
        if 0 <= idx < len(points) and idx not in ticks:
            ticks.append(idx)
    x_labels = []
    for idx in ticks:
        tick_x = x_positions[idx]
        tick_label = points[idx][0].strftime(tick_fmt)
        x_labels.append(
            f"<text x='{tick_x:.2f}' y='{height - pad_bottom + 18:.2f}' fill='{axis_color}' "
            "font-size='12' text-anchor='middle'>"
            + html_escape(tick_label)
            + "</text>"
        )
    return (
        f"<svg class='chart chart--detail' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' "
        f"role='img' aria-label='Price history detailed chart'>"
        f"<rect x='{pad_left:.2f}' y='{pad_top:.2f}' width='{inner_width:.2f}' "
        f"height='{inner_height:.2f}' fill='none' stroke='rgba(148,163,184,0.18)' stroke-width='1'/>"
        f"<path d='{fill_data}' fill='{color}' fill-opacity='0.15' stroke='none'/>"
        f"<path d='{path_data}' fill='none' stroke='{color}' stroke-width='2.5' stroke-linejoin='round'/>"
        f"<line x1='{pad_left:.2f}' y1='{baseline:.2f}' x2='{pad_left + inner_width:.2f}' y2='{baseline:.2f}' "
        f"stroke='{axis_color}' stroke-width='1'/>"
        f"<line x1='{pad_left:.2f}' y1='{pad_top:.2f}' x2='{pad_left:.2f}' y2='{baseline:.2f}' "
        f"stroke='{axis_color}' stroke-width='1'/>"
        + "".join(y_labels)
        + "".join(x_labels)
        + "</svg>"
    )


# Ensure core market instruments exist after migrations
ensure_default_instrument()
ensure_default_learning_content()


# ---------------------------------------------------------------------------
# Gamification helpers
# ---------------------------------------------------------------------------
def _update_gamification(child: Child, award_cents: int) -> None:
    today = now_local().date()
    if child.last_chore_date == today - timedelta(days=1):
        child.streak_days += 1
    elif child.last_chore_date == today:
        pass
    else:
        child.streak_days = 1 if award_cents > 0 else 0
    child.last_chore_date = today
    points = max(1, award_cents // 25)
    child.total_points += points
    child.level = max(1, 1 + child.total_points // 100)
    badges = {badge for badge in (child.badges or "").split(",") if badge}
    if child.streak_days >= 7:
        badges.add("Streak 7")
    if child.streak_days >= 30:
        badges.add("Streak 30")
    if child.total_points >= 250:
        badges.add("Centurion")
    child.badges = ",".join(sorted(badges))


def _badges_html(badges_csv: Optional[str]) -> str:
    badges = [badge for badge in (badges_csv or "").split(",") if badge]
    if not badges:
        return "<span class='muted'>(none)</span>"
    return " ".join(f"<span class='pill'>{badge}</span>" for badge in badges)

@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest_webmanifest() -> JSONResponse:
    return JSONResponse(
        {
            "name": "KidBank",
            "short_name": "KidBank",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#0b1220",
            "theme_color": "#2563eb",
            "scope": "/",
            "description": "Kid-friendly banking for chores, goals, and investing.",
            "icons": [
                {"src": "/pwa-icon.svg", "sizes": "192x192", "type": "image/svg+xml"},
                {"src": "/pwa-icon.svg", "sizes": "512x512", "type": "image/svg+xml"},
            ],
        }
    )


@app.get("/service-worker.js", include_in_schema=False)
def service_worker_js() -> Response:
    return Response(SERVICE_WORKER_JS, media_type="application/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/pwa-icon.svg", include_in_schema=False)
def pwa_icon() -> Response:
    return Response(PWA_ICON_SVG, media_type="image/svg+xml")


@app.get("/offline", response_class=HTMLResponse)
def offline_page(request: Request) -> HTMLResponse:
    inner = """
    <div class='card'>
      <h3>Offline mode</h3>
      <p class='muted'>You're viewing the cached KidBank shell. Actions like payouts or transfers need a connection.</p>
      <p class='muted'>Reconnect to sync progress.</p>
    </div>
    """
    return render_page(request, "KidBank — Offline", inner)


# ---------------------------------------------------------------------------
# Kid-facing routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def landing(request: Request) -> HTMLResponse:
    if kid_authed(request):
        return RedirectResponse("/kid", status_code=302)
    remembered_kid = html_escape(request.cookies.get(REMEMBER_NAME_COOKIE, "") or "")
    kid_prefill_attr = f" value='{remembered_kid}'" if remembered_kid else ""
    inner = """
    <div class='grid'>
      <div class='card'>
        <h3>Kid Sign-In</h3>
        <form method='post' action='/kid/login'>
          <label>kid_id</label><input name='kid_id' placeholder='e.g. alex01'""" + kid_prefill_attr + """ required>
          <label style='margin-top:8px;'>PIN</label><input name='kid_pin' placeholder='your PIN' required>
          <label class='checkbox' style='margin-top:8px; display:flex; align-items:center; gap:6px;'><input type='checkbox' name='remember_me' value='1'> <span>Remember me</span></label>
          <button type='submit' style='margin-top:10px;'>View My Account</button>
        </form>
      </div>
      <div class='card'>
        <h3>Parent / Admin</h3>
        <p class='muted'>Manage kids, balances, prizes, chores, goals and investing.</p>
        <a href='/admin/login'><button>Go to Admin Login</button></a>
      </div>
    </div>
    """
    return render_page(request, "Kid Bank — Sign In", inner)


@app.post("/kid/login", response_class=HTMLResponse)
def kid_login(
    request: Request,
    kid_id: str = Form(...),
    kid_pin: str = Form(...),
    remember_me: Optional[str] = Form(None),
) -> HTMLResponse:
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child or (child.kid_pin or "") != (kid_pin or ""):
            body = "<div class='card'><p style='color:#ff6b6b;'>Invalid kid_id or PIN.</p><p><a href='/'>Back</a></p></div>"
            return render_page(request, "Kid Login", body)
    request.session["kid_authed"] = kid_id
    response = RedirectResponse("/kid", status_code=302)
    remember_flag = str(remember_me or "").strip().lower() in {"1", "true", "on", "yes"}
    if remember_flag:
        expires_at = datetime.utcnow() + REMEMBER_COOKIE_LIFETIME
        token = _encode_remember_token(kid_id, expires_at)
        response.set_cookie(
            REMEMBER_COOKIE_NAME,
            token,
            max_age=REMEMBER_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            path="/",
        )
        response.set_cookie(
            REMEMBER_NAME_COOKIE,
            kid_id,
            max_age=REMEMBER_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            path="/",
        )
    else:
        response.delete_cookie(REMEMBER_COOKIE_NAME, path="/")
        response.delete_cookie(REMEMBER_NAME_COOKIE, path="/")
    return response


@app.get("/kid", response_class=HTMLResponse)
def kid_home(
    request: Request,
    section: str = Query("overview"),
    chore_day: Optional[str] = Query(None),
    activity_search: str = Query(""),
    activity_dir: str = Query("all"),
) -> HTMLResponse:
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    selected_section = (section or "overview").strip().lower()
    selected_section = {"invest": "investing", "investments": "investing"}.get(
        selected_section, selected_section
    )
    moment = now_local()
    today = moment.date()
    requested_day = (chore_day or "").strip()
    try:
        selected_chore_day = date.fromisoformat(requested_day) if requested_day else today
    except ValueError:
        selected_chore_day = today
    global_infos: List[Dict[str, Any]] = []
    kid_global_claims: List[GlobalChoreClaim] = []
    global_chore_lookup: Dict[int, Chore] = {}
    chores_today: List[Tuple[Chore, Optional[ChoreInstance]]] = []
    activity_query = (activity_search or "").strip()
    activity_direction = (activity_dir or "all").strip().lower()
    if activity_direction not in {"all", "credit", "debit", "zero"}:
        activity_direction = "all"
    lessons: List[Lesson] = []
    quiz_by_lesson: Dict[int, Quiz] = {}
    latest_attempt_by_quiz: Dict[int, QuizAttempt] = {}
    goal_interest_bps = 0
    marketplace_open_listings: List[MarketplaceListing] = []
    marketplace_my_listings: List[MarketplaceListing] = []
    marketplace_claimed_by_me: List[MarketplaceListing] = []
    try:
        others: List[Child] = []
        incoming_requests: List[MoneyRequest] = []
        outgoing_requests: List[MoneyRequest] = []
        with Session(engine) as session:
            child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
            if not child:
                request.session.pop("kid_authed", None)
                return RedirectResponse("/", status_code=302)
            chores_today = list_chore_instances_for_kid(kid_id, target_day=today)
            if selected_chore_day == today:
                chores = chores_today
            else:
                chores = list_chore_instances_for_kid(kid_id, target_day=selected_chore_day)
            event_limit = 120 if activity_query else 40
            events = session.exec(
                select(Event)
                .where(Event.child_id == kid_id)
                .order_by(desc(Event.timestamp))
                .limit(event_limit)
            ).all()
            goals = session.exec(
                select(Goal)
                .where(Goal.kid_id == kid_id)
                .order_by(desc(Goal.created_at))
            ).all()
            others = session.exec(
                select(Child).where(Child.kid_id != kid_id).order_by(Child.name)
            ).all()
            incoming_requests = session.exec(
                select(MoneyRequest)
                .where(MoneyRequest.to_kid_id == kid_id)
                .where(MoneyRequest.status == "pending")
                .order_by(desc(MoneyRequest.created_at))
            ).all()
            outgoing_requests = session.exec(
                select(MoneyRequest)
                .where(MoneyRequest.from_kid_id == kid_id)
                .order_by(desc(MoneyRequest.created_at))
                .limit(12)
            ).all()
            ensure_default_learning_content()
            lessons = session.exec(select(Lesson).order_by(Lesson.id)).all()
            quizzes = session.exec(select(Quiz).order_by(Quiz.lesson_id)).all()
            attempts = session.exec(
                select(QuizAttempt)
                .where(QuizAttempt.child_id == kid_id)
                .order_by(desc(QuizAttempt.created_at))
            ).all()
            quiz_by_lesson = {quiz.lesson_id: quiz for quiz in quizzes}
            for attempt in attempts:
                if attempt.quiz_id not in latest_attempt_by_quiz:
                    latest_attempt_by_quiz[attempt.quiz_id] = attempt
            try:
                goal_interest_bps = int(MetaDAO.get(session, "goal_interest_rate_bps") or "0")
            except (TypeError, ValueError):
                goal_interest_bps = 0
            global_chores = session.exec(
                select(Chore)
                .where(Chore.kid_id == GLOBAL_CHORE_KID_ID)
                .where(Chore.active == True)  # noqa: E712
                .order_by(Chore.name)
            ).all()
            global_chore_lookup.update({ch.id: ch for ch in global_chores})
            kid_global_claims = session.exec(
                select(GlobalChoreClaim)
                .where(GlobalChoreClaim.kid_id == kid_id)
                .order_by(desc(GlobalChoreClaim.submitted_at))
                .limit(30)
            ).all()
            marketplace_my_listings = _safe_marketplace_list(
                session,
                select(MarketplaceListing)
                .where(MarketplaceListing.owner_kid_id == kid_id)
                .order_by(desc(MarketplaceListing.created_at)),
            )
            marketplace_claimed_by_me = _safe_marketplace_list(
                session,
                select(MarketplaceListing)
                .where(MarketplaceListing.claimed_by == kid_id)
                .order_by(desc(MarketplaceListing.created_at)),
            )
            marketplace_open_listings = _safe_marketplace_list(
                session,
                select(MarketplaceListing)
                .where(MarketplaceListing.status == MARKETPLACE_STATUS_OPEN)
                .where(MarketplaceListing.owner_kid_id != kid_id)
                .order_by(desc(MarketplaceListing.created_at)),
            )

            for claim in kid_global_claims:
                if claim.chore_id not in global_chore_lookup:
                    chore_ref = session.get(Chore, claim.chore_id)
                    if chore_ref:
                        global_chore_lookup[chore_ref.id] = chore_ref
            for gchore in global_chores:
                if not is_chore_in_window(gchore, today):
                    continue
                period_key = global_chore_period_key(moment, gchore)
                total_claims = count_global_claims(session, gchore.id, period_key, include_pending=True)
                approved_claims = count_global_claims(session, gchore.id, period_key, include_pending=False)
                existing_claim = get_global_claim(session, gchore.id, kid_id, period_key)
                global_infos.append(
                    {
                        "chore": gchore,
                        "period_key": period_key,
                        "total_claims": total_claims,
                        "approved_claims": approved_claims,
                        "existing_claim": existing_claim,
                    }
                )
        kid_lookup: Dict[str, Child] = {child.kid_id: child}
        for other in others:
            kid_lookup[other.kid_id] = other
        filtered_events = filter_events(
            events,
            search=activity_query,
            direction=activity_direction,
        )
        event_rows = "".join(
            f"<tr><td data-label='When'>{event.timestamp.strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td data-label='Δ Amount' class='right'>{'+' if event.change_cents>=0 else ''}{usd(event.change_cents)}</td>"
            f"<td data-label='Reason'>{html_escape(event.reason)}</td></tr>"
            for event in filtered_events
        ) or "<tr><td colspan='3' class='muted'>No activity matched your filters.</td></tr>"
        filter_summary_html = ""
        if filtered_events:
            if activity_query or activity_direction != "all":
                filter_summary_html = (
                    f"<div class='muted' style='margin-bottom:8px;'>Showing {len(filtered_events)} of {len(events)} recent entries.</div>"
                )
            else:
                filter_summary_html = (
                    f"<div class='muted' style='margin-bottom:8px;'>Showing the latest {len(filtered_events)} entries.</div>"
                )
        else:
            filter_summary_html = (
                f"<div class='muted' style='margin-bottom:8px;'>No matches found across {len(events)} recent entries.</div>"
            )
        activity_search_value = html_escape(activity_query)
        activity_dir_options = []
        for value, label in [
            ("all", "All activity"),
            ("credit", "Credits"),
            ("debit", "Debits"),
            ("zero", "Zero change"),
        ]:
            selected_attr = " selected" if activity_direction == value else ""
            activity_dir_options.append(f"<option value='{value}'{selected_attr}>{label}</option>")
        activity_dir_select = "".join(activity_dir_options)
        reset_filters_html = (
            "<a href='/kid?section=activity' class='button-link secondary' style='margin-top:6px;'>Reset</a>"
            if activity_query or activity_direction != "all"
            else ""
        )
        is_selected_today = selected_chore_day == today
        day_label = "Today" if is_selected_today else selected_chore_day.strftime("%A")
        day_full = selected_chore_day.strftime("%B %d, %Y")
        status_lookup: Dict[str, Tuple[str, str]] = {
            "available": ("available", "Not started"),
            "pending": ("pending", "Awaiting review"),
            CHORE_STATUS_PENDING_MARKETPLACE: ("pending", "Awaiting review"),
            "paid": ("paid", "Completed"),
        }
        chore_tiles: List[str] = []
        for chore, inst in chores:
            status = (inst.status if inst else "available") or "available"
            status_class, status_text = status_lookup.get(status, ("available", status.title()))
            if is_selected_today and status == "available":
                action_html = (
                    f"<form class='inline' method='post' action='/kid/checkoff'>"
                    f"<input type='hidden' name='chore_id' value='{chore.id}'>"
                    "<button type='submit'>Mark complete</button></form>"
                )
            elif is_selected_today and status in {"pending", CHORE_STATUS_PENDING_MARKETPLACE}:
                action_html = "<span class='pill status-pending'>Awaiting review</span>"
            elif is_selected_today and status == "paid":
                action_html = "<span class='pill status-paid'>Completed</span>"
            else:
                action_html = f"<span class='pill status-{status_class}'>{status_text}</span>"
            schedule_bits: List[str] = []
            if chore.start_date or chore.end_date:
                start_display = html_escape(str(chore.start_date)) if chore.start_date else "…"
                end_display = html_escape(str(chore.end_date)) if chore.end_date else "…"
                schedule_bits.append(f"Active {start_display} → {end_display}")
            weekday_set = chore_weekdays(chore)
            if weekday_set:
                schedule_bits.append(f"Days: {format_weekdays(weekday_set)}")
            specific_dates = chore_specific_dates(chore)
            if specific_dates:
                schedule_bits.append(
                    "Dates: " + ", ".join(sorted(d.isoformat() for d in specific_dates))
                )
            schedule_line = (
                f"<div class='muted chore-item__schedule'>{' • '.join(schedule_bits)}</div>"
                if schedule_bits
                else ""
            )
            completed_line = ""
            if inst and inst.completed_at:
                completed_line = (
                    f"<div class='muted chore-item__schedule'>Marked "
                    f"{inst.completed_at.strftime('%b %d %H:%M')}</div>"
                )
            chore_tiles.append(
                "<div class='chore-item'>"
                + "<div class='chore-item__info'>"
                + f"<div class='chore-item__title'><b>{html_escape(chore.name)}</b>"
                + f"<span class='pill chore-item__type'>{html_escape(chore.type.title())}</span></div>"
                + f"<div class='muted chore-item__meta'>Reward {usd(chore.award_cents)}</div>"
                + schedule_line
                + completed_line
                + "</div>"
                + f"<div class='chore-item__action'>{action_html}</div>"
                + "</div>"
            )
        chore_list_html = (
            "".join(chore_tiles) or "<div class='muted chore-empty'>No chores scheduled for this day.</div>"
        )
        cal = calendar.Calendar(firstweekday=0)
        weekday_labels = list(calendar.day_abbr)
        header_cells = "".join(f"<th>{label}</th>" for label in weekday_labels)
        month_weeks = cal.monthdatescalendar(selected_chore_day.year, selected_chore_day.month)
        calendar_rows = []
        for week in month_weeks:
            cells = []
            for day in week:
                cell_classes = ["calendar-cell"]
                if day.month != selected_chore_day.month:
                    cell_classes.append("calendar-cell--faded")
                if day == today:
                    cell_classes.append("calendar-cell--today")
                if day == selected_chore_day:
                    cell_classes.append("calendar-cell--selected")
                day_url = f"/kid?section=chores&chore_day={day.isoformat()}"
                cells.append(
                    f"<td class='{' '.join(cell_classes)}'><a href='{day_url}'>{day.day}</a></td>"
                )
            calendar_rows.append(f"<tr>{''.join(cells)}</tr>")
        calendar_table = (
            "<table class='calendar-table'>"
            + f"<thead><tr>{header_cells}</tr></thead>"
            + f"<tbody>{''.join(calendar_rows)}</tbody>"
            + "</table>"
        )
        month_start = selected_chore_day.replace(day=1)
        month_title = month_start.strftime("%B %Y")
        prev_month_start = (month_start - timedelta(days=1)).replace(day=1)
        next_month_start = (month_start + timedelta(days=32)).replace(day=1)
        prev_url = f"/kid?section=chores&chore_day={prev_month_start.isoformat()}"
        next_url = f"/kid?section=chores&chore_day={next_month_start.isoformat()}"
        calendar_nav = (
            "<div class='calendar-nav'>"
            + f"<a class='calendar-nav__btn' href='{prev_url}'>&larr; {prev_month_start.strftime('%b')}</a>"
            + f"<div class='calendar-nav__title'>{month_title}</div>"
            + f"<a class='calendar-nav__btn' href='{next_url}'>{next_month_start.strftime('%b')} &rarr;</a>"
            + "</div>"
        )
        if not is_selected_today:
            calendar_nav += (
                f"<a class='calendar-nav__today' href='/kid?section=chores'>Jump to today</a>"
            )
        chores_intro = (
            "<div class='muted chore-intro'>Mark chores as complete to send them for approval.</div>"
            if is_selected_today
            else "<div class='muted chore-intro'>Past and future days are read-only.</div>"
        )
        chores_content = f"""
          <div class='card chore-dashboard'>
            <div class='chore-header'>
              <div>
                <h3>Chore Overview <a href='#modal-chores-help' class='help-icon' aria-label='Learn how chores work'>?</a></h3>
                <div class='muted'>Click any calendar date to review that day's chores.</div>
              </div>
              <div class='chore-header__date'>
                <span class='chore-header__label'>{day_label}</span>
                <div class='chore-header__value'>{day_full}</div>
              </div>
            </div>
            <div class='chore-columns'>
              <div class='chore-column chore-column--tasks'>
                {chores_intro}
                {chore_list_html}
              </div>
              <div class='chore-column chore-column--calendar'>
                <div class='chore-column__headline'>Month view</div>
                {calendar_nav}
                {calendar_table}
              </div>
            </div>
          </div>
        """
        global_sections = ""
        for info in global_infos:
            chore = info["chore"]
            period_key = info["period_key"]
            period_display = html_escape(period_key)
            total_claims = info["total_claims"]
            approved_count = info["approved_claims"]
            existing_claim = info.get("existing_claim")
            pending_count = max(0, total_claims - approved_count)
            spots_left = max(0, chore.max_claimants - total_claims)
            name_html = html_escape(chore.name)
            notes_line = (
                f"<div class='muted' style='margin-top:4px;'>{html_escape(chore.notes or '')}</div>"
                if chore.notes
                else ""
            )
            schedule_bits: List[str] = []
            if chore.start_date or chore.end_date:
                schedule_bits.append(
                    f"Active: {chore.start_date or '…'} → {chore.end_date or '…'}"
                )
            weekday_set = chore_weekdays(chore)
            if weekday_set:
                schedule_bits.append(f"Weekdays: {format_weekdays(weekday_set)}")
            specific_dates = chore_specific_dates(chore)
            if specific_dates:
                schedule_bits.append(
                    "Dates: "
                    + ", ".join(sorted(d.isoformat() for d in specific_dates))
                )
            schedule_line = (
                f"<div class='muted' style='margin-top:4px;'>{' • '.join(schedule_bits)}</div>"
                if schedule_bits
                else ""
            )
            status_tags: List[str] = []
            if existing_claim:
                if existing_claim.status == GLOBAL_CHORE_STATUS_PENDING:
                    status_tags.append("<span class='pill'>Pending review</span>")
                elif existing_claim.status == GLOBAL_CHORE_STATUS_APPROVED:
                    status_tags.append(
                        f"<span class='pill' style='background:#15803d;'>Approved {usd(existing_claim.award_cents)}</span>"
                    )
                elif existing_claim.status == GLOBAL_CHORE_STATUS_REJECTED:
                    status_tags.append(
                        "<span class='pill' style='background:#b91c1c;'>Not selected</span>"
                    )
            elif spots_left <= 0:
                status_tags.append("<span class='pill'>All spots claimed</span>")
            status_html = "".join(status_tags)
            can_apply = spots_left > 0 and (
                not existing_claim or existing_claim.status == GLOBAL_CHORE_STATUS_REJECTED
            )
            apply_html = (
                f"<form method='post' action='/kid/global_chore/apply' style='margin-top:8px;'>"
                f"<input type='hidden' name='chore_id' value='{chore.id}'>"
                "<button type='submit'>Apply for this chore</button>"
                "</form>"
            ) if can_apply else ""
            if (
                existing_claim
                and existing_claim.status == GLOBAL_CHORE_STATUS_REJECTED
                and spots_left > 0
            ):
                status_html += "<div class='muted' style='margin-top:4px;'>You can try again while spots remain.</div>"
            stats_line = (
                f"<div class='muted' style='margin-top:4px;'>Approved: {approved_count} • Pending: {pending_count} • Spots left: {spots_left}</div>"
            )
            global_sections += (
                "<div style='margin-top:12px; padding:12px; border:1px solid #1f2937; border-radius:10px;'>"
                + f"<div style='font-weight:600;'>{name_html}</div>"
                + f"<div class='muted' style='margin-top:2px;'>Reward: {usd(chore.award_cents)} shared by up to {chore.max_claimants} kid{'s' if chore.max_claimants != 1 else ''} ({period_display})</div>"
                + notes_line
                + schedule_line
                + stats_line
                + (f"<div style='margin-top:6px;'>{status_html}</div>" if status_html else "")
                + apply_html
                + "</div>"
            )
        if not global_sections:
            global_sections = "<div class='muted' style='margin-top:10px;'>No Free-for-all chores are available right now. Check back soon!</div>"
        history_items = []
        for claim in kid_global_claims[:8]:
            chore_ref = global_chore_lookup.get(claim.chore_id)
            name = html_escape(chore_ref.name) if chore_ref else f"Chore #{claim.chore_id}"
            when = (claim.approved_at or claim.submitted_at).strftime("%Y-%m-%d %H:%M")
            status_text = claim.status.title()
            award_text = f" • {usd(claim.award_cents)}" if claim.status == GLOBAL_CHORE_STATUS_APPROVED else ""
            history_items.append(
                f"<li><b>{name}</b> — {status_text}{award_text} <span class='muted'>({html_escape(claim.period_key)}, {when})</span></li>"
            )
        history_html = (
            "<div style='margin-top:12px;'>"
            "<div style='font-weight:600;'>Recent submissions</div>"
            f"<ul style='margin:6px 0 0 18px; padding:0; list-style:disc;'>{''.join(history_items)}</ul>"
            "</div>"
        ) if history_items else ""
        global_card = f"""
          <div class='card'>
            <h3>Free-for-all</h3>
            <div class='muted'>Optional chores open to everyone for extra rewards.</div>
            {global_sections}
            {history_html}
          </div>
        """
        if not global_infos and not kid_global_claims:
            global_card = f"""
          <div class='card'>
            <h3>Free-for-all</h3>
            <div class='muted'>Optional chores open to everyone for extra rewards.</div>
            <div class='muted' style='margin-top:10px;'>No global chores have been posted yet.</div>
          </div>
        """
        goal_rows = "".join(
            f"<tr><td data-label='Goal'><b>{html_escape(goal.name)}</b>"
            + (" <span class='pill' title='Goal reached'>Reached</span>" if goal.saved_cents >= goal.target_cents else "")
            + "</td>"
            f"<td data-label='Saved' class='right'>{usd(goal.saved_cents)} / {usd(goal.target_cents)}"
            f"<div class='muted'>{format_percent(percent_complete(goal.saved_cents, goal.target_cents))} complete</div></td>"
            "<td data-label='Actions' class='right'>"
            f"<form class='inline' method='post' action='/kid/goal_deposit'>"
            f"<input type='hidden' name='goal_id' value='{goal.id}'>"
            "<input name='amount' type='text' data-money placeholder='deposit $' style='max-width:150px'>"
            "<button type='submit'>Save to Goal</button></form> "
            "<form class='inline' method='post' action='/kid/goal_delete' "
            "onsubmit=\"return confirm('Delete goal and refund to balance?');\" style='margin-left:6px;'>"
            f"<input type='hidden' name='goal_id' value='{goal.id}'>"
            "<button type='submit' class='danger'>Delete</button></form></td></tr>"
            for goal in goals
        ) or "<tr><td>(no goals)</td></tr>"
        goal_projection_html = ""
        if goals:
            goal_payload = [
                {
                    "id": goal.id,
                    "name": goal.name,
                    "saved_cents": goal.saved_cents,
                    "target_cents": goal.target_cents,
                }
                for goal in goals
            ]
            try:
                max_weekly = max(
                    5.0,
                    min(
                        100.0,
                        max((goal.target_cents - goal.saved_cents) / 100 for goal in goals) / 4 or 5.0,
                    ),
                )
            except ValueError:
                max_weekly = 50.0
            interest_note = ""
            if goal_interest_bps:
                interest_note = f"<div class='muted' style='margin-top:4px;'>Includes interest at {goal_interest_bps / 100:.2f}% APR.</div>"
            goal_options = "".join(
                f"<option value='{goal.id}'>{html_escape(goal.name)}</option>" for goal in goals
            )
            goal_projection_html = textwrap.dedent(
                """
                  <div class='goal-projection'>
                    <h4>Goal projection <a href='#modal-goal-tips' class='help-icon' aria-label='How projections work'>?</a></h4>
                    <div class='stacked-form'>
                      <label for='goal-projection-select'>Choose a goal</label>
                      <select id='goal-projection-select'>
                        __GOAL_OPTIONS__
                      </select>
                      <label for='goal-projection-weekly'>Weekly contribution: <span id='goal-projection-amount'>$0.00</span></label>
                      <input id='goal-projection-weekly' type='range' min='0' max='__MAX_WEEKLY__' step='0.50' value='5'>
                      <div class='goal-projection__metrics'>Estimated completion: <span class='goal-projection__eta' id='goal-projection-eta'>Add a weekly amount to see your ETA.</span></div>
                      __INTEREST_NOTE__
                    </div>
                  </div>
                  <script>
                    (function(){
                      const data = __GOAL_DATA__;
                      const rate = __GOAL_RATE__ / 10000;
                      const select = document.getElementById('goal-projection-select');
                      const slider = document.getElementById('goal-projection-weekly');
                      const amountEl = document.getElementById('goal-projection-amount');
                      const etaEl = document.getElementById('goal-projection-eta');
                      function compute(goal, weekly){
                        let saved = goal.saved_cents / 100;
                        const target = goal.target_cents / 100;
                        if (saved >= target) {
                          return 'Goal already reached!';
                        }
                        if (weekly <= 0) {
                          return 'Add a weekly amount to see your projection.';
                        }
                        let weeks = 0;
                        const maxWeeks = 520;
                        while (saved < target && weeks < maxWeeks){
                          saved += weekly;
                          if (rate > 0){
                            saved += saved * (rate / 52);
                          }
                          weeks += 1;
                        }
                        if (weeks >= maxWeeks){
                          return 'More than 10 years — try adding more each week.';
                        }
                        const etaDate = new Date();
                        etaDate.setDate(etaDate.getDate() + weeks * 7);
                        return weeks + ' week' + (weeks === 1 ? '' : 's') + ' (' + etaDate.toLocaleDateString() + ')';
                      }
                      function update(){
                        const selectedId = parseInt(select.value || data[0].id, 10);
                        const weekly = parseFloat(slider.value || '0');
                        const goal = data.find(item => item.id === selectedId) || data[0];
                        amountEl.textContent = '$' + weekly.toFixed(2);
                        etaEl.textContent = compute(goal, weekly);
                      }
                      select.addEventListener('change', update);
                      slider.addEventListener('input', update);
                      update();
                    })();
                  </script>
                """
            )
            goal_projection_html = (
                goal_projection_html
                .replace("__GOAL_OPTIONS__", goal_options)
                .replace("__MAX_WEEKLY__", f"{max_weekly:.2f}")
                .replace("__INTEREST_NOTE__", interest_note)
                .replace("__GOAL_DATA__", json.dumps(goal_payload))
                .replace("__GOAL_RATE__", str(goal_interest_bps))
            )
        lesson_cards: List[str] = []
        lesson_pass_count = 0
        for lesson in lessons:
            quiz = quiz_by_lesson.get(lesson.id)
            quiz_info: Dict[str, Any] = {}
            if quiz:
                try:
                    quiz_info = json.loads(quiz.payload)
                except json.JSONDecodeError:
                    quiz_info = {}
            attempt = latest_attempt_by_quiz.get(quiz.id if quiz else -1)
            summary_text = html_escape((lesson.summary or lesson.content_md.splitlines()[0]).strip())
            if len(summary_text) > 140:
                summary_text = summary_text[:137] + "…"
            status_bits: List[str] = []
            badge = ""
            if attempt and quiz:
                passing = int(quiz_info.get("passing_score", attempt.max_score))
                passed = attempt.score >= max(passing, 0)
                status_bits.append(f"Last score {attempt.score}/{attempt.max_score}")
                if passed:
                    badge = "<span class='badge-earned pill'>Passed</span>"
                    lesson_pass_count += 1
            reward_line = ""
            if quiz and quiz.reward_cents:
                reward_line = f"<div class='lesson-card__meta'>Reward: {usd(quiz.reward_cents)}</div>"
            action_label = "Review lesson" if attempt else "Start lesson"
            lesson_cards.append(
                "<div class='lesson-card'>"
                + f"<div class='lesson-card__header'><div><b>{html_escape(lesson.title)}</b> {badge}</div>"
                + f"<a class='button-link secondary' href='/kid/lesson/{lesson.id}'>{action_label}</a></div>"
                + f"<div class='lesson-card__summary'>{summary_text}</div>"
                + (f"<div class='lesson-card__meta'>{' • '.join(status_bits)}</div>" if status_bits else "")
                + reward_line
                + "</div>"
            )
        if not lesson_cards:
            lesson_cards.append("<div class='muted'>No lessons published yet.</div>")
        learning_content = f"""
          <div class='card'>
            <h3>Learning Lab <a href='#modal-learning-help' class='help-icon' aria-label='Learn about lessons'>?</a></h3>
            <div class='muted'>Quick lessons and quizzes that build your money skills.</div>
            <div class='lesson-list'>{''.join(lesson_cards)}</div>
          </div>
        """
        investing_snapshot = _kid_investing_snapshot(kid_id)
        investing_card = _safe_investing_card(kid_id, snapshot=investing_snapshot)
        notice_msg, notice_kind = pop_kid_notice(request)
        notice_html = ""
        if notice_msg:
            if notice_kind == "error":
                notice_style = "background:#fee2e2; border-left:4px solid #fca5a5; color:#b91c1c;"
            else:
                notice_style = "background:#dcfce7; border-left:4px solid #86efac; color:#166534;"
            notice_html = (
                f"<div class='card' style='margin-top:12px; {notice_style}'><div>{notice_msg}</div></div>"
            )
        def kid_name(identifier: str) -> str:
            person = kid_lookup.get(identifier)
            return person.name if person else identifier

        request_options = "".join(
            f"<option value='{other.kid_id}'>{html_escape(other.name)} ({other.kid_id})</option>"
            for other in others
        )
        if request_options:
            request_form = (
                "<form method='post' action='/kid/request_money'>"
                f"<label>Ask</label><select name='target_kid' required>{request_options}</select>"
                "<label style='margin-top:6px;'>Amount (dollars)</label><input name='amount' type='text' data-money placeholder='e.g. 5.00' required>"
                "<label style='margin-top:6px;'>Reason</label>"
                "<textarea name='reason' placeholder='What do you need it for?' style='width:100%; min-height:56px;' required></textarea>"
                "<button type='submit' style='margin-top:10px;'>Request Money</button>"
                "</form>"
            )
        else:
            request_form = "<p class='muted'>No other kids to request money from yet.</p>"
        send_options = "".join(
            f"<option value='{other.kid_id}'>{html_escape(other.name)} ({other.kid_id})</option>"
            for other in others
        )
        if send_options:
            send_form = (
                f"<form method='post' action='/kid/send_money'>"
                f"<label>Send to</label><select name='to_kid' required>{send_options}</select>"
                "<label style='margin-top:6px;'>Amount (dollars)</label><input name='amount' type='text' data-money placeholder='e.g. 2.00' required>"
                "<label style='margin-top:6px;'>Reason</label>"
                "<textarea name='reason' placeholder='Why are you sending money?' style='width:100%; min-height:56px;' required></textarea>"
                "<button type='submit' style='margin-top:10px;'>Send Money</button>"
                "</form>"
            )
        else:
            send_form = "<p class='muted'>No other kids to send money to yet.</p>"
        pending_requests_section = ""
        pending_requests = [req for req in outgoing_requests if req.status == "pending"]
        if pending_requests:
            pending_items = "".join(
                "<li><b>"
                + html_escape(kid_name(req.to_kid_id))
                + "</b> — "
                + usd(req.amount_cents)
                + (" • " + html_escape(req.reason) if req.reason else "")
                + "</li>"
                for req in pending_requests
            )
            pending_requests_section = (
                "<div style='margin-top:14px;'>"
                "<div style='font-weight:600;'>Requests you sent</div>"
                f"<ul style='margin:8px 0 0 20px; padding:0; list-style:disc;'>{pending_items}</ul>"
                "</div>"
            )
        notifications_html = ""
        if incoming_requests:
            request_blocks = []
            for money_request in incoming_requests:
                requester_name = html_escape(kid_name(money_request.from_kid_id))
                if money_request.reason:
                    reason_line = (
                        "<div class='muted' style='margin-top:4px;'>Reason: "
                        + html_escape(money_request.reason)
                        + "</div>"
                    )
                else:
                    reason_line = "<div class='muted' style='margin-top:4px;'>No reason provided.</div>"
                sent_at = money_request.created_at.strftime("%Y-%m-%d %H:%M")
                request_blocks.append(
                    "<div style='margin-top:12px; padding:12px; border-radius:10px; background:#fffbeb; border:1px solid #fbbf24;'>"
                    + f"<div><b>{requester_name}</b> asked for {usd(money_request.amount_cents)}</div>"
                    + reason_line
                    + f"<div class='muted' style='margin-top:4px;'>Sent {sent_at}</div>"
                    + "<form method='post' action='/kid/request/respond' style='display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;'>"
                    + f"<input type='hidden' name='request_id' value='{money_request.id}'>"
                    + "<button type='submit' name='decision' value='accept'>Accept</button>"
                    + "<button type='submit' name='decision' value='decline' class='danger'>Decline</button>"
                    + "</form>"
                    + "</div>"
                )
            notifications_html = f"""
        <div class='card' style='border:2px solid #f59e0b; background:#fffbeb;'>
          <h3>Notifications</h3>
          <div class='muted'>These money requests need your response.</div>
          {''.join(request_blocks)}
        </div>
        """
        money_card = f"""
          <div class='card'>
            <h3>Send/Request <a href='#modal-money-help' class='help-icon' aria-label='How sending and requesting works'>?</a></h3>
            <div class='muted'>Share allowance or ask for help from siblings.</div>
            <div style='font-weight:600; margin-top:8px;'>Request Money</div>
            {request_form}
            <div style='font-weight:600; margin-top:14px;'>Send Money</div>
            {send_form}
            {pending_requests_section}
          </div>
        """
        available_chore_count = sum(
            1 for _, inst in chores_today if not inst or inst.status == "available"
        )
        pending_chore_count = sum(
            1
            for _, inst in chores_today
            if inst and inst.status in {"pending", CHORE_STATUS_PENDING_MARKETPLACE}
        )
        open_global_count = sum(
            1 for info in global_infos if info["total_claims"] < info["chore"].max_claimants
        )
        pending_global_count = sum(
            1 for claim in kid_global_claims if claim.status == GLOBAL_CHORE_STATUS_PENDING
        )
        approved_global_count = sum(
            1 for claim in kid_global_claims if claim.status == GLOBAL_CHORE_STATUS_APPROVED
        )
        achieved_goals = sum(1 for goal in goals if goal.saved_cents >= goal.target_cents)
        incoming_count = len(incoming_requests)
        pending_outgoing_count = len(pending_requests)
        snapshot_ok = bool(investing_snapshot.get("ok"))
        holdings_count = investing_snapshot.get("holdings_count", 0) if snapshot_ok else 0
        cd_count = investing_snapshot.get("cd_count", 0) if snapshot_ok else 0
        total_invested_c = investing_snapshot.get("total_c", 0) if snapshot_ok else 0
        total_invested = usd(total_invested_c)
        kid_name_html = html_escape(child.name)
        kid_id_html = html_escape(child.kid_id)
        level_points = child.total_points % 100
        progress_pct = int(round((level_points / 100) * 100)) if child.total_points else 0
        progress_pct = max(0, min(100, progress_pct))
        points_to_next = 100 - level_points if level_points else 100
        hero_html = f"""
          <div class='card hero-card'>
            <div class='hero-card__intro'>
              <div class='hero-card__greeting'>Welcome, {kid_name_html}!</div>
              <div class='hero-card__meta'>Kid ID: <span class='pill hero-card__pill'>{kid_id_html}</span></div>
              <div class='hero-card__meta'>Level {child.level} • {child.streak_days}-day streak</div>
              <div class='hero-card__badges'>Badges: {_badges_html(child.badges)}</div>
              <div class='hero-card__progress'>
                <div class='hero-card__progress-label'>Progress to next level <span class='muted'>({points_to_next} pts to go)</span></div>
                <div class='progress-bar'><div class='progress-bar__fill' style='width:{progress_pct}%;'></div></div>
              </div>
            </div>
            <div class='hero-card__balance'>
              <div class='muted'>Account balance</div>
              <div class='hero-card__amount'>{usd(child.balance_cents)}</div>
              <div class='muted hero-card__allowance'>Weekly allowance {usd(child.allowance_cents)}</div>
            </div>
          </div>
        """
        lessons_total = len(lessons)
        marketplace_available_count = len(marketplace_open_listings)
        my_listing_active_count = sum(
            1
            for listing in marketplace_my_listings
            if listing.status in {MARKETPLACE_STATUS_OPEN, MARKETPLACE_STATUS_CLAIMED}
        )
        my_claimed_active_count = sum(
            1
            for listing in marketplace_claimed_by_me
            if listing.status == MARKETPLACE_STATUS_CLAIMED
        )
        quick_cards: List[str] = [
            (
                "<div class='stat-card'>"
                "<div class='stat-card__label'>Today's chores</div>"
                f"<div class='stat-card__value'>{available_chore_count} ready</div>"
                f"<div class='stat-card__meta'>Pending approval {pending_chore_count}</div>"
                "<a href='/kid?section=chores' class='stat-card__action'>Open chores</a></div>"
            ),
            (
                "<div class='stat-card'>"
                "<div class='stat-card__label'>Free-for-all</div>"
                f"<div class='stat-card__value'>{open_global_count} open</div>"
                f"<div class='stat-card__meta'>Pending submissions {pending_global_count}</div>"
                "<a href='/kid?section=freeforall' class='stat-card__action'>See challenges</a></div>"
            ),
            (
                "<div class='stat-card'>"
                "<div class='stat-card__label'>Marketplace</div>"
                f"<div class='stat-card__value'>{marketplace_available_count} open</div>"
                f"<div class='stat-card__meta'>My listings {my_listing_active_count} • My claims {my_claimed_active_count}</div>"
                "<a href='/kid?section=marketplace' class='stat-card__action'>Visit market</a></div>"
            ),
            (
                "<div class='stat-card'>"
                "<div class='stat-card__label'>Goals</div>"
                f"<div class='stat-card__value'>{len(goals)} active</div>"
                f"<div class='stat-card__meta'>Reached {achieved_goals}</div>"
                "<a href='/kid?section=goals' class='stat-card__action'>Manage goals</a></div>"
            ),
            (
                "<div class='stat-card'>"
                "<div class='stat-card__label'>Learning</div>"
                f"<div class='stat-card__value'>{lesson_pass_count}/{lessons_total or 0} passed</div>"
                f"<div class='stat-card__meta'>Lessons available {lessons_total}</div>"
                "<a href='/kid?section=learning' class='stat-card__action'>Open lab</a></div>"
            ),
            (
                "<div class='stat-card'>"
                "<div class='stat-card__label'>Send/Request</div>"
                f"<div class='stat-card__value'>{incoming_count} waiting</div>"
                f"<div class='stat-card__meta'>Outgoing pending {pending_outgoing_count}</div>"
                "<a href='/kid?section=money' class='stat-card__action'>Open hub</a></div>"
            ),
        ]
        if snapshot_ok:
            quick_cards.append(
                "<div class='stat-card'>"
                "<div class='stat-card__label'>Investing</div>"
                f"<div class='stat-card__value'>{total_invested}</div>"
                f"<div class='stat-card__meta'>Markets {holdings_count} • CDs {cd_count}</div>"
                "<a href='/kid?section=investing' class='stat-card__action'>View details</a></div>"
            )
        else:
            quick_cards.append(
                "<div class='stat-card'>"
                "<div class='stat-card__label'>Investing</div>"
                "<div class='stat-card__value'>—</div>"
                "<div class='stat-card__meta'>Link a market to begin</div>"
                "<a href='/kid?section=investing' class='stat-card__action'>Set up</a></div>"
            )
        overview_quick_html = "<div class='overview-stats-grid'>" + "".join(quick_cards) + "</div>"
        lookback_start = moment - timedelta(days=30)
        recent_events = [
            event for event in events if event.timestamp >= lookback_start and event.change_cents
        ]
        earned_c = sum(event.change_cents for event in recent_events if event.change_cents > 0)
        spent_c = sum(-event.change_cents for event in recent_events if event.change_cents < 0)
        net_cents = earned_c - spent_c
        avg_daily_c = 0
        if recent_events:
            avg_daily_c = int(
                (Decimal(net_cents) / Decimal(30)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            )
        biggest_event = max(recent_events, key=lambda ev: abs(ev.change_cents), default=None)
        if biggest_event:
            biggest_reason = html_escape(biggest_event.reason or "Activity")
            biggest_amount = usd(biggest_event.change_cents)
            biggest_when = biggest_event.timestamp.strftime("%b %d")
            biggest_line = f"Top move: {biggest_amount} for {biggest_reason} on {biggest_when}."
        else:
            biggest_line = "No recent money movement."
        net_class = "text-good" if net_cents >= 0 else "text-bad"
        money_insights_html = f"""
          <div class='card analytics-card'>
            <h3>Money insights</h3>
            <div class='muted'>Last 30 days</div>
            <div class='insight-grid'>
              <div>
                <div class='insight-label'>Earned</div>
                <div class='insight-value text-good'>{usd(earned_c)}</div>
              </div>
              <div>
                <div class='insight-label'>Spent</div>
                <div class='insight-value text-bad'>{usd(spent_c)}</div>
              </div>
              <div>
                <div class='insight-label'>Net change</div>
                <div class='insight-value {net_class}'>{usd(net_cents)}</div>
                <div class='insight-meta'>Avg {usd(avg_daily_c)}/day</div>
              </div>
            </div>
            <div class='muted insight-note'>{biggest_line}</div>
          </div>
        """
        if snapshot_ok:
            total_market_c = investing_snapshot.get("total_market_c", 0)
            market_pct = (
                int(round((total_market_c / total_invested_c) * 100))
                if total_invested_c
                else 0
            )
            market_pct = max(0, min(100, market_pct))
            cd_pct = max(0, 100 - market_pct)
            primary = investing_snapshot.get("primary") or {}
            primary_symbol = html_escape(str(primary.get("symbol", "SP500")))
            primary_value = usd(primary.get("value_c", 0))
            rate_summary = html_escape(investing_snapshot.get("rate_summary", ""))
            ready_count = investing_snapshot.get("ready_count", 0)
            distribution_bar = (
                "<div class='distribution-bar'>"
                + f"<div class='distribution-bar__segment distribution-bar__segment--market' style='width:{market_pct}%;'></div>"
                + f"<div class='distribution-bar__segment distribution-bar__segment--cd' style='width:{cd_pct}%;'></div>"
                + "</div>"
            )
            rate_line = (
                f"<div class='muted insight-note'>CD rates {rate_summary}</div>"
                if rate_summary
                else ""
            )
            investing_overview_html = f"""
          <div class='card analytics-card investing-summary'>
            <h3>Investing progress</h3>
            <div class='muted'>Distribution of your investing balance.</div>
            {distribution_bar}
            <div class='distribution-legend'><span>Markets {market_pct}%</span><span>CDs {cd_pct}%</span></div>
            <div class='insight-grid insight-grid--tight'>
              <div>
                <div class='insight-label'>Total invested</div>
                <div class='insight-value'>{total_invested}</div>
              </div>
              <div>
                <div class='insight-label'>Holdings</div>
                <div class='insight-value'>{holdings_count}</div>
                <div class='insight-meta'>{primary_symbol} {primary_value}</div>
              </div>
              <div>
                <div class='insight-label'>CDs</div>
                <div class='insight-value'>{cd_count}</div>
                <div class='insight-meta'>{ready_count} ready to cash out</div>
              </div>
            </div>
            {rate_line}
            <a href='/kid?section=investing' class='button-link secondary insight-link'>Open investing tools</a>
          </div>
        """
        else:
            investing_overview_html = """
          <div class='card analytics-card investing-summary'>
            <h3>Investing progress</h3>
            <div class='muted'>Link a market to start building your portfolio.</div>
            <a href='/kid?section=investing' class='button-link secondary insight-link'>Set up investing</a>
          </div>
        """
        available_preview = [
            (chore, inst)
            for chore, inst in chores_today
            if not inst or inst.status == "available"
        ][:3]
        highlight_items: List[str] = [
            "<li><b>"
            + html_escape(chore.name)
            + "</b> ready (+"
            + usd(chore.award_cents)
            + ")</li>"
            for chore, _ in available_preview
        ]
        if incoming_count:
            highlight_items.append(
                f"<li><b>{incoming_count}</b> request{'s' if incoming_count != 1 else ''} waiting in Send/Request.</li>"
            )
        if pending_outgoing_count:
            highlight_items.append(
                f"<li>You have <b>{pending_outgoing_count}</b> outgoing request{'s' if pending_outgoing_count != 1 else ''} pending.</li>"
            )
        if pending_global_count:
            highlight_items.append(
                f"<li><b>{pending_global_count}</b> Free-for-all submission{'s' if pending_global_count != 1 else ''} pending review.</li>"
            )
        if approved_global_count:
            highlight_items.append(
                f"<li><b>{approved_global_count}</b> Free-for-all win{'s' if approved_global_count != 1 else ''} awarded.</li>"
            )
        if achieved_goals:
            highlight_items.append(
                f"<li><b>{achieved_goals}</b> goal{'s' if achieved_goals != 1 else ''} reached so far.</li>"
            )
        if events:
            last_event = events[0]
            change_text = ("+" if last_event.change_cents >= 0 else "") + usd(last_event.change_cents)
            highlight_items.append(
                "<li>Latest activity: "
                + last_event.timestamp.strftime("%b %d, %I:%M %p")
                + " • "
                + change_text
                + " for "
                + html_escape(last_event.reason)
                + "</li>"
            )
        if not highlight_items:
            highlight_items.append("<li class='muted'>You're all caught up!</li>")
        highlight_list_html = "".join(highlight_items)
        highlights_card = f"""
          <div class='card'>
            <h3>Highlights</h3>
            <ul style='margin:12px 0 0 18px; padding:0; list-style:disc;'>
              {highlight_list_html}
            </ul>
            <div class='actions' style='margin-top:12px;'>
              <a href='/kid?section=chores' class='button-link secondary'>My chores</a>
              <a href='/kid?section=freeforall' class='button-link secondary'>Free-for-all</a>
              <a href='/kid?section=money' class='button-link secondary'>Send/Request</a>
              <a href='/kid?section=goals' class='button-link secondary'>Goals</a>
              <a href='/kid?section=activity' class='button-link secondary'>Recent activity</a>
            </div>
          </div>
        """
        overview_content = hero_html + overview_quick_html + money_insights_html + investing_overview_html + highlights_card
        goals_content = f"""
          <div class='card'>
            <h3>My Goals <a href='#modal-goal-tips' class='help-icon' aria-label='Goal tips'>?</a></h3>
            <form method='post' action='/kid/goal_create' class='inline'>
              <input name='name' placeholder='e.g. Lego set' required>
              <input name='target' type='text' data-money placeholder='target $' required>
              <button type='submit'>Create Goal</button>
            </form>
            <table style='margin-top:8px;'><tr><th>Goal</th><th>Saved</th><th>Actions</th></tr>{goal_rows}</table>
          </div>
          {goal_projection_html}
        """
        activity_content = f"""
          <div class='card'>
            <h3>Recent Activity</h3>
            <form method='get' action='/kid' class='stacked-form' style='margin-bottom:12px;'>
              <input type='hidden' name='section' value='activity'>
              <label>Search</label>
              <input name='activity_search' placeholder='Search reason, amount, or date' value='{activity_search_value}'>
              <label>Type</label>
              <select name='activity_dir'>{activity_dir_select}</select>
              <button type='submit'>Apply</button>
              {reset_filters_html}
            </form>
            {filter_summary_html}
            <table><tr><th>When</th><th>Δ Amount</th><th>Reason</th></tr>{event_rows}</table>
          </div>
        """
        active_listing_chore_ids = {
            listing.chore_id
            for listing in marketplace_my_listings
            if listing.status
            in {
                MARKETPLACE_STATUS_OPEN,
                MARKETPLACE_STATUS_CLAIMED,
            }

        }
        listing_options: List[str] = []
        for chore_obj, _ in chores_today:
            if not chore_obj.id or chore_obj.id in active_listing_chore_ids:
                continue
            option_label = html_escape(chore_obj.name)
            award_line = usd(chore_obj.award_cents)
            listing_options.append(
                f"<option value='{chore_obj.id}'>{option_label} — award {award_line}</option>"
            )
        listing_select_html = "".join(listing_options)
        if listing_select_html:
            listing_form_html = (
                "<form method='post' action='/kid/marketplace/list' class='stacked-form'>"
                "<label>Chore</label>"
                f"<select name='chore_id' required>{listing_select_html}</select>"
                "<label>Offer ($)</label><input name='offer' type='text' data-money placeholder='0.50' required>"
                "<p class='muted'>Your offer is held until the listing is completed or cancelled.</p>"
                "<button type='submit'>Post listing</button>"
                "</form>"
            )
        else:
            listing_form_html = (
                "<p class='muted' style='margin-top:8px;'>"
                "All of today's chores are already listed or claimed."

                "</p>"
            )
        status_styles = {
            MARKETPLACE_STATUS_OPEN: ("Open", "#dbeafe", "#1d4ed8"),
            MARKETPLACE_STATUS_CLAIMED: ("Claimed", "#fef3c7", "#b45309"),
            MARKETPLACE_STATUS_SUBMITTED: ("Submitted", "#e0e7ff", "#4338ca"),
            MARKETPLACE_STATUS_COMPLETED: ("Completed", "#dcfce7", "#166534"),
            MARKETPLACE_STATUS_CANCELLED: ("Cancelled", "#fee2e2", "#b91c1c"),
            MARKETPLACE_STATUS_REJECTED: ("Rejected", "#fee2e2", "#b91c1c"),

        }
        empty_action_html = "<span class='muted'>—</span>"

        def _status_badge(listing: MarketplaceListing) -> str:
            label, bg, fg = status_styles.get(
                listing.status, (listing.status.title(), "#e2e8f0", "#334155")
            )
            return (
                f"<span class='pill' style='background:{bg}; color:{fg};'>{label}</span>"
            )

        def _format_ts(value: Optional[datetime]) -> str:
            if not value:
                return "—"
            try:
                return value.strftime("%Y-%m-%d %H:%M")
            except Exception:
                return str(value)

        available_rows = []
        for listing in marketplace_open_listings:
            owner = kid_lookup.get(listing.owner_kid_id)
            owner_name = (
                html_escape(owner.name)
                if isinstance(owner, Child)
                else html_escape(str(listing.owner_kid_id))
            )
            potential = usd(listing.offer_cents + listing.chore_award_cents)
            created_text = _format_ts(listing.created_at)
            available_rows.append(
                "<tr>"
                f"<td data-label='Owner'><b>{owner_name}</b><div class='muted'>{html_escape(listing.chore_name)}</div></td>"
                f"<td data-label='Offer' class='right'>{usd(listing.offer_cents)}</td>"
                f"<td data-label='Earn' class='right'>{potential}<div class='muted'>Award {usd(listing.chore_award_cents)}</div></td>"
                f"<td data-label='Posted'>{created_text}</td>"
                "<td data-label='Action' class='right'>"
                "<form method='post' action='/kid/marketplace/claim' class='inline'>"
                f"<input type='hidden' name='listing_id' value='{listing.id}'>"
                "<button type='submit'>Claim</button>"
                "</form>"
                "</td>"
                "</tr>"
            )
        available_table = (
            "".join(available_rows)
            or "<tr><td colspan='5' class='muted'>(No open listings right now.)</td></tr>"
        )

        my_listing_rows = []
        for listing in marketplace_my_listings:
            claimer = (
                kid_lookup.get(listing.claimed_by)
                if listing.claimed_by
                else None
            )
            claimer_name = (
                html_escape(claimer.name)
                if isinstance(claimer, Child)
                else (html_escape(str(listing.claimed_by)) if listing.claimed_by else "")
            )
            status_html = _status_badge(listing)
            if listing.status == MARKETPLACE_STATUS_CLAIMED and claimer_name:
                status_html += f"<div class='muted'>By {claimer_name}</div>"
            elif listing.status == MARKETPLACE_STATUS_SUBMITTED and claimer_name:
                status_html += (
                    f"<div class='muted'>Submitted by {claimer_name}</div>"
                    f"<div class='muted'>{_format_ts(listing.submitted_at)}</div>"
                )

            elif listing.status == MARKETPLACE_STATUS_COMPLETED:
                status_html += f"<div class='muted'>Completed {_format_ts(listing.completed_at)}</div>"
            elif listing.status == MARKETPLACE_STATUS_CANCELLED:
                status_html += f"<div class='muted'>Cancelled {_format_ts(listing.cancelled_at)}</div>"
            elif listing.status == MARKETPLACE_STATUS_REJECTED:
                note = html_escape(listing.payout_note) if listing.payout_note else ""
                status_html += f"<div class='muted'>Rejected {_format_ts(listing.completed_at)}</div>"
                if note:
                    status_html += f"<div class='muted'>{note}</div>"

            actions_html = ""
            if listing.status == MARKETPLACE_STATUS_OPEN:
                actions_html = (
                    "<form method='post' action='/kid/marketplace/cancel' class='inline'>"
                    f"<input type='hidden' name='listing_id' value='{listing.id}'>"
                    "<button type='submit' class='danger'>Cancel</button>"
                    "</form>"
                )
            my_listing_rows.append(
                "<tr>"
                f"<td data-label='Chore'><b>{html_escape(listing.chore_name)}</b><div class='muted'>Award {usd(listing.chore_award_cents)}</div></td>"
                f"<td data-label='Offer' class='right'>{usd(listing.offer_cents)}</td>"
                f"<td data-label='Status'>{status_html}</td>"
                f"<td data-label='Actions' class='right'>{actions_html or empty_action_html}</td>"
                "</tr>"
            )
        my_listings_table = (
            "".join(my_listing_rows)
            or "<tr><td colspan='4' class='muted'>You have not posted any marketplace listings yet.</td></tr>"
        )

        my_claim_rows = []
        for listing in marketplace_claimed_by_me:
            owner = kid_lookup.get(listing.owner_kid_id)
            owner_name = (
                html_escape(owner.name)
                if isinstance(owner, Child)
                else html_escape(str(listing.owner_kid_id))
            )
            total_earn = usd(listing.offer_cents + listing.chore_award_cents)
            status_html = _status_badge(listing)
            if listing.status == MARKETPLACE_STATUS_COMPLETED:
                status_html += f"<div class='muted'>Completed {_format_ts(listing.completed_at)}</div>"
            elif listing.status == MARKETPLACE_STATUS_CANCELLED:
                status_html += "<div class='muted'>Cancelled by owner</div>"
            elif listing.status == MARKETPLACE_STATUS_SUBMITTED:
                status_html += f"<div class='muted'>Waiting on approval {_format_ts(listing.submitted_at)}</div>"
            elif listing.status == MARKETPLACE_STATUS_REJECTED:
                note = html_escape(listing.payout_note) if listing.payout_note else ""
                status_html += "<div class='muted'>Rejected by admin</div>"
                if note:
                    status_html += f"<div class='muted'>{note}</div>"

            actions_html = ""
            if listing.status == MARKETPLACE_STATUS_CLAIMED and listing.claimed_by == kid_id:
                actions_html = (
                    "<form method='post' action='/kid/marketplace/complete' class='inline'>"
                    f"<input type='hidden' name='listing_id' value='{listing.id}'>"
                    "<button type='submit'>Mark completed</button>"
                    "</form>"
                )
            my_claim_rows.append(
                "<tr>"
                f"<td data-label='Chore'><b>{html_escape(listing.chore_name)}</b><div class='muted'>From {owner_name}</div></td>"
                f"<td data-label='You'll earn' class='right'>{total_earn}<div class='muted'>Offer {usd(listing.offer_cents)}</div></td>"
                f"<td data-label='Status'>{status_html}</td>"
                f"<td data-label='Actions' class='right'>{actions_html or empty_action_html}</td>"
                "</tr>"
            )
        my_claims_table = (
            "".join(my_claim_rows)
            or "<tr><td colspan='4' class='muted'>Claim a listing to see it here.</td></tr>"
        )

        marketplace_content = f"""
          <div class='card'>
            <h3>Post a marketplace listing</h3>
            <p class='muted'>Offer one of your chores for another kid to complete. You set the extra cash and they also earn the chore award.</p>
            {listing_form_html}
          </div>
          <div class='card'>
            <h3>Available listings</h3>
            <table><tr><th>Owner</th><th>Offer</th><th>You'll earn</th><th>Posted</th><th>Action</th></tr>{available_table}</table>
          </div>
          <div class='card'>
            <h3>My listings</h3>
            <table><tr><th>Chore</th><th>Offer</th><th>Status</th><th>Actions</th></tr>{my_listings_table}</table>
          </div>
          <div class='card'>
            <h3>My claims</h3>
            <table><tr><th>Chore</th><th>You'll earn</th><th>Status</th><th>Actions</th></tr>{my_claims_table}</table>
          </div>
        """
        money_content = (notifications_html if notifications_html else "") + money_card
        sections: List[Tuple[str, str, str]] = [
            ("overview", "Overview", overview_content),
            ("chores", "My Chores", chores_content),
            ("freeforall", "Free-for-all", global_card),
            ("marketplace", "Marketplace", marketplace_content),
            ("goals", "Goals", goals_content),
            ("learning", "Learning Lab", learning_content),
            ("money", "Send/Request", money_content),
            ("investing", "Investing", investing_card),
            ("activity", "Activity", activity_content),
        ]
        sections_map = {key: {"label": label, "content": content} for key, label, content in sections}
        if selected_section not in sections_map:
            selected_section = "overview"
        sidebar_links = "".join(
            f"<a href='/kid?section={key}' class='{ 'active' if key == selected_section else ''}'>{html_escape(cfg['label'])}</a>"
            for key, cfg in sections_map.items()
        )
        content_html = f"{notice_html}{sections_map[selected_section]['content']}"
        requests_badge = ""
        if incoming_count:
            requests_badge = (
                f"<span class='pill' style='background:#f59e0b; color:#78350f;'>Requests: {incoming_count}</span>"
            )
        pref_controls = preference_controls_html(request)
        topbar = (
            "<div class='topbar'><h3>Kid Kiosk</h3><div style='display:flex; flex-direction:column; gap:6px; align-items:flex-end;'>"
            "<div style='display:flex; gap:8px; align-items:center; flex-wrap:wrap;'>"
            f"<span class='pill'>{kid_name_html} ({kid_id_html})</span>"
            + requests_badge
            + "<form method='post' action='/kid/logout' style='display:inline-block;'><button type='submit' class='pill'>Logout</button></form>"
            + "</div>"
            + pref_controls
            + "</div></div>"
        )
        inner = (
            topbar
            + "<div class='layout'><nav class='sidebar'>"
            + sidebar_links
            + "</nav><div class='content'>"
            + content_html
            + "</div></div>"
        )
        max_quiz_reward_cents = 0
        if quiz_by_lesson:
            max_quiz_reward_cents = max(quiz.reward_cents for quiz in quiz_by_lesson.values())
        if goal_interest_bps:
            goal_interest_tip = (
                f"<p class='muted'>Projections include goal interest at about {goal_interest_bps / 100:.2f}% APR when enabled by parents.</p>"
            )
        else:
            goal_interest_tip = (
                "<p class='muted'>Weekly deposits add up fast. Parents can enable goal interest for extra growth.</p>"
            )
        reward_note = (
            f"Earn up to {usd(max_quiz_reward_cents)} per passing quiz."
            if max_quiz_reward_cents
            else "Quizzes help you practice new skills."
        )
        modals = [
            f"""
        <div id='modal-goal-tips' class='modal-overlay'>
          <div class='modal-card'>
            <div class='modal-head'><h3>Goal tips</h3><a href='#' class='pill'>Close</a></div>
            <p>Savings goals grow when you plan a steady rhythm.</p>
            <ul class='tutorial-list'>
              <li>Pick a weekly amount on the slider to see how long the goal will take.</li>
              <li>Mix in chore rewards, allowance, or money gifts to finish sooner.</li>
              <li>Reached goals can be celebrated or reset for the next dream.</li>
            </ul>
            {goal_interest_tip}
          </div>
        </div>
        """,
            f"""
        <div id='modal-learning-help' class='modal-overlay'>
          <div class='modal-card'>
            <div class='modal-head'><h3>Learning Lab</h3><a href='#' class='pill'>Close</a></div>
            <p>Short lessons explain how money, goals, and interest work.</p>
            <ul class='tutorial-list'>
              <li>Read the lesson, then answer the quick quiz to check your understanding.</li>
              <li>{reward_note}</li>
              <li>Your latest score shows on each card so you know what to revisit.</li>
            </ul>
          </div>
        </div>
        """,
            """
        <div id='modal-chores-help' class='modal-overlay'>
          <div class='modal-card'>
            <div class='modal-head'><h3>Chores primer</h3><a href='#' class='pill'>Close</a></div>
            <p>Daily chores show on today's calendar. Tap a chore to send it for approval.</p>
            <ul class='tutorial-list'>
              <li><b>Available</b> chores are ready to complete.</li>
              <li><b>Pending</b> chores await a parent's review.</li>
              <li><b>Completed</b> chores are paid and count toward streaks.</li>
              <li>Free-for-all chores are extra challenges that multiple kids can try.</li>
            </ul>
          </div>
        </div>
        """,
            """
        <div id='modal-money-help' class='modal-overlay'>
          <div class='modal-card'>
            <div class='modal-head'><h3>Send &amp; request</h3><a href='#' class='pill'>Close</a></div>
            <p>Share allowance or ask for help in a friendly way.</p>
            <ul class='tutorial-list'>
              <li>Requests wait for approval and show up in the notifications card.</li>
              <li>Sending money moves funds instantly between balances.</li>
              <li>Remember to explain <em>why</em> in the note so everyone knows the story.</li>
            </ul>
          </div>
        </div>
        """,
        ]
        inner += "".join(modals)
        return render_page(request, f"{child.name} — Kid", inner)
    except Exception:
        body = """
        <div class='card'>
          <h3>We hit a snag</h3>
          <p class='muted'>The kid dashboard ran into an error. Check server logs.</p>
          <a href='/'><button>Back to Sign In</button></a>
        </div>
        """
        return render_page(request, "Kid — Error", body)


@app.get("/kid/lesson/{lesson_id}", response_class=HTMLResponse)
def kid_lesson_page(request: Request, lesson_id: int) -> HTMLResponse:
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    notice_msg, notice_kind = pop_kid_notice(request)
    notice_html = ""
    if notice_msg:
        notice_style = "background:#dcfce7; border-left:4px solid #86efac; color:#166534;"
        if notice_kind == "error":
            notice_style = "background:#fee2e2; border-left:4px solid #fca5a5; color:#b91c1c;"
        notice_html = (
            f"<div class='card' style='margin-bottom:12px; {notice_style}'><div>{notice_msg}</div></div>"
        )
    ensure_default_learning_content()
    kid_display_name = ""
    kid_identifier = kid_id
    lesson_title = "Lesson"
    lesson_content_md = ""
    has_quiz = False
    quiz_payload_text = ""
    quiz_reward_cents = 0
    latest_attempt_info: Optional[Dict[str, Any]] = None
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        lesson = session.get(Lesson, lesson_id)
        if not child or not lesson:
            body = "<div class='card'><p style='color:#f87171;'>Lesson not found.</p><p><a href='/kid?section=learning'>Back</a></p></div>"
            return render_page(request, "Lesson", body, status_code=404)
        kid_display_name = child.name
        kid_identifier = child.kid_id
        lesson_title = lesson.title
        lesson_content_md = lesson.content_md
        quiz_row = session.exec(select(Quiz).where(Quiz.lesson_id == lesson_id)).first()
        if quiz_row:
            has_quiz = True
            quiz_payload_text = quiz_row.payload or ""
            quiz_reward_cents = int(quiz_row.reward_cents or 0)
            latest_attempt_row = session.exec(
                select(QuizAttempt)
                .where(QuizAttempt.child_id == kid_id)
                .where(QuizAttempt.quiz_id == quiz_row.id)
                .order_by(desc(QuizAttempt.created_at))
            ).first()
            if latest_attempt_row:
                latest_attempt_info = {
                    "score": latest_attempt_row.score,
                    "max_score": latest_attempt_row.max_score,
                    "created_at": latest_attempt_row.created_at,
                }
    quiz_data: Dict[str, Any] = {}
    if has_quiz and quiz_payload_text:
        try:
            parsed = json.loads(quiz_payload_text)
            if isinstance(parsed, dict):
                quiz_data = parsed
        except json.JSONDecodeError:
            quiz_data = {}
    questions = quiz_data.get("questions", []) if isinstance(quiz_data.get("questions"), list) else []
    passing_score = int(quiz_data.get("passing_score", len(questions) if questions else 0)) if has_quiz else 0
    lesson_body_html = simple_markdown_to_html(lesson_content_md)
    reward_line = ""
    if has_quiz and quiz_reward_cents:
        reward_line = f"<div class='muted' style='margin-top:10px;'>Passing earns {usd(quiz_reward_cents)}.</div>"
    attempt_summary = ""
    if latest_attempt_info and latest_attempt_info["max_score"]:
        passed = latest_attempt_info["score"] >= max(passing_score, 0)
        status_text = "Passed" if passed else "Keep practicing"
        when = latest_attempt_info["created_at"].strftime("%Y-%m-%d %H:%M")
        attempt_summary = (
            f"<div class='muted' style='margin-top:10px;'>Last attempt {latest_attempt_info['score']}/{latest_attempt_info['max_score']} — {status_text} ({when}).</div>"
        )
    if has_quiz and not questions:
        quiz_form_html = "<div class='muted' style='margin-top:16px;'>Quiz coming soon.</div>"
    elif has_quiz:
        invalid_question = any(
            not isinstance(question.get("options"), list) or not question.get("options")
            for question in questions
        )
        if invalid_question:
            quiz_form_html = "<div class='muted' style='margin-top:16px;'>Quiz setup is incomplete. Try again soon.</div>"
        else:
            question_blocks: List[str] = []
            for idx, question in enumerate(questions):
                prompt = html_escape(str(question.get("prompt", f"Question {idx + 1}")))
                options = question.get("options", [])
                option_html_parts: List[str] = []
                for opt_idx, option in enumerate(options):
                    label = html_escape(str(option))
                    required_attr = " required" if opt_idx == 0 else ""
                    option_html_parts.append(
                        f"<label><input type='radio' name='q{idx}' value='{opt_idx}'{required_attr}> {label}</label>"
                    )
                question_blocks.append(
                    "<fieldset class='quiz-question'>"
                    + f"<legend>{prompt}</legend>"
                    + "".join(option_html_parts)
                    + "</fieldset>"
                )
            quiz_form_html = (
                f"<form method='post' action='/kid/lesson/{lesson_id}' class='stacked-form' style='margin-top:16px;'>"
                + "".join(question_blocks)
                + "<button type='submit'>Submit answers</button>"
                + "</form>"
            )
    else:
        quiz_form_html = "<div class='muted' style='margin-top:16px;'>This lesson is read-only right now.</div>"
    kid_name_html = html_escape(kid_display_name)
    kid_id_html = html_escape(kid_identifier)
    pref_controls = preference_controls_html(request)
    topbar = (
        "<div class='topbar'><h3>Learning Lab</h3><div style='display:flex; flex-direction:column; gap:6px; align-items:flex-end;'>"
        + "<div style='display:flex; gap:8px; align-items:center; flex-wrap:wrap;'>"
        + f"<span class='pill'>{kid_name_html} ({kid_id_html})</span>"
        + "<a href='/kid?section=learning' class='button-link secondary'>← Back to lessons</a>"
        + "<form method='post' action='/kid/logout' style='display:inline-block;'><button type='submit' class='pill'>Logout</button></form>"
        + "</div>"
        + pref_controls
        + "</div></div>"
    )
    lesson_card = (
        "<div class='card'>"
        + f"<h3>{html_escape(lesson_title)}</h3>"
        + reward_line
        + attempt_summary
        + f"<div style='margin-top:12px;'>{lesson_body_html}</div>"
        + quiz_form_html
        + "</div>"
    )
    inner = topbar + notice_html + lesson_card
    return render_page(request, f"{lesson_title} — Lesson", inner)


@app.post("/kid/lesson/{lesson_id}")
async def kid_lesson_submit(request: Request, lesson_id: int) -> Response:
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    ensure_default_learning_content()
    form = await request.form()
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        lesson = session.get(Lesson, lesson_id)
        if not child or not lesson:
            body = "<div class='card'><p style='color:#f87171;'>Lesson not found.</p><p><a href='/kid?section=learning'>Back</a></p></div>"
            return render_page(request, "Lesson", body, status_code=404)
        quiz = session.exec(select(Quiz).where(Quiz.lesson_id == lesson_id)).first()
        if not quiz:
            set_kid_notice(request, "This lesson doesn't have a quiz yet.", "error")
            return RedirectResponse(f"/kid/lesson/{lesson_id}", status_code=302)
        try:
            quiz_data = json.loads(quiz.payload)
        except json.JSONDecodeError:
            quiz_data = {}
        questions = quiz_data.get("questions", []) if isinstance(quiz_data.get("questions"), list) else []
        if not questions:
            set_kid_notice(request, "Quiz setup is incomplete. Try again later.", "error")
            return RedirectResponse(f"/kid/lesson/{lesson_id}", status_code=302)
        invalid_question = any(
            not isinstance(question.get("options"), list) or not question.get("options")
            for question in questions
        )
        if invalid_question:
            set_kid_notice(request, "Quiz setup is incomplete. Try again later.", "error")
            return RedirectResponse(f"/kid/lesson/{lesson_id}", status_code=302)
        passing_score = int(quiz_data.get("passing_score", len(questions))) if questions else 0
        answers: List[int] = []
        score = 0
        for idx, question in enumerate(questions):
            raw_answer = form.get(f"q{idx}")
            try:
                choice = int(str(raw_answer))
            except (TypeError, ValueError):
                choice = -1
            answers.append(choice)
            correct = question.get("answer")
            if isinstance(correct, int) and choice == correct:
                score += 1
        max_score = len(questions)
        prior_pass = session.exec(
            select(QuizAttempt)
            .where(QuizAttempt.child_id == kid_id)
            .where(QuizAttempt.quiz_id == quiz.id)
            .where(QuizAttempt.score >= max(passing_score, 0))
        ).first()
        attempt = QuizAttempt(
            child_id=kid_id,
            quiz_id=quiz.id,
            score=score,
            max_score=max_score,
            responses=json.dumps({"answers": answers}),
        )
        session.add(attempt)
        passed_now = score >= max(passing_score, 0)
        reward_cents = 0
        if passed_now and not prior_pass and quiz.reward_cents > 0:
            reward_cents = quiz.reward_cents
            child.balance_cents += reward_cents
            child.updated_at = datetime.utcnow()
            _update_gamification(child, reward_cents)
            session.add(child)
            session.add(
                Event(
                    child_id=kid_id,
                    change_cents=reward_cents,
                    reason=f"lesson_reward:{lesson_id}",
                )
            )
        session.commit()
    if passed_now:
        if reward_cents:
            set_kid_notice(
                request,
                f"Great job! You scored {score}/{max_score} and earned {usd(reward_cents)}.",
                "success",
            )
        else:
            set_kid_notice(
                request,
                f"You scored {score}/{max_score}! Keep it up.",
                "success",
            )
    else:
        set_kid_notice(
            request,
            f"You scored {score}/{max_score}. Review the lesson and try again.",
            "info",
        )
    return RedirectResponse(f"/kid/lesson/{lesson_id}", status_code=302)


def _kid_investing_snapshot(kid_id: str) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"ok": False}
    try:
        instruments = list_market_instruments_for_kid(kid_id)
        holdings: List[Dict[str, Any]] = []
        cd_total_c = 0
        cd_count = 0
        ready_count = 0
        cd_rates_bps = {code: DEFAULT_CD_RATE_BPS for code, _, _ in CD_TERM_OPTIONS}
        moment = datetime.utcnow()
        with Session(engine) as session:
            if not instruments:
                instruments = list_market_instruments_for_kid(kid_id, session=session)
            for instrument in instruments:
                metrics = compute_holdings_metrics(kid_id, instrument.symbol)
                holdings.append(
                    {
                        "symbol": instrument.symbol,
                        "name": instrument.name or instrument.symbol,
                        "kind": instrument.kind,
                        "metrics": metrics,
                    }
                )
            certificates = session.exec(
                select(Certificate)
                .where(Certificate.kid_id == kid_id)
                .where(Certificate.matured_at == None)  # noqa: E711
                .order_by(desc(Certificate.opened_at))
            ).all()
            cd_rates_bps = get_all_cd_rate_bps(session)
        certificate_details: List[Dict[str, Any]] = []
        if certificates:
            for certificate in certificates:
                cd_total_c += certificate_value_cents(certificate, at=moment)
                if moment >= certificate_maturity_date(certificate):
                    ready_count += 1
                certificate_details.append(
                    {
                        "id": certificate.id,
                        "principal_cents": certificate.principal_cents,
                        "rate_bps": certificate.rate_bps,
                        "term_label": certificate_term_label(certificate),
                        "value_cents": certificate_value_cents(certificate, at=moment),
                        "matures_at": certificate_maturity_date(certificate),
                        "matured": moment >= certificate_maturity_date(certificate),
                    }
                )
            cd_count = len(certificates)
        total_market_c = sum(item["metrics"]["market_value_c"] for item in holdings)
        total_invested_c = sum(item["metrics"]["invested_cost_c"] for item in holdings)
        total_unrealized_c = sum(item["metrics"]["unrealized_pl_c"] for item in holdings)
        total_realized_c = sum(item["metrics"]["realized_pl_c"] for item in holdings)
        total_c = total_market_c + cd_total_c
        default_symbol = _normalize_symbol(DEFAULT_MARKET_SYMBOL)
        primary_entry = next(
            (item for item in holdings if _normalize_symbol(item["symbol"]) == default_symbol),
            holdings[0] if holdings else None,
        )
        rate_summary = ", ".join(
            f"{label} {cd_rates_bps.get(code, DEFAULT_CD_RATE_BPS) / 100:.2f}%"
            for code, label, _ in CD_TERM_OPTIONS
        )
        snapshot.update(
            {
                "ok": True,
                "total_market_c": total_market_c,
                "total_c": total_c,
                "cd_total_c": cd_total_c,
                "cd_count": cd_count,
                "ready_count": ready_count,
                "holdings_count": len(holdings),
                "primary": primary_entry,
                "rate_summary": rate_summary,
                "holdings": holdings,
                "certificates": certificate_details,
                "total_invested_c": total_invested_c,
                "unrealized_pl_c": total_unrealized_c,
                "realized_pl_c": total_realized_c,
            }
        )
        return snapshot
    except Exception:
        return snapshot


def _safe_investing_card(kid_id: str, snapshot: Optional[Dict[str, Any]] = None) -> str:
    try:
        data = snapshot or _kid_investing_snapshot(kid_id)
        if not data.get("ok"):
            raise RuntimeError("snapshot unavailable")
        total_market_c = data.get("total_market_c", 0)
        total_c = data.get("total_c", 0)
        cd_total_c = data.get("cd_total_c", 0)
        cd_count = data.get("cd_count", 0)
        ready_count = data.get("ready_count", 0)
        holdings_count = data.get("holdings_count", 0)
        rate_summary = data.get("rate_summary", "")
        holdings = data.get("holdings") or []
        certificates = data.get("certificates") or []
        total_invested_c = data.get("total_invested_c", 0)
        total_unrealized_c = data.get("unrealized_pl_c", 0)
        total_realized_c = data.get("realized_pl_c", 0)
        net_pl_c = total_unrealized_c + total_realized_c
        def fmt(value: int) -> str:
            return f"{'+' if value >= 0 else ''}{usd(value)}"
        if cd_count:
            ready_text = f" • {ready_count} ready" if ready_count else ""
            cd_line = f"Certificates: <b>{usd(cd_total_c)}</b> across {cd_count} active{ready_text}"
        else:
            cd_line = "Certificates: <span class='muted'>none yet</span>"
        if holdings_count:
            primary = data.get("primary") or {}
            shares = primary.get("shares", 0.0)
            price_c = primary.get("price_c", 0)
            value_c = primary.get("value_c", 0)
            if price_c > 0:
                primary_line = f"Primary market: <b>{usd(value_c)}</b> ({shares:.4f} @ {usd(price_c)})"
            else:
                primary_line = f"Primary market: <b>{usd(value_c)}</b>"
        else:
            primary_line = "Markets: <span class='muted'>no holdings yet</span>"
        market_line = (
            f"Markets: <b>{usd(total_market_c)}</b> across {holdings_count} instrument{'s' if holdings_count != 1 else ''}"
        )
        holding_items: List[str] = []
        for entry in holdings:
            metrics = entry.get("metrics", {})
            symbol = html_escape(entry.get("symbol", "?"))
            name = html_escape(entry.get("name") or symbol)
            shares = metrics.get("shares", 0.0)
            price_c = metrics.get("price_c", 0)
            value_c = metrics.get("market_value_c", 0)
            invested_c = metrics.get("invested_cost_c", 0)
            unrealized_c = metrics.get("unrealized_pl_c", 0)
            realized_c = metrics.get("realized_pl_c", 0)
            total_pl = unrealized_c + realized_c
            if invested_c:
                pct = (total_pl / invested_c) * 100
                pct_text = f" • Return {pct:.1f}%"
            else:
                pct_text = ""
            holding_items.append(
                "<li>"
                + f"<div style='display:flex; flex-direction:column; gap:2px;'>"
                + f"<div><b>{symbol}</b> <span class='muted'>{name}</span></div>"
                + f"<div class='muted'>Shares {shares:.4f} @ {usd(price_c)}</div>"
                + f"<div>Value {usd(value_c)} • P/L {fmt(total_pl)}{pct_text}</div>"
                + "</div>"
                + "</li>"
            )
        holdings_html = (
            "<ul class='investing-holdings'>" + "".join(holding_items) + "</ul>"
        ) if holding_items else "<div class='muted'>No tracked markets yet.</div>"
        certificate_items: List[str] = []
        for certificate in certificates:
            principal_c = certificate.get("principal_cents", 0)
            value_c = certificate.get("value_cents", 0)
            rate_bps = certificate.get("rate_bps", 0)
            term_label = html_escape(certificate.get("term_label", ""))
            maturity = certificate.get("matures_at")
            matured = certificate.get("matured")
            if matured:
                status = "Matured"
            elif isinstance(maturity, datetime):
                status = f"Matures {maturity:%Y-%m-%d}"
            else:
                status = "Growing"
            certificate_items.append(
                "<li>"
                + f"<div style='display:flex; flex-direction:column; gap:2px;'>"
                + f"<div><b>{usd(principal_c)}</b> principal • {rate_bps / 100:.2f}% ({term_label})</div>"
                + f"<div>Value {usd(value_c)} • {status}</div>"
                + "</div>"
                + "</li>"
            )
        certificates_html = (
            "<ul class='investing-cds'>" + "".join(certificate_items) + "</ul>"
        ) if certificate_items else "<div class='muted'>No certificates yet.</div>"
        return f"""
          <div class='card'>
            <h3>Investing</h3>
            <div class='muted'>Stocks &amp; certificates of deposit</div>
            <div style='margin-top:6px;'>{market_line}</div>
            <div style='margin-top:4px;'>{primary_line}</div>
            <div style='margin-top:4px;'>{cd_line}</div>
            <div class='muted' style='margin-top:4px;'>Total balance: <b>{usd(total_c)}</b> • Net P/L {fmt(net_pl_c)} • CD rates {rate_summary}</div>
            <div class='insight-grid insight-grid--tight' style='margin-top:10px;'>
              <div><div class='insight-label'>Markets</div><div class='insight-value'>{usd(total_market_c)}</div></div>
              <div><div class='insight-label'>Invested</div><div class='insight-value'>{usd(total_invested_c)}</div></div>
              <div><div class='insight-label'>Realized P/L</div><div class='insight-value'>{fmt(total_realized_c)}</div></div>
            </div>
            <h4 style='margin-top:12px;'>Tracked markets</h4>
            {holdings_html}
            <h4 style='margin-top:12px;'>Certificates of deposit</h4>
            {certificates_html}
            <p class='muted' style='margin-top:10px;'>Need more tools? <a href='/kid/invest'>Open the full investing dashboard</a>.</p>
          </div>
        """
    except Exception:
        return """
          <div class='card'>
            <h3>Investing</h3>
            <p class='muted'>Temporarily unavailable.</p>
            <a href='/kid/invest'><button style='margin-top:8px;'>Try Full View</button></a>
          </div>
        """

@app.post("/kid/checkoff")
def kid_checkoff(request: Request, chore_id: int = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    moment = now_local()
    today = moment.date()
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore or chore.kid_id != kid_id or not chore.active:
            set_kid_notice(request, "That chore isn't available right now.", "error")
            return RedirectResponse("/kid?section=chores", status_code=302)
        active_listing = _safe_marketplace_first(
            session,

            select(MarketplaceListing)
            .where(MarketplaceListing.chore_id == chore.id)
            .where(MarketplaceListing.owner_kid_id == kid_id)
            .where(
                MarketplaceListing.status.in_(
                    [
                        MARKETPLACE_STATUS_OPEN,
                        MARKETPLACE_STATUS_CLAIMED,
                        MARKETPLACE_STATUS_SUBMITTED,
                    ]
                )
            ),
        )

        if active_listing:
            set_kid_notice(
                request,
                "That chore is currently listed in the marketplace.",
                "error",
            )
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        if not is_chore_in_window(chore, today):
            set_kid_notice(request, "That chore can't be completed today.", "error")
            return RedirectResponse("/kid?section=chores", status_code=302)
        chore_type = normalize_chore_type(chore.type)
        pk = "SPECIAL" if chore_type == "special" else period_key_for(chore_type, moment)
        query = select(ChoreInstance).where(ChoreInstance.chore_id == chore.id)
        if chore_type != "special":
            query = query.where(ChoreInstance.period_key == pk)
        inst = session.exec(query.where(ChoreInstance.status == "available").order_by(desc(ChoreInstance.id))).first()
        if not inst:
            inst = ChoreInstance(chore_id=chore.id, period_key=pk, status="available")
            session.add(inst)
            session.commit()
            session.refresh(inst)
        if inst.status == "available":
            inst.status = "pending"
            inst.completed_at = datetime.utcnow()
            session.add(inst)
            session.commit()
            set_kid_notice(request, f"Sent '{chore.name}' for approval!", "success")
        elif inst.status in {"pending", CHORE_STATUS_PENDING_MARKETPLACE}:
            set_kid_notice(request, "That chore is already waiting for approval.", "error")
        else:
            set_kid_notice(request, "That chore has already been paid out.", "error")
    return RedirectResponse("/kid?section=chores", status_code=302)


@app.post("/kid/global_chore/apply")
def kid_global_chore_apply(request: Request, chore_id: int = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    moment = now_local()
    today = moment.date()
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if (
            not chore
            or chore.kid_id != GLOBAL_CHORE_KID_ID
            or not chore.active
            or not is_chore_in_window(chore, today)
        ):
            set_kid_notice(request, "That Free-for-all chore is not available right now.", "error")
            return RedirectResponse("/kid?section=freeforall", status_code=302)
        period_key = global_chore_period_key(moment, chore)
        existing_claim = get_global_claim(session, chore.id, kid_id, period_key)
        if existing_claim:
            set_kid_notice(request, "You already submitted this Free-for-all chore for this period.", "error")
            return RedirectResponse("/kid?section=freeforall", status_code=302)
        total_claims = count_global_claims(session, chore.id, period_key, include_pending=True)
        if total_claims >= chore.max_claimants:
            set_kid_notice(request, "All spots are taken for that chore.", "error")
            return RedirectResponse("/kid?section=freeforall", status_code=302)
        claim = GlobalChoreClaim(
            chore_id=chore.id,
            kid_id=kid_id,
            period_key=period_key,
            status=GLOBAL_CHORE_STATUS_PENDING,
        )
        session.add(claim)
        session.commit()
    set_kid_notice(request, "Submitted your Free-for-all claim!", "success")
    return RedirectResponse("/kid?section=freeforall", status_code=302)


@app.post("/kid/request_money")
def kid_request_money(
    request: Request,
    target_kid: str = Form(...),
    amount: str = Form(...),
    reason: str = Form(""),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    target = (target_kid or "").strip()
    amount_c = to_cents_from_dollars_str(amount, 0)
    note = " ".join((reason or "").split())
    if len(note) > 160:
        note = note[:157] + "…"
    if not target:
        set_kid_notice(request, "Choose who to ask for money.", "error")
        return RedirectResponse("/kid?section=money", status_code=302)
    if target == kid_id:
        set_kid_notice(request, "Choose someone else to ask for money.", "error")
        return RedirectResponse("/kid?section=money", status_code=302)
    if amount_c <= 0:
        set_kid_notice(request, "Enter an amount greater than zero to request money.", "error")
        return RedirectResponse("/kid?section=money", status_code=302)
    target_name = target
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            request.session.pop("kid_authed", None)
            return RedirectResponse("/", status_code=302)
        target_child = session.exec(select(Child).where(Child.kid_id == target)).first()
        if not target_child:
            set_kid_notice(request, "Could not find that kid.", "error")
            return RedirectResponse("/kid?section=money", status_code=302)
        target_name = target_child.name
        detail_suffix = f" — {note}" if note else ""
        money_request = MoneyRequest(
            from_kid_id=child.kid_id,
            to_kid_id=target_child.kid_id,
            amount_cents=amount_c,
            reason=note,
        )
        session.add(money_request)
        session.add(
            Event(
                child_id=child.kid_id,
                change_cents=0,
                reason=f"Request sent to {target_child.name}: {usd(amount_c)}{detail_suffix}",
            )
        )
        session.add(
            Event(
                child_id=target_child.kid_id,
                change_cents=0,
                reason=f"Request from {child.name}: {usd(amount_c)}{detail_suffix}",
            )
        )
        now = datetime.utcnow()
        child.updated_at = now
        target_child.updated_at = now
        session.add(child)
        session.add(target_child)
        session.commit()
    set_kid_notice(request, f"Asked {target_name} for {usd(amount_c)}.", "success")
    return RedirectResponse("/kid?section=money", status_code=302)


@app.post("/kid/request/respond")
def kid_request_respond(
    request: Request,
    request_id: int = Form(...),
    decision: str = Form(...),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    decision_value = (decision or "").strip().lower()
    if decision_value not in {"accept", "decline"}:
        set_kid_notice(request, "Choose to accept or decline the request.", "error")
        return RedirectResponse("/kid?section=money", status_code=302)
    with Session(engine) as session:
        money_request = session.get(MoneyRequest, request_id)
        if not money_request or money_request.to_kid_id != kid_id:
            set_kid_notice(request, "That request is no longer waiting on you.", "error")
            return RedirectResponse("/kid?section=money", status_code=302)
        if money_request.status != "pending":
            set_kid_notice(request, "That request has already been handled.", "error")
            return RedirectResponse("/kid?section=money", status_code=302)
        responder = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        requester = session.exec(select(Child).where(Child.kid_id == money_request.from_kid_id)).first()
        if not responder or not requester:
            set_kid_notice(request, "Could not process that request right now.", "error")
            return RedirectResponse("/kid?section=money", status_code=302)
        detail_suffix = f" — {money_request.reason}" if money_request.reason else ""
        amount_text = usd(money_request.amount_cents)
        now = datetime.utcnow()
        if decision_value == "decline":
            money_request.status = "declined"
            money_request.resolved_at = now
            responder.updated_at = now
            requester.updated_at = now
            session.add(money_request)
            session.add(responder)
            session.add(requester)
            session.add(
                Event(
                    child_id=responder.kid_id,
                    change_cents=0,
                    reason=f"Declined request from {requester.name}: {amount_text}{detail_suffix}",
                )
            )
            session.add(
                Event(
                    child_id=requester.kid_id,
                    change_cents=0,
                    reason=f"Request declined by {responder.name}: {amount_text}{detail_suffix}",
                )
            )
            session.commit()
            set_kid_notice(request, f"Declined {requester.name}'s request.", "success")
            return RedirectResponse("/kid?section=money", status_code=302)
        if responder.balance_cents < money_request.amount_cents:
            set_kid_notice(request, "Not enough funds to accept this request right now.", "error")
            return RedirectResponse("/kid?section=money", status_code=302)
        responder.balance_cents -= money_request.amount_cents
        requester.balance_cents += money_request.amount_cents
        responder.updated_at = now
        requester.updated_at = now
        money_request.status = "accepted"
        money_request.resolved_at = now
        session.add(responder)
        session.add(requester)
        session.add(money_request)
        session.add(
            Event(
                child_id=responder.kid_id,
                change_cents=-money_request.amount_cents,
                reason=f"Accepted request from {requester.name}: {amount_text}{detail_suffix}",
            )
        )
        session.add(
            Event(
                child_id=requester.kid_id,
                change_cents=money_request.amount_cents,
                reason=f"Request accepted by {responder.name}: {amount_text}{detail_suffix}",
            )
        )
        session.commit()
        set_kid_notice(request, f"Sent {amount_text} to {requester.name}.", "success")
    return RedirectResponse("/kid?section=money", status_code=302)


@app.post("/kid/send_money")
def kid_send_money(
    request: Request,
    to_kid: str = Form(...),
    amount: str = Form(...),
    reason: str = Form(""),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    from_kid = kid_authed(request)
    assert from_kid
    target = (to_kid or "").strip()
    amount_c = to_cents_from_dollars_str(amount, 0)
    note = " ".join((reason or "").split())
    if len(note) > 160:
        note = note[:157] + "…"
    if not target:
        set_kid_notice(request, "Choose who to send money to.", "error")
        return RedirectResponse("/kid?section=money", status_code=302)
    if amount_c <= 0:
        set_kid_notice(request, "Enter an amount greater than zero to send money.", "error")
        return RedirectResponse("/kid?section=money", status_code=302)
    recipient_name = ""
    with Session(engine) as session:
        sender = session.exec(select(Child).where(Child.kid_id == from_kid)).first()
        if not sender:
            request.session.pop("kid_authed", None)
            return RedirectResponse("/", status_code=302)
        recipient = session.exec(select(Child).where(Child.kid_id == target)).first()
        if not recipient:
            set_kid_notice(request, "Could not find that kid.", "error")
            return RedirectResponse("/kid?section=money", status_code=302)
        if recipient.kid_id == sender.kid_id:
            set_kid_notice(request, "Choose someone else to send money to.", "error")
            return RedirectResponse("/kid?section=money", status_code=302)
        if sender.balance_cents < amount_c:
            set_kid_notice(request, "Not enough funds to send that amount.", "error")
            return RedirectResponse("/kid?section=money", status_code=302)
        description = note or "Shared money"
        sender.balance_cents -= amount_c
        sender.updated_at = datetime.utcnow()
        recipient.balance_cents += amount_c
        recipient.updated_at = datetime.utcnow()
        session.add(
            Event(
                child_id=sender.kid_id,
                change_cents=-amount_c,
                reason=f"Sent to {recipient.name}: {description}",
            )
        )
        session.add(
            Event(
                child_id=recipient.kid_id,
                change_cents=amount_c,
                reason=f"Received from {sender.name}: {description}",
            )
        )
        session.add(sender)
        session.add(recipient)
        session.commit()
        recipient_name = recipient.name
    set_kid_notice(request, f"Sent {usd(amount_c)} to {recipient_name}!", "success")
    return RedirectResponse("/kid?section=money", status_code=302)


@app.post("/kid/marketplace/list")
def kid_marketplace_list(
    request: Request,
    chore_id: int = Form(...),
    offer: str = Form(...),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    offer_cents = to_cents_from_dollars_str(offer, 0)
    if offer_cents <= 0:
        set_kid_notice(request, "Enter an offer greater than zero.", "error")
        return RedirectResponse("/kid?section=marketplace", status_code=302)
    moment = now_local()
    today = moment.date()
    listed_chore_name = ""
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        chore = session.get(Chore, chore_id)
        if not child or not chore or chore.kid_id != kid_id:
            set_kid_notice(request, "Could not list that chore right now.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        if not chore.active or not is_chore_in_window(chore, today):
            set_kid_notice(request, "That chore isn't available to list today.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        listed_chore_name = chore.name
        existing_listing = _safe_marketplace_first(
            session,

            select(MarketplaceListing)
            .where(MarketplaceListing.chore_id == chore.id)
            .where(MarketplaceListing.owner_kid_id == kid_id)
            .where(
                MarketplaceListing.status.in_(
                    [MARKETPLACE_STATUS_OPEN, MARKETPLACE_STATUS_CLAIMED]
                )
            ),
        )

        if existing_listing:
            set_kid_notice(request, "That chore already has an active listing.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        if child.balance_cents < offer_cents:
            set_kid_notice(request, "Not enough balance to cover that offer.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        child.balance_cents -= offer_cents
        child.updated_at = datetime.utcnow()
        session.add(
            Event(
                child_id=child.kid_id,
                change_cents=-offer_cents,
                reason=f"Marketplace offer held: {chore.name}",
            )
        )
        listing = MarketplaceListing(
            owner_kid_id=kid_id,
            chore_id=chore.id,
            chore_name=chore.name,
            chore_award_cents=chore.award_cents,
            offer_cents=offer_cents,
        )
        session.add(child)
        session.add(listing)
        session.commit()
    set_kid_notice(request, f"Listed '{listed_chore_name}' in the marketplace!", "success")
    return RedirectResponse("/kid?section=marketplace", status_code=302)


@app.post("/kid/marketplace/cancel")
def kid_marketplace_cancel(request: Request, listing_id: int = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    with Session(engine) as session:
        listing = session.get(MarketplaceListing, listing_id)
        if not listing or listing.owner_kid_id != kid_id:
            set_kid_notice(request, "That listing is no longer available to cancel.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        if listing.status != MARKETPLACE_STATUS_OPEN:
            set_kid_notice(request, "Only open listings can be cancelled.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        owner = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not owner:
            set_kid_notice(request, "Could not process that cancellation.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        listing.status = MARKETPLACE_STATUS_CANCELLED
        listing.cancelled_at = datetime.utcnow()
        owner.balance_cents += listing.offer_cents
        owner.updated_at = datetime.utcnow()
        session.add(
            Event(
                child_id=owner.kid_id,
                change_cents=listing.offer_cents,
                reason=f"Marketplace offer returned: {listing.chore_name}",
            )
        )
        session.add(owner)
        session.add(listing)
        session.commit()
    set_kid_notice(request, "Listing cancelled and offer returned to your balance.", "success")
    return RedirectResponse("/kid?section=marketplace", status_code=302)


@app.post("/kid/marketplace/claim")
def kid_marketplace_claim(request: Request, listing_id: int = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    with Session(engine) as session:
        listing = session.get(MarketplaceListing, listing_id)
        if not listing or listing.status != MARKETPLACE_STATUS_OPEN:
            set_kid_notice(request, "That listing is no longer open.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        if listing.owner_kid_id == kid_id:
            set_kid_notice(request, "You can't claim your own listing.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        listing.status = MARKETPLACE_STATUS_CLAIMED
        listing.claimed_by = kid_id
        listing.claimed_at = datetime.utcnow()
        session.add(listing)
        session.commit()
    set_kid_notice(request, "Listing claimed! Finish the chore to collect the payout.", "success")
    return RedirectResponse("/kid?section=marketplace", status_code=302)


@app.post("/kid/marketplace/complete")
def kid_marketplace_complete(request: Request, listing_id: int = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    with Session(engine) as session:
        listing = session.get(MarketplaceListing, listing_id)
        if not listing or listing.status != MARKETPLACE_STATUS_CLAIMED:
            set_kid_notice(request, "That listing is not waiting on you.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        if listing.claimed_by != kid_id:
            set_kid_notice(request, "Another kid claimed that listing.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        worker = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        owner = session.exec(select(Child).where(Child.kid_id == listing.owner_kid_id)).first()
        chore = session.get(Chore, listing.chore_id)
        if not worker or not owner:
            set_kid_notice(request, "Could not process that completion right now.", "error")
            return RedirectResponse("/kid?section=marketplace", status_code=302)
        award_cents = listing.chore_award_cents or (chore.award_cents if chore else 0)
        if award_cents < 0:
            award_cents = 0
        if listing.chore_award_cents != award_cents:
            listing.chore_award_cents = award_cents
        moment = datetime.utcnow()
        listing.status = MARKETPLACE_STATUS_SUBMITTED
        listing.submitted_at = moment
        listing.completed_at = None
        listing.final_payout_cents = None
        listing.payout_note = None
        listing.resolved_by = None
        listing.payout_event_id = None

        session.add(
            Event(
                child_id=owner.kid_id,
                change_cents=0,
                reason=f"Marketplace helper {worker.name} submitted {listing.chore_name}",

            )
        )
        if chore:
            chore_type = normalize_chore_type(chore.type)
            period_key = (
                "SPECIAL"
                if chore_type == "special"
                else period_key_for(chore_type, now_local())
            )
            query = select(ChoreInstance).where(ChoreInstance.chore_id == chore.id)
            if chore_type != "special":
                query = query.where(ChoreInstance.period_key == period_key)
            inst = session.exec(query.order_by(desc(ChoreInstance.id))).first()
            if not inst:
                inst = ChoreInstance(
                    chore_id=chore.id,
                    period_key=period_key,
                    status="available",
                )
            inst.status = CHORE_STATUS_PENDING_MARKETPLACE
            inst.completed_at = moment
            session.add(inst)
        session.add(listing)
        session.commit()
    set_kid_notice(
        request,
        "Marketplace chore submitted! A parent will review the payout soon.",
        "success",
    )

    return RedirectResponse("/kid?section=marketplace", status_code=302)


@app.post("/kid/logout")
def kid_logout(request: Request):
    request.session.pop("kid_authed", None)
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(REMEMBER_COOKIE_NAME, path="/")
    return response


@app.post("/kid/goal_create")
def kid_goal_create(request: Request, name: str = Form(...), target: str = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    target_c = to_cents_from_dollars_str(target, 0)
    goal_name = name.strip()
    with Session(engine) as session:
        session.add(Goal(kid_id=kid_id, name=goal_name, target_cents=target_c))
        session.commit()
    set_kid_notice(request, f"Created goal '{goal_name}'!", "success")
    return RedirectResponse("/kid?section=goals", status_code=302)


@app.post("/kid/goal_deposit")
def kid_goal_deposit(request: Request, goal_id: int = Form(...), amount: str = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        set_kid_notice(request, "Enter an amount greater than zero.", "error")
        return RedirectResponse("/kid?section=goals", status_code=302)
    goal_name = ""
    with Session(engine) as session:
        # When tests monkeypatch goal.saved_cents to None we can trigger an
        # autoflush before we have a chance to normalise the value. Wrapping the
        # initial lookups in no_autoflush lets us safely load the objects before
        # we mutate them.
        with session.no_autoflush:
            goal = session.get(Goal, goal_id)
            child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not goal or not child or goal.kid_id != kid_id:
            set_kid_notice(request, "Could not find that goal.", "error")
            return RedirectResponse("/kid?section=goals", status_code=302)
        if amount_c > child.balance_cents:
            set_kid_notice(request, "Not enough funds to save that amount.", "error")
            return RedirectResponse("/kid?section=goals", status_code=302)
        goal_name = goal.name
        child.balance_cents -= amount_c
        current_saved = goal.saved_cents or 0
        if goal.saved_cents is None:
            history = inspect(goal).attrs.saved_cents.history
            if history.deleted:
                current_saved = history.deleted[0] or 0
        goal_target = goal.target_cents or 0
        goal.saved_cents = current_saved + amount_c
        child.updated_at = datetime.utcnow()
        if goal.saved_cents >= goal_target and goal.achieved_at is None:
            goal.achieved_at = datetime.utcnow()
            session.add(Event(child_id=kid_id, change_cents=0, reason=f"goal_reached:{goal.name}"))
        session.add(child)
        session.add(goal)
        session.add(Event(child_id=kid_id, change_cents=-amount_c, reason=f"goal_deposit:{goal.name}"))
        session.commit()
    set_kid_notice(request, f"Saved {usd(amount_c)} to {goal_name or 'your goal'}.", "success")
    return RedirectResponse("/kid?section=goals", status_code=302)


@app.post("/kid/goal_delete")
def kid_goal_delete(request: Request, goal_id: int = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not goal or not child or goal.kid_id != kid_id:
            set_kid_notice(request, "Could not find that goal.", "error")
            return RedirectResponse("/kid?section=goals", status_code=302)
        goal_name = goal.name
        if goal.saved_cents > 0:
            child.balance_cents += goal.saved_cents
            session.add(Event(child_id=kid_id, change_cents=goal.saved_cents, reason=f"goal_refund_delete:{goal.name}"))
        session.delete(goal)
        child.updated_at = datetime.utcnow()
        session.add(child)
        session.commit()
    set_kid_notice(request, f"Deleted goal '{goal_name}' and returned the savings.", "success")
    return RedirectResponse("/kid?section=goals", status_code=302)


@app.get("/kid/invest", response_class=HTMLResponse)
def kid_invest_home(
    request: Request,
    symbol: Optional[str] = Query(None),
    range_code: str = Query(DEFAULT_PRICE_RANGE, alias="range"),
    lookup: str = Query(""),
    chart_view: str = Query(DEFAULT_CHART_VIEW, alias="chart"),
) -> HTMLResponse:
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    notice_msg, notice_kind = pop_kid_notice(request)
    notice_html = ""
    if notice_msg:
        notice_class = "error" if notice_kind == "error" else "success"
        notice_html = (
            f"<div class='notice {notice_class}'>{html_escape(notice_msg)}</div>"
        )
    try:
        instruments = list_market_instruments_for_kid(kid_id)
        if not instruments:
            raise RuntimeError("No market instruments available.")
        instrument_map = {_normalize_symbol(inst.symbol): inst for inst in instruments}
        requested_symbol = _normalize_symbol(symbol) if symbol else ""
        lookup_query = (lookup or "").strip()
        lookup_results = search_market_symbols(lookup_query) if lookup_query else []
        default_symbol = _normalize_symbol(DEFAULT_MARKET_SYMBOL)
        selected_symbol = requested_symbol or default_symbol
        if selected_symbol not in instrument_map:
            selected_symbol = default_symbol if default_symbol in instrument_map else next(iter(instrument_map.keys()))
        active_instrument = instrument_map[selected_symbol]
        instrument_symbol_raw = active_instrument.symbol
        selected_range = normalize_history_range(range_code)
        chart_mode = normalize_chart_view(chart_view)
        metrics = compute_holdings_metrics(kid_id, selected_symbol)
        history = fetch_price_history_range(selected_symbol, selected_range)
        if chart_mode == CHART_VIEW_DETAIL:
            svg = detailed_history_chart_svg(history, width=640, height=260)
            enlarged_svg = detailed_history_chart_svg(history, width=1024, height=420)
        else:
            svg = sparkline_svg_from_history(history, width=360, height=120)
            enlarged_svg = sparkline_svg_from_history(history, width=960, height=240)
        cd_rates_bps = {code: DEFAULT_CD_RATE_BPS for code, _, _ in CD_TERM_OPTIONS}
        with Session(engine) as session:
            child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
            balance_c = child.balance_cents if child else 0
            certificates = session.exec(
                select(Certificate)
                .where(Certificate.kid_id == kid_id)
                .order_by(desc(Certificate.opened_at))
            ).all()
            cd_rates_bps = get_all_cd_rate_bps(session)
            penalty_days_by_term = get_all_cd_penalty_days(session)

        def fmt(value: int) -> str:
            return f"{'+' if value >= 0 else ''}{usd(value)}"

        moment = _time_provider()
        cd_total_c = 0
        matured_ready = 0
        next_maturity: Optional[datetime] = None
        cert_rows = ""
        for certificate in certificates:
            value_c = certificate_value_cents(certificate, at=moment)
            maturity = certificate_maturity_date(certificate)
            progress_pct = certificate_progress_percent(certificate, at=moment)
            rate_display = certificate.rate_bps / 100
            if certificate.matured_at:
                status = f"Cashed out on {certificate.matured_at.strftime('%Y-%m-%d')}"
                progress_pct = 100.0
            elif moment >= maturity:
                status = "Matured — ready to cash out"
                matured_ready += 1
            else:
                days_left = max(0, (maturity.date() - moment.date()).days)
                status = f"Matures {maturity:%Y-%m-%d} ({days_left} days left)"
            if certificate.matured_at is None:
                cd_total_c += value_c
                if next_maturity is None or maturity < next_maturity:
                    next_maturity = maturity
            button_class_attr = " class='danger'" if certificate.matured_at is None else ""
            cert_rows += (
                f"<tr><td data-label='Principal'>{usd(certificate.principal_cents)}</td>"
                f"<td data-label='Rate'>{rate_display:.2f}%</td>"
                f"<td data-label='Term'>{certificate_term_label(certificate)}</td>"
                f"<td data-label='Value Today'>{usd(value_c)}</td>"
                f"<td data-label='Progress'>{progress_pct:.1f}%</td>"
                f"<td data-label='Status'>{status}</td>"
                f"<td data-label='Actions'>"
                f"<form class='inline' method='post' action='/kid/invest/cd/cashout'>"
                f"<input type='hidden' name='certificate_id' value='{certificate.id}'>"
                f"<input type='hidden' name='symbol' value='{instrument_symbol_raw}'>"
                f"<input type='hidden' name='range' value='{selected_range}'>"
                f"<input type='hidden' name='chart' value='{chart_mode}'>"
                f"<button type='submit'{button_class_attr}>{'Cash Out' if certificate.matured_at is None else 'Remove'}</button>"
                f"</form></td></tr>"
            )
        if not cert_rows:
            cert_rows = "<tr><td colspan='7' class='muted'>No certificates yet.</td></tr>"
        cd_rates_pct = {code: rate / 100 for code, rate in cd_rates_bps.items()}
        rate_summary_text = ", ".join(
            f"{label} {cd_rates_pct.get(code, DEFAULT_CD_RATE_BPS / 100):.2f}%"
            for code, label, _ in CD_TERM_OPTIONS
        )
        summary_bits = [
            f"<div><b>Current rates:</b> {rate_summary_text}</div>",
            f"<div>Total active value: <b>{usd(cd_total_c)}</b></div>",
        ]
        if next_maturity:
            summary_bits.append(f"<div>Next maturity: <b>{next_maturity:%Y-%m-%d}</b></div>")
        if matured_ready:
            summary_bits.append(
                f"<div class='muted'>{matured_ready} certificate{'s' if matured_ready != 1 else ''} ready to cash out.</div>"
            )
        penalty_active = any(days > 0 for days in penalty_days_by_term.values())
        if penalty_active:
            penalty_parts = []
            for code, label, _ in CD_TERM_OPTIONS:
                days = penalty_days_by_term.get(code, 0)
                penalty_parts.append(
                    f"{label}: {days} day{'s' if days != 1 else ''}"
                )
            penalty_summary = (
                "<div>Early withdrawal penalty: <b>"
                + ", ".join(penalty_parts)
                + "</b></div>"
            )
        else:
            penalty_summary = "<div>No penalty for early withdrawals right now.</div>"
        summary_bits.append(penalty_summary)
        cd_summary_html = "".join(summary_bits)
        cash_out_form = ""
        if matured_ready:
            cash_out_form = (
                "<form method='post' action='/kid/invest/cd/mature' style='margin-top:10px;'>"
                f"<input type='hidden' name='symbol' value='{instrument_symbol_raw}'>"
                f"<input type='hidden' name='range' value='{selected_range}'>"
                f"<input type='hidden' name='chart' value='{chart_mode}'>"
                "<button type='submit'>Cash out matured</button>"
                "</form>"
            )

        term_options_html = "".join(
            f"<option value='{code}'{' selected' if code == DEFAULT_CD_TERM_CODE else ''}>{label} — {cd_rates_pct[code]:.2f}% APR</option>"
            for code, label, _ in CD_TERM_OPTIONS
        )

        tabs_html = ""
        if len(instruments) > 1:
            links: List[str] = []
            for inst in instruments:
                normalized = _normalize_symbol(inst.symbol)
                active_style = "background:var(--accent); color:#fff;" if normalized == selected_symbol else ""
                link_url = (
                    f"/kid/invest?symbol={inst.symbol}&range={selected_range}&chart={chart_mode}"
                )
                links.append(
                    f"<a href='{link_url}' class='pill' style='margin-right:6px;{active_style}'>"
                    f"{html_escape(inst.name or inst.symbol)}</a>"
                )
            tabs_html = "<div class='muted' style='margin-bottom:8px;'>Markets: " + "".join(links) + "</div>"

        range_links: List[str] = []
        for code, cfg in PRICE_HISTORY_RANGES.items():
            label = cfg.get("label", code)
            active_style = "background:var(--accent); color:#fff;" if code == selected_range else ""
            link_url = (
                f"/kid/invest?symbol={active_instrument.symbol}&range={code}&chart={chart_mode}"
            )
            range_links.append(
                f"<a href='{link_url}' class='pill' style='margin-right:6px;{active_style}'>{label}</a>"
            )
        range_selector_html = "<div class='muted' style='margin-top:8px;'>Range: " + "".join(range_links) + "</div>"
        compact_url = (
            f"/kid/invest?symbol={active_instrument.symbol}&range={selected_range}&chart={CHART_VIEW_COMPACT}"
        )
        detail_url = (
            f"/kid/invest?symbol={active_instrument.symbol}&range={selected_range}&chart={CHART_VIEW_DETAIL}"
        )
        chart_toggle_html = (
            "<div class='chart-toggle'>Chart: "
            + f"<a href='{compact_url}' class='{'active' if chart_mode == CHART_VIEW_COMPACT else ''}'>Compact</a>"
            + f"<a href='{detail_url}' class='{'active' if chart_mode == CHART_VIEW_DETAIL else ''}'>Detailed</a>"
            + "</div>"
        )
        lookup_value = html_escape(lookup_query)
        if lookup_query and lookup_results:
            suggestion_items = []
            for match in lookup_results:
                symbol_val = match.get("symbol") or ""
                name_val = match.get("name") or symbol_val
                kind_val = match.get("kind") or INSTRUMENT_KIND_STOCK
                kind_label = "Crypto" if kind_val == INSTRUMENT_KIND_CRYPTO else "Stock"
                suggestion_items.append(
                    "<li><b>"
                    + html_escape(symbol_val)
                    + "</b> — "
                    + html_escape(name_val)
                    + f" <span class='muted'>({kind_label})</span>"
                    + "<form method='post' action='/kid/invest/track' class='inline' style='margin-top:6px;'>"
                    + f"<input type='hidden' name='symbol' value='{html_escape(symbol_val)}'>"
                    + f"<input type='hidden' name='name' value='{html_escape(name_val)}'>"
                    + f"<input type='hidden' name='kind' value='{html_escape(kind_val)}'>"
                    + f"<input type='hidden' name='range' value='{selected_range}'>"
                    + f"<input type='hidden' name='chart' value='{chart_mode}'>"
                    + "<button type='submit'>Track</button></form></li>"
                )
            search_results_html = (
                "<ul style='margin:10px 0 0 18px; padding:0; list-style:disc;'>"
                + "".join(suggestion_items)
                + "</ul>"
            )
        elif lookup_query:
            search_results_html = "<div class='muted' style='margin-top:6px;'>No matches found.</div>"
        else:
            search_results_html = "<div class='muted' style='margin-top:6px;'>Try symbols like AAPL, VTI, or BTC-USD.</div>"
        search_card_html = f"""
        <div class='card'>
          <h3>Add a ticker</h3>
          <p class='muted'>Search by ticker or name to add a new stock, fund, or crypto.</p>
          <form method='get' action='/kid/invest' class='inline'>
            <input type='hidden' name='symbol' value='{html_escape(instrument_symbol_raw)}'>
            <input type='hidden' name='range' value='{selected_range}'>
            <input type='hidden' name='chart' value='{chart_mode}'>
            <input name='lookup' placeholder='Search ticker or name' value='{lookup_value}'>
            <button type='submit'>Search</button>
          </form>
          {search_results_html}
        </div>
        """

        instrument_label = html_escape(active_instrument.name or instrument_symbol_raw)
        instrument_symbol = html_escape(instrument_symbol_raw)
        kind_label = "Crypto" if active_instrument.kind == INSTRUMENT_KIND_CRYPTO else "Stock"
        unit_label = "per coin" if active_instrument.kind == INSTRUMENT_KIND_CRYPTO else "per share"
        allow_remove = _normalize_symbol(instrument_symbol_raw) != _normalize_symbol(DEFAULT_MARKET_SYMBOL)
        remove_button_html = ""
        if allow_remove:
            remove_button_html = (
                "<form method='post' action='/kid/invest/delete' style='margin-top:10px;'>"
                f"<input type='hidden' name='symbol' value='{instrument_symbol}'>"
                f"<input type='hidden' name='range' value='{selected_range}'>"
                f"<input type='hidden' name='chart' value='{chart_mode}'>"
                "<button type='submit' class='danger secondary'>Remove from dashboard</button>"
                "</form>"
            )
        has_chart_history = len(history) >= 2
        chart_modal_id = f"chart-{_normalize_symbol(active_instrument.symbol).lower()}"
        chart_display_html = (
            f"<a href='#{chart_modal_id}' class='chart-popout' aria-label='Expand price chart'>{svg}</a>"
        )
        chart_hint_html = "<div class='chart-hint'>Tap or click the chart to expand.</div>" if has_chart_history else ""
        chart_modal_html = f"""
        <div id='{chart_modal_id}' class='modal-overlay chart-modal'>
          <div class='modal-card chart-modal__card'>
            <div class='modal-head'><h3>{instrument_label} — expanded chart</h3><a href='#' class='pill'>Close</a></div>
            <div class='chart-modal__body'>{enlarged_svg}</div>
          </div>
        </div>
        """

        inner = f"""
        {notice_html}{search_card_html}
        {tabs_html}
        <div class='card'>
          <h3>Investing — {instrument_label}</h3>
          <div class='muted'>{instrument_symbol} • {kind_label}</div>
          <div style='margin-bottom:12px;'><b>Available Balance:</b> {usd(balance_c)}</div>
          <div class='grid investing-grid'>
            <div class='card investing-chart-card'>
              <div><b>Current Price</b></div>
              <div style='font-size:28px; font-weight:800; margin-top:6px;'>{usd(metrics['price_c'])}</div>
              <div class='muted'>{unit_label}</div>
              {chart_display_html}
              {chart_hint_html}
              {range_selector_html}
              {chart_toggle_html}
            </div>
            <div class='card'>
              <div><b>Your Holdings</b></div>
              <div style='margin-top:6px;'>Shares: <b>{metrics['shares']:.4f}</b></div>
              <div>Value: <b>{usd(metrics['market_value_c'])}</b></div>
              <div>Avg Cost: <b>{usd(metrics['avg_cost_c'])}</b></div>
              <div>Invested: <b>{usd(metrics['invested_cost_c'])}</b></div>
              <div style='color:#{'16a34a' if metrics['unrealized_pl_c']>=0 else 'dc2626'};'>Unrealized P/L: <b>{fmt(metrics['unrealized_pl_c'])}</b></div>
              <div style='color:#{'16a34a' if metrics['realized_pl_c']>=0 else 'dc2626'};'>Realized P/L: <b>{fmt(metrics['realized_pl_c'])}</b></div>
              {remove_button_html}
            </div>
          </div>
          <h4 style='margin-top:12px;'>Buy (deposit from balance)</h4>
          <form method='post' action='/kid/invest/buy' class='inline'>
            <input type='hidden' name='symbol' value='{instrument_symbol}'>
            <input type='hidden' name='range' value='{selected_range}'>
            <input type='hidden' name='chart' value='{chart_mode}'>
            <input name='amount' type='text' data-money placeholder='amount $' required>
            <button type='submit'>Buy</button>
          </form>
          <h4 style='margin-top:12px;'>Sell (withdraw to balance)</h4>
          <form method='post' action='/kid/invest/sell' class='inline'>
            <input type='hidden' name='symbol' value='{instrument_symbol}'>
            <input type='hidden' name='range' value='{selected_range}'>
            <input type='hidden' name='chart' value='{chart_mode}'>
            <input name='amount' type='text' data-money placeholder='amount $' required>
            <button type='submit' class='danger'>Sell</button>
          </form>
        </div>
        <div class='card'>
          <h3>Certificates of Deposit</h3>
          <p class='muted'>Lock part of your balance to earn interest.</p>
          {cd_summary_html}
          {cash_out_form}
          <h4 style='margin-top:12px;'>Open a certificate</h4>
          <form method='post' action='/kid/invest/cd/open' class='inline'>
            <input type='hidden' name='symbol' value='{instrument_symbol}'>
            <input type='hidden' name='range' value='{selected_range}'>
            <input type='hidden' name='chart' value='{chart_mode}'>
            <input name='amount' type='text' data-money placeholder='amount $' required>
            <label style='margin-left:6px;'>Term</label>
            <select name='term_choice'>
              {term_options_html}
            </select>
            <button type='submit' style='margin-left:6px;'>Lock Savings</button>
          </form>
          <p class='muted' style='margin-top:6px;'>Funds move from your balance into the certificate.</p>
          <table style='margin-top:10px;'><tr><th>Principal</th><th>Rate</th><th>Term</th><th>Value Today</th><th>Progress</th><th>Status</th><th>Actions</th></tr>{cert_rows}</table>
        </div>
        <p class='muted' style='margin-top:10px;'><a href='/kid'>← Back to My Account</a></p>
        {chart_modal_html}
        """
        return render_page(request, f"Investing — {instrument_label}", inner)
    except Exception:
        body = """
        <div class='card'>
          <h3>Investing</h3>
          <p class='muted'>The investing dashboard hit an error. Check server logs.</p>
          <a href='/kid'><button>Back</button></a>
        </div>
        """
        return render_page(request, "Investing — Error", body)
@app.post("/kid/invest/track")
def kid_invest_track(
    request: Request,
    symbol: str = Form(...),
    name: str = Form(""),
    kind: str = Form(INSTRUMENT_KIND_STOCK),
    range_code: str = Form(DEFAULT_PRICE_RANGE, alias="range"),
    chart_view: str = Form(DEFAULT_CHART_VIEW, alias="chart"),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    normalized_symbol = _normalize_symbol(symbol)
    if not normalized_symbol:
        set_kid_notice(request, "Enter a ticker symbol to track.", "error")
        return RedirectResponse("/kid/invest", status_code=302)
    resolved = lookup_symbol_profile(normalized_symbol)
    if resolved:
        resolved_name = resolved.get("name") or normalized_symbol
        resolved_kind = resolved.get("kind") or kind
    else:
        resolved_name = name.strip() or normalized_symbol
        resolved_kind = kind if kind in {INSTRUMENT_KIND_STOCK, INSTRUMENT_KIND_CRYPTO} else INSTRUMENT_KIND_STOCK
    instrument = add_market_instrument(normalized_symbol, resolved_name, resolved_kind)
    if not instrument:
        set_kid_notice(request, "Could not add that symbol.", "error")
        return RedirectResponse("/kid/invest", status_code=302)
    existing_link = None
    with Session(engine) as session:
        existing_link = session.exec(
            select(KidMarketInstrument)
            .where(KidMarketInstrument.kid_id == kid_id)
            .where(KidMarketInstrument.symbol == instrument.symbol)
        ).first()
        if not existing_link:
            session.add(KidMarketInstrument(kid_id=kid_id, symbol=instrument.symbol))
            session.commit()
    message = f"Tracking {instrument.symbol}."
    if existing_link:
        message = f"Already tracking {instrument.symbol}."
    set_kid_notice(request, message, "success")
    next_range = normalize_history_range(range_code)
    next_chart = normalize_chart_view(chart_view)
    return RedirectResponse(
        f"/kid/invest?symbol={instrument.symbol}&range={next_range}&chart={next_chart}",
        status_code=302,
    )


@app.post("/kid/invest/buy")
def kid_invest_buy(
    request: Request,
    amount: str = Form(...),
    symbol: str = Form(DEFAULT_MARKET_SYMBOL),
    range_code: str = Form(DEFAULT_PRICE_RANGE, alias="range"),
    chart_view: str = Form(DEFAULT_CHART_VIEW, alias="chart"),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    next_range = normalize_history_range(range_code)
    chart_mode = normalize_chart_view(chart_view)
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        return RedirectResponse(
            f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
            status_code=302,
        )
    normalized_symbol = _normalize_symbol(symbol)
    price_c = market_price_cents(normalized_symbol)
    if price_c <= 0:
        return RedirectResponse(
            f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
            status_code=302,
        )
    price = price_c / 100.0
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child or amount_c > child.balance_cents:
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        moment = _time_provider()
        shares = (amount_c / 100.0) / price
        holding = session.exec(
            select(Investment).where(Investment.kid_id == kid_id, Investment.fund == normalized_symbol)
        ).first()
        if not holding:
            holding = Investment(kid_id=kid_id, fund=normalized_symbol, shares=0.0)
        holding.shares += shares
        child.balance_cents -= amount_c
        child.updated_at = moment
        tx = InvestmentTx(
            kid_id=kid_id,
            fund=normalized_symbol,
            tx_type="buy",
            shares=shares,
            price_cents=price_c,
            amount_cents=-amount_c,
            realized_pl_cents=0,
        )
        session.add(holding)
        session.add(child)
        session.add(tx)
        session.add(
            Event(
                child_id=kid_id,
                change_cents=-amount_c,
                reason=f"invest_buy_{normalized_symbol}:{shares:.4f}sh @ {usd(price_c)}",
            )
        )
        session.commit()
    return RedirectResponse(
        f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
        status_code=302,
    )


@app.post("/kid/invest/sell")
def kid_invest_sell(
    request: Request,
    amount: str = Form(...),
    symbol: str = Form(DEFAULT_MARKET_SYMBOL),
    range_code: str = Form(DEFAULT_PRICE_RANGE, alias="range"),
    chart_view: str = Form(DEFAULT_CHART_VIEW, alias="chart"),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    next_range = normalize_history_range(range_code)
    chart_mode = normalize_chart_view(chart_view)
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        return RedirectResponse(
            f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
            status_code=302,
        )
    normalized_symbol = _normalize_symbol(symbol)
    price_c = market_price_cents(normalized_symbol)
    if price_c <= 0:
        return RedirectResponse(
            f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
            status_code=302,
        )
    price = price_c / 100.0
    with Session(engine) as session:
        holding = session.exec(
            select(Investment).where(Investment.kid_id == kid_id, Investment.fund == normalized_symbol)
        ).first()
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not holding or not child or holding.shares <= 0:
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        need_shares = (amount_c / 100.0) / price
        sell_shares = min(holding.shares, need_shares)
        proceeds_c = int(round(sell_shares * price * 100))
        if sell_shares <= 0 or proceeds_c <= 0:
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        txs = session.exec(
            select(InvestmentTx)
            .where(InvestmentTx.kid_id == kid_id, InvestmentTx.fund == normalized_symbol)
            .order_by(InvestmentTx.ts)
        ).all()
        running_shares = 0.0
        running_cost_c = 0
        for tx in txs:
            if tx.tx_type == "buy":
                running_cost_c += int(round(tx.shares * tx.price_cents))
                running_shares += tx.shares
            else:
                if running_shares > 1e-12:
                    avg_cost = running_cost_c / running_shares
                    sold_cost = int(round(tx.shares * avg_cost))
                    running_cost_c -= sold_cost
                    running_shares -= tx.shares
        avg_cost_before = (running_cost_c / running_shares) if running_shares > 1e-12 else 0
        sold_cost_c = int(round(sell_shares * avg_cost_before))
        realized_pl = proceeds_c - sold_cost_c
        holding.shares -= sell_shares
        child.balance_cents += proceeds_c
        moment = _time_provider()
        child.updated_at = moment
        tx = InvestmentTx(
            kid_id=kid_id,
            fund=normalized_symbol,
            tx_type="sell",
            shares=sell_shares,
            price_cents=price_c,
            amount_cents=proceeds_c,
            realized_pl_cents=realized_pl,
        )
        session.add(holding)
        session.add(child)
        session.add(tx)
        session.add(
            Event(
                child_id=kid_id,
                change_cents=proceeds_c,
                reason=f"invest_sell_{normalized_symbol}:{sell_shares:.4f}sh @ {usd(price_c)}",
            )
        )
        session.commit()
    return RedirectResponse(
        f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
        status_code=302,
    )


@app.post("/kid/invest/cd/open")
def kid_invest_cd_open(
    request: Request,
    amount: str = Form(...),
    term_choice: str = Form(...),
    symbol: str = Form(DEFAULT_MARKET_SYMBOL),
    range_code: str = Form(DEFAULT_PRICE_RANGE, alias="range"),
    chart_view: str = Form(DEFAULT_CHART_VIEW, alias="chart"),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    amount_c = to_cents_from_dollars_str(amount, 0)
    term_code, term_days, term_months_value = resolve_certificate_term(term_choice)
    if amount_c <= 0:
        next_range = normalize_history_range(range_code)
        chart_mode = normalize_chart_view(chart_view)
        return RedirectResponse(
            f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
            status_code=302,
        )
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child or amount_c > child.balance_cents:
            next_range = normalize_history_range(range_code)
            chart_mode = normalize_chart_view(chart_view)
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        rate_bps = get_cd_rate_bps(session, term_code)
        penalty_days = get_cd_penalty_days(session, term_code)
        now = _time_provider()
        certificate = Certificate(
            kid_id=kid_id,
            principal_cents=amount_c,
            rate_bps=rate_bps,
            term_months=term_months_value,
            term_days=term_days,
            opened_at=now,
            penalty_days=penalty_days,
        )
        child.balance_cents -= amount_c
        child.updated_at = now
        rate_pct = rate_bps / 100
        session.add(child)
        session.add(certificate)
        session.add(
            Event(
                child_id=kid_id,
                change_cents=-amount_c,
                reason=f"invest_cd_open:{term_code} @ {rate_pct:.2f}%",
            )
        )
        session.commit()
    next_range = normalize_history_range(range_code)
    chart_mode = normalize_chart_view(chart_view)
    set_kid_notice(request, f"Locked savings for a {term_code} certificate.", "success")
    return RedirectResponse(
        f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
        status_code=302,
    )


@app.post("/kid/invest/cd/mature")
def kid_invest_cd_mature(
    request: Request,
    symbol: str = Form(DEFAULT_MARKET_SYMBOL),
    range_code: str = Form(DEFAULT_PRICE_RANGE, alias="range"),
    chart_view: str = Form(DEFAULT_CHART_VIEW, alias="chart"),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    next_range = normalize_history_range(range_code)
    chart_mode = normalize_chart_view(chart_view)
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        certificates = session.exec(
            select(Certificate)
            .where(Certificate.kid_id == kid_id)
            .where(Certificate.matured_at == None)  # noqa: E711
        ).all()
        moment = _time_provider()
        payout_total = 0
        matured_count = 0
        for certificate in certificates:
            if moment >= certificate_maturity_date(certificate):
                payout = certificate_maturity_value_cents(certificate)
                payout_total += payout
                matured_count += 1
                certificate.matured_at = moment
                session.add(certificate)
        if payout_total > 0 and matured_count > 0:
            child.balance_cents += payout_total
            child.updated_at = moment
            session.add(child)
            session.add(
                Event(
                    child_id=kid_id,
                    change_cents=payout_total,
                    reason=f"invest_cd_mature:{matured_count}x",
                )
            )
            session.commit()
            plural = "s" if matured_count != 1 else ""
            set_kid_notice(
                request,
                f"Cashed out {matured_count} matured certificate{plural}.",
                "success",
            )
        else:
            session.rollback()
            set_kid_notice(request, "No certificates were ready to mature yet.", "error")
    return RedirectResponse(
        f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
        status_code=302,
    )


@app.post("/kid/invest/cd/cashout")
def kid_invest_cd_cashout(
    request: Request,
    certificate_id: int = Form(...),
    symbol: str = Form(DEFAULT_MARKET_SYMBOL),
    range_code: str = Form(DEFAULT_PRICE_RANGE, alias="range"),
    chart_view: str = Form(DEFAULT_CHART_VIEW, alias="chart"),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    next_range = normalize_history_range(range_code)
    chart_mode = normalize_chart_view(chart_view)
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        certificate = session.get(Certificate, certificate_id)
        if not child or not certificate or certificate.kid_id != kid_id:
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        if certificate.matured_at is not None:
            session.delete(certificate)
            session.commit()
            set_kid_notice(request, "Removed the cashed-out certificate.", "success")
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        moment = _time_provider()
        maturity = certificate_maturity_date(certificate)
        if moment < maturity:
            return RedirectResponse(
                f"/kid/invest/cd/sell?certificate_id={certificate.id}&symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        payout_c = certificate_maturity_value_cents(certificate)
        certificate.matured_at = moment
        child.balance_cents += payout_c
        child.updated_at = moment
        session.add(child)
        session.add(certificate)
        session.add(
            Event(
                child_id=kid_id,
                change_cents=payout_c,
                reason=f"invest_cd_mature:1x",
            )
        )
        session.commit()
    set_kid_notice(request, "Cashed out your certificate!", "success")
    return RedirectResponse(
        f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
        status_code=302,
    )


@app.get("/kid/invest/cd/sell", response_class=HTMLResponse)
def kid_invest_cd_sell_confirm(
    request: Request,
    certificate_id: int = Query(...),
    symbol: str = Query(DEFAULT_MARKET_SYMBOL),
    range_code: str = Query(DEFAULT_PRICE_RANGE, alias="range"),
    chart_view: str = Query(DEFAULT_CHART_VIEW, alias="chart"),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    with Session(engine) as session:
        certificate = session.get(Certificate, certificate_id)
        if not certificate or certificate.kid_id != kid_id:
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={normalize_history_range(range_code)}&chart={normalize_chart_view(chart_view)}",
                status_code=302,
            )
    if certificate.matured_at is not None:
        return RedirectResponse(
            f"/kid/invest?symbol={symbol}&range={normalize_history_range(range_code)}&chart={normalize_chart_view(chart_view)}",
            status_code=302,
        )
    moment = _time_provider()
    gross_c, penalty_c, net_c = certificate_sale_breakdown_cents(certificate, at=moment)
    maturity = certificate_maturity_date(certificate)
    matured = moment >= maturity
    term_days = certificate_term_days(certificate)
    penalty_days = min(certificate.penalty_days, term_days)
    rate_display = certificate.rate_bps / 100
    next_range = normalize_history_range(range_code)
    chart_mode = normalize_chart_view(chart_view)
    if matured or penalty_c == 0:
        penalty_note = "No penalty applies. Selling now returns your savings to the balance."
    else:
        day_label = "day" if penalty_days == 1 else "days"
        penalty_note = (
            f"Selling early forfeits {usd(penalty_c)} of interest (up to {penalty_days} {day_label})."
        )
    button_class_attr = " class='danger'" if penalty_c > 0 and not matured else ""
    symbol_safe = html_escape(symbol)
    inner = f"""
    <div class='card'>
      <h3>Sell certificate?</h3>
      <p>{penalty_note}</p>
      <div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:8px; margin-top:12px;'>
        <div><b>Principal</b><div>{usd(certificate.principal_cents)}</div></div>
        <div><b>Rate</b><div>{rate_display:.2f}% APR</div></div>
        <div><b>Term</b><div>{certificate_term_label(certificate)}</div></div>
        <div><b>Value today</b><div>{usd(gross_c)}</div></div>
        <div><b>Penalty if sold</b><div>{usd(penalty_c)}</div></div>
        <div><b>Net to balance</b><div>{usd(net_c)}</div></div>
      </div>
      <form method='post' action='/kid/invest/cd/sell' style='margin-top:16px;'>
        <input type='hidden' name='certificate_id' value='{certificate.id}'>
        <input type='hidden' name='symbol' value='{symbol_safe}'>
        <input type='hidden' name='range' value='{next_range}'>
        <input type='hidden' name='chart' value='{chart_mode}'>
        <button type='submit'{button_class_attr}>Yes, sell certificate</button>
      </form>
      <p class='muted' style='margin-top:12px;'><a href='/kid/invest?symbol={symbol_safe}&range={next_range}&chart={chart_mode}'>No, keep it growing</a></p>
    </div>
    """
    return render_page(request, "Sell Certificate", inner)


@app.post("/kid/invest/cd/sell")
def kid_invest_cd_sell(
    request: Request,
    certificate_id: int = Form(...),
    symbol: str = Form(DEFAULT_MARKET_SYMBOL),
    range_code: str = Form(DEFAULT_PRICE_RANGE, alias="range"),
    chart_view: str = Form(DEFAULT_CHART_VIEW, alias="chart"),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    next_range = normalize_history_range(range_code)
    chart_mode = normalize_chart_view(chart_view)
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        certificate = session.get(Certificate, certificate_id)
        if not child or not certificate or certificate.kid_id != kid_id:
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        if certificate.matured_at is not None:
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        moment = _time_provider()
        gross_c, penalty_c, net_c = certificate_sale_breakdown_cents(certificate, at=moment)
        certificate.matured_at = moment
        session.add(certificate)
        child.balance_cents += gross_c
        if penalty_c > 0:
            child.balance_cents -= penalty_c
        child.updated_at = moment
        session.add(child)
        status_tag = "matured" if moment >= certificate_maturity_date(certificate) else "early"
        session.add(
            Event(
                child_id=kid_id,
                change_cents=gross_c,
                reason=f"invest_cd_sell_{status_tag}:{certificate_term_code(certificate)}",
            )
        )
        if penalty_c > 0:
            session.add(
                Event(
                    child_id=kid_id,
                    change_cents=-penalty_c,
                    reason=f"invest_cd_penalty:{certificate.penalty_days}d",
                )
            )
        session.commit()
    set_kid_notice(request, f"Added {usd(net_c)} to your balance.", "success")
    return RedirectResponse(
        f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
        status_code=302,
    )


@app.post("/kid/invest/delete")
def kid_invest_delete(
    request: Request,
    symbol: str = Form(...),
    range_code: str = Form(DEFAULT_PRICE_RANGE, alias="range"),
    chart_view: str = Form(DEFAULT_CHART_VIEW, alias="chart"),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    normalized_symbol = _normalize_symbol(symbol)
    next_range = normalize_history_range(range_code)
    chart_mode = normalize_chart_view(chart_view)
    if not normalized_symbol:
        set_kid_notice(request, "Choose a stock to remove first.", "error")
        return RedirectResponse(
            f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
            status_code=302,
        )
    if normalized_symbol == _normalize_symbol(DEFAULT_MARKET_SYMBOL):
        set_kid_notice(request, "The default market cannot be removed.", "error")
        return RedirectResponse(
            f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
            status_code=302,
        )
    removed = False
    already_removed = False
    with Session(engine) as session:
        holding = session.exec(
            select(Investment).where(Investment.kid_id == kid_id, Investment.fund == normalized_symbol)
        ).first()
        if holding and abs(holding.shares) > 1e-4:
            set_kid_notice(request, "Sell your shares before removing this market.", "error")
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        link = session.exec(
            select(KidMarketInstrument)
            .where(KidMarketInstrument.kid_id == kid_id)
            .where(KidMarketInstrument.symbol == normalized_symbol)
        ).first()
        if not link:
            already_removed = True
        else:
            session.delete(link)
            removed = True
        if holding:
            session.delete(holding)
        session.commit()
    remaining_symbols = [
        sym for sym in list_kid_market_symbols(kid_id) if sym != normalized_symbol
    ]
    fallback_symbol = remaining_symbols[0] if remaining_symbols else DEFAULT_MARKET_SYMBOL
    if already_removed and not removed:
        set_kid_notice(request, f"{normalized_symbol} was already removed.", "success")
    else:
        set_kid_notice(request, f"Removed {normalized_symbol} from your dashboard.", "success")
    return RedirectResponse(
        f"/kid/invest?symbol={fallback_symbol}&range={next_range}&chart={chart_mode}",
        status_code=302,
    )


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    if admin_authed(request):
        return RedirectResponse("/admin", status_code=302)
    inner = """
    <div class='card'>
      <h3>Parent Login</h3>
      <form method='post' action='/admin/login'>
        <label>PIN</label><input name='pin' placeholder='****' required>
        <button type='submit' style='margin-top:10px;'>Sign In</button>
      </form>
      <p class='muted' style='margin-top:6px;'><a href='/'>← Back</a></p>
    </div>
    """
    return render_page(request, "Parent Login", inner)


@app.post("/admin/login")
def admin_login(request: Request, pin: str = Form(...)):
    with Session(engine) as session:
        role = resolve_admin_role(pin, session=session)
    if not role:
        body = "<div class='card'><p style='color:#ff6b6b;'>Incorrect PIN.</p><p><a href='/admin/login'>Try again</a></p></div>"
        return render_page(request, "Parent Login", body)
    request.session["admin_role"] = role
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/logout")
def admin_logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)


def _role_badge(role: Optional[str]) -> str:
    if not role:
        return ""
    label = parent_role_label(role)
    return f"<span class='pill'>Signed in as {html_escape(label)}</span>"


def _kid_options(kids: Iterable[Child]) -> str:
    return "".join(f"<option value='{kid.kid_id}'>{kid.name} ({kid.kid_id})</option>" for kid in kids)


@app.get("/admin", response_class=HTMLResponse)
def admin_home(
    request: Request,
    section: str = Query("overview"),
    child: Optional[str] = Query(None),
    events_search: str = Query(""),
    events_dir: str = Query("all"),
    events_kid: str = Query(""),
):
    if (redirect := require_admin(request)) is not None:
        return redirect
    run_weekly_allowance_if_needed()
    role = admin_role(request)
    selected_section = (section or "overview").strip().lower()
    selected_child = (child or "").strip()
    cd_rates_bps = {code: DEFAULT_CD_RATE_BPS for code, _, _ in CD_TERM_OPTIONS}
    admin_events_query = (events_search or "").strip()
    admin_events_dir = (events_dir or "all").strip().lower()
    if admin_events_dir not in {"all", "credit", "debit", "zero"}:
        admin_events_dir = "all"
    admin_events_kid = (events_kid or "").strip()
    moment_admin = _time_provider()
    notice_msg, notice_kind = pop_admin_notice(request)
    notice_html = ""
    if notice_msg:
        style = (
            "background:#fee2e2; border-left:4px solid #fca5a5; color:#b91c1c;"
            if notice_kind == "error"
            else "background:#dcfce7; border-left:4px solid #86efac; color:#166534;"
        )
        notice_html = f"<div class='card' style='margin-bottom:12px; {style}'><div>{html_escape(notice_msg)}</div></div>"
    with Session(engine) as session:
        kids = session.exec(select(Child).order_by(Child.name)).all()
        all_kids = list(kids)
        admin_privs = current_admin_privileges(request, session)
        prizes = session.exec(select(Prize).order_by(desc(Prize.created_at))).all()
        events_limit = 160 if (admin_events_query or admin_events_kid or admin_events_dir != "all") else 60
        events = session.exec(
            select(Event).order_by(desc(Event.timestamp)).limit(events_limit)
        ).all()
        analytics_events = session.exec(
            select(Event).order_by(desc(Event.timestamp)).limit(240)
        ).all()
        pending = session.exec(
            select(ChoreInstance, Chore, Child)
            .where(ChoreInstance.status == "pending")
            .where(ChoreInstance.chore_id == Chore.id)
            .where(Chore.kid_id == Child.kid_id)
            .order_by(desc(ChoreInstance.completed_at))
        ).all()
        approval_pairs = session.exec(
            select(ChoreInstance, Event)
            .where(ChoreInstance.status == "paid")
            .where(ChoreInstance.paid_event_id == Event.id)
            .order_by(desc(Event.timestamp))
            .limit(120)
        ).all()
        global_pending = session.exec(
            select(GlobalChoreClaim, Child, Chore)
            .where(GlobalChoreClaim.chore_id == Chore.id)
            .where(Chore.kid_id == GLOBAL_CHORE_KID_ID)
            .where(Child.kid_id == GlobalChoreClaim.kid_id)
            .where(GlobalChoreClaim.status == GLOBAL_CHORE_STATUS_PENDING)
            .order_by(desc(GlobalChoreClaim.submitted_at))
        ).all()
        bonus_on_all = MetaDAO.get(session, "rule_bonus_all_complete") == "1"
        bonus_cents = int(MetaDAO.get(session, "rule_bonus_cents") or "0")
        penalty_on_miss = MetaDAO.get(session, "rule_penalty_on_miss") == "1"
        penalty_cents = int(MetaDAO.get(session, "rule_penalty_cents") or "0")
        needs = session.exec(
            select(Goal, Child)
            .where(Goal.kid_id == Child.kid_id)
            .where(Goal.saved_cents >= Goal.target_cents)
            .order_by(desc(Goal.created_at))
        ).all()
        cd_rates_bps = get_all_cd_rate_bps(session)
        cd_penalty_days = get_all_cd_penalty_days(session)
        active_certs = session.exec(
            select(Certificate).where(Certificate.matured_at == None)  # noqa: E711
        ).all()
        time_settings = get_time_settings(session)
        parent_admins = all_parent_admins(session)
        privileges_by_role = {
            entry["role"]: load_admin_privileges(session, entry["role"])
            for entry in parent_admins
        }
        approved_global_claims = session.exec(
            select(GlobalChoreClaim)
            .where(GlobalChoreClaim.status == GLOBAL_CHORE_STATUS_APPROVED)
        ).all()
        pending_money_requests = session.exec(
            select(MoneyRequest).where(MoneyRequest.status == "pending")
        ).all()
        goals_all = session.exec(select(Goal)).all()
        marketplace_listings = _safe_marketplace_list(
            session,
            select(MarketplaceListing).order_by(desc(MarketplaceListing.created_at)),
        )

    def _kid_allowed(identifier: Optional[str]) -> bool:
        if not identifier:
            return True
        return admin_privs.allows_kid(identifier)

    if not admin_privs.is_all_kids:
        kids = [kid for kid in kids if admin_privs.allows_kid(kid.kid_id)]
    else:
        kids = list(kids)

    events = [event for event in events if _kid_allowed(event.child_id)]
    analytics_events = [event for event in analytics_events if _kid_allowed(event.child_id)]
    approval_pairs = [pair for pair in approval_pairs if _kid_allowed(pair[1].child_id)]
    pending = [item for item in pending if _kid_allowed(item[2].kid_id)]
    global_pending = [item for item in global_pending if _kid_allowed(item[1].kid_id)]
    approved_global_claims = [
        claim for claim in approved_global_claims if _kid_allowed(claim.kid_id)
    ]
    pending_money_requests = [
        req
        for req in pending_money_requests
        if _kid_allowed(req.from_kid_id) or _kid_allowed(req.to_kid_id)
    ]
    goals_all = [goal for goal in goals_all if _kid_allowed(goal.kid_id)]
    needs = [entry for entry in needs if _kid_allowed(entry[1].kid_id)]
    marketplace_listings = [
        listing
        for listing in marketplace_listings
        if _kid_allowed(listing.owner_kid_id)
        and (not listing.claimed_by or _kid_allowed(listing.claimed_by))
    ]

    approved_lookup: Dict[Tuple[int, str], List[GlobalChoreClaim]] = {}
    for approved in approved_global_claims:
        approved_lookup.setdefault((approved.chore_id, approved.period_key), []).append(approved)
    kids_by_id = {kid.kid_id: kid for kid in kids}
    all_kids_by_id = {kid.kid_id: kid for kid in all_kids}
    total_cash_c = sum(child.balance_cents for child in kids)
    total_market_value_c = 0
    total_cd_value_c = 0
    total_unrealized_c = 0
    total_realized_c = 0
    total_cd_ready = 0
    total_active_certificates = 0
    portfolio_summaries: Dict[str, Dict[str, Any]] = {}
    for kid in kids:
        instruments_for_kid = list_market_instruments_for_kid(kid.kid_id)
        holdings_details: List[Dict[str, Any]] = []
        kid_market_total = 0
        kid_invested_total = 0
        kid_unrealized_total = 0
        kid_realized_total = 0
        active_holdings: List[Dict[str, Any]] = []
        for inst in instruments_for_kid:
            metrics = compute_holdings_metrics(kid.kid_id, inst.symbol)
            holdings_details.append(
                {
                    "symbol": inst.symbol,
                    "name": inst.name or inst.symbol,
                    "kind": inst.kind,
                    "metrics": metrics,
                }
            )
            kid_market_total += metrics["market_value_c"]
            kid_invested_total += metrics["invested_cost_c"]
            kid_unrealized_total += metrics["unrealized_pl_c"]
            kid_realized_total += metrics["realized_pl_c"]
            if (
                metrics["shares"] > 1e-6
                or metrics["market_value_c"]
                or metrics["invested_cost_c"]
            ):
                active_holdings.append(
                    {
                        "symbol": inst.symbol,
                        "name": inst.name or inst.symbol,
                        "kind": inst.kind,
                        "metrics": metrics,
                    }
                )
        certificate_details: List[Dict[str, Any]] = []
        kid_cd_total = 0
        kid_cd_ready = 0
        for cert in active_certs:
            if cert.kid_id != kid.kid_id:
                continue
            value_c = certificate_value_cents(cert, at=moment_admin)
            kid_cd_total += value_c
            maturity_at = certificate_maturity_date(cert)
            matured = moment_admin >= maturity_at
            if matured:
                kid_cd_ready += 1
            certificate_details.append(
                {
                    "principal_cents": cert.principal_cents,
                    "rate_bps": cert.rate_bps,
                    "term_label": certificate_term_label(cert),
                    "value_cents": value_c,
                    "matures_at": maturity_at,
                    "matured": matured,
                    "penalty_days": cert.penalty_days,
                }
            )
        total_market_value_c += kid_market_total
        total_cd_value_c += kid_cd_total
        total_unrealized_c += kid_unrealized_total
        total_realized_c += kid_realized_total
        total_cd_ready += kid_cd_ready
        total_active_certificates += len(certificate_details)
        largest_position_c = max(
            (holding["metrics"]["market_value_c"] for holding in active_holdings),
            default=0,
        )
        largest_position_pct = (
            (largest_position_c / kid_market_total) if kid_market_total else 0.0
        )
        total_assets_c = kid.balance_cents + kid_market_total + kid_cd_total
        cash_ratio = (kid.balance_cents / total_assets_c) if total_assets_c else 0.0
        market_ratio = (kid_market_total / total_assets_c) if total_assets_c else 0.0
        cd_ratio = (kid_cd_total / total_assets_c) if total_assets_c else 0.0
        total_pl_c = kid_unrealized_total + kid_realized_total
        portfolio_summaries[kid.kid_id] = {
            "kid": kid,
            "cash_cents": kid.balance_cents,
            "holdings": holdings_details,
            "total_market_c": kid_market_total,
            "total_invested_c": kid_invested_total,
            "unrealized_pl_c": kid_unrealized_total,
            "realized_pl_c": kid_realized_total,
            "cd_value_c": kid_cd_total,
            "cd_ready": kid_cd_ready,
            "certificates": certificate_details,
            "holding_count": len(active_holdings),
            "largest_position_c": largest_position_c,
            "largest_position_pct": largest_position_pct,
            "total_assets_c": total_assets_c,
            "cash_ratio": cash_ratio,
            "market_ratio": market_ratio,
            "cd_ratio": cd_ratio,
            "total_pl_c": total_pl_c,
        }
    kid_options_html = _kid_options(kids)
    parent_options_html = "".join(
        f"<option value='{admin['role']}'>{html_escape(admin['label'])}</option>"
        for admin in parent_admins
    )
    if admin_privs.can_manage_allowance:
        goals_rows = "".join(
            (
                "<tr>"
                f"<td data-label='Kid'><b>{html_escape(child.name)}</b><div class='muted'>{child.kid_id}</div></td>"
                f"<td data-label='Goal'><b>{html_escape(goal.name)}</b></td>"
                f"<td data-label='Saved' class='right'>{usd(goal.saved_cents)} / {usd(goal.target_cents)}"
                f"<div class='muted'>{format_percent(percent_complete(goal.saved_cents, goal.target_cents))} complete</div></td>"
                f"<td data-label='Actions' class='right'>"
                f"<form class='inline' method='post' action='/admin/goal_grant'><input type='hidden' name='goal_id' value='{goal.id}'><button type='submit'>Grant Goal</button></form> "
                f"<form class='inline' method='post' action='/admin/goal_return_funds' style='margin-left:6px;'><input type='hidden' name='goal_id' value='{goal.id}'><button type='submit' class='danger'>Return Funds</button></form>"
                "</td></tr>"
            )
            for goal, child in needs
        ) or "<tr><td colspan='4' class='muted'>(none)</td></tr>"
        goals_card = (
            "<div class='card'>"
            "<h3>Goals Needing Action</h3>"
            "<table><tr><th>Kid</th><th>Goal</th><th>Saved</th><th>Actions</th></tr>"
            f"{goals_rows}</table>"
            "<p class='muted' style='margin-top:6px;'>Grant allows the kid to spend the saved amount. Return moves funds back to their balance.</p>"
            "</div>"
        )
    else:
        goals_rows = "".join(
            (
                "<tr>"
                f"<td data-label='Kid'><b>{html_escape(child.name)}</b><div class='muted'>{child.kid_id}</div></td>"
                f"<td data-label='Goal'><b>{html_escape(goal.name)}</b></td>"
                f"<td data-label='Saved' class='right'>{usd(goal.saved_cents)} / {usd(goal.target_cents)}"
                f"<div class='muted'>{format_percent(percent_complete(goal.saved_cents, goal.target_cents))} complete</div></td>"
                "</tr>"
            )
            for goal, child in needs
        ) or "<tr><td colspan='3' class='muted'>(none)</td></tr>"
        goals_card = (
            "<div class='card'>"
            "<h3>Goals Needing Action</h3>"
            "<p class='muted'>View-only — goals can be granted or refunded by a full administrator.</p>"
            "<table><tr><th>Kid</th><th>Goal</th><th>Saved</th></tr>"
            f"{goals_rows}</table>"
            "</div>"
        )
    marketplace_open = [
        listing for listing in marketplace_listings if listing.status == MARKETPLACE_STATUS_OPEN
    ]
    marketplace_claimed = [
        listing for listing in marketplace_listings if listing.status == MARKETPLACE_STATUS_CLAIMED
    ]
    marketplace_submitted = [
        listing for listing in marketplace_listings if listing.status == MARKETPLACE_STATUS_SUBMITTED
    ]
    marketplace_completed = [
        listing for listing in marketplace_listings if listing.status == MARKETPLACE_STATUS_COMPLETED
    ]
    marketplace_cancelled = [
        listing for listing in marketplace_listings if listing.status == MARKETPLACE_STATUS_CANCELLED
    ]
    marketplace_rejected = [
        listing for listing in marketplace_listings if listing.status == MARKETPLACE_STATUS_REJECTED
    ]
    escrow_total_c = sum(
        listing.offer_cents
        for listing in marketplace_listings
        if listing.status
        in {
            MARKETPLACE_STATUS_OPEN,
            MARKETPLACE_STATUS_CLAIMED,
            MARKETPLACE_STATUS_SUBMITTED,
        }
    )
    payout_total_c = sum(
        (listing.final_payout_cents or (listing.offer_cents + listing.chore_award_cents))
        for listing in marketplace_listings
        if listing.status == MARKETPLACE_STATUS_COMPLETED
    )
    status_styles_market = {
        MARKETPLACE_STATUS_OPEN: ("Open", "#dbeafe", "#1d4ed8"),
        MARKETPLACE_STATUS_CLAIMED: ("Claimed", "#fef3c7", "#b45309"),
        MARKETPLACE_STATUS_SUBMITTED: ("Submitted", "#e0e7ff", "#4338ca"),
        MARKETPLACE_STATUS_COMPLETED: ("Completed", "#dcfce7", "#166534"),
        MARKETPLACE_STATUS_CANCELLED: ("Cancelled", "#fee2e2", "#b91c1c"),
        MARKETPLACE_STATUS_REJECTED: ("Rejected", "#fee2e2", "#b91c1c"),
    }

    def _format_market_ts(value: Optional[datetime]) -> str:
        if not value:
            return "—"
        try:
            return value.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(value)

    pending_rows_parts: List[str] = []
    for inst, chore, child in pending:
        submitted = inst.completed_at.strftime("%Y-%m-%d %H:%M") if inst.completed_at else ""
        pending_rows_parts.append(
            "<tr>"
            f"<td data-label='Kid'><b>{html_escape(child.name)}</b><div class='muted'>{child.kid_id}</div></td>"
            f"<td data-label='Chore'><b>{html_escape(chore.name)}</b><div class='muted'>{chore.type}</div></td>"
            f"<td data-label='Award' class='right'><b>{usd(chore.award_cents)}</b></td>"
            f"<td data-label='Completed'>{submitted}</td>"
            "<td data-label='Actions' class='right'>"
            f"<form class='inline' method='post' action='/admin/chore_payout'><input type='hidden' name='instance_id' value='{inst.id}'><input type='hidden' name='redirect' value='/admin?section=payouts'>"
            "<input name='amount' type='text' data-money placeholder='override $ (optional)' style='max-width:150px'>"
            "<input name='reason' type='text' placeholder='reason (optional)' style='max-width:200px'>"
            "<button type='submit'>Payout</button></form> "
            f"<form class='inline' method='post' action='/admin/chore_deny' style='margin-left:6px;' onsubmit='return confirm(\"Deny and push back to Available?\");'><input type='hidden' name='instance_id' value='{inst.id}'><input type='hidden' name='redirect' value='/admin?section=payouts'><button type='submit' class='danger'>Deny</button></form>"
            "</td></tr>"
        )
    for listing in marketplace_submitted:
        owner = all_kids_by_id.get(listing.owner_kid_id)
        worker = all_kids_by_id.get(listing.claimed_by) if listing.claimed_by else None
        owner_name = html_escape(owner.name) if owner else html_escape(listing.owner_kid_id)
        worker_name = html_escape(worker.name) if worker else html_escape(listing.claimed_by or "Unknown")
        submitted_at = _format_market_ts(listing.submitted_at)
        total_value = listing.offer_cents + listing.chore_award_cents
        pending_rows_parts.append(
            "<tr>"
            f"<td data-label='Kid'><b>{worker_name}</b><div class='muted'>{html_escape(listing.claimed_by or '')}</div></td>"
            f"<td data-label='Chore'><b>{html_escape(listing.chore_name)}</b><div class='muted'>Marketplace from {owner_name}</div></td>"
            f"<td data-label='Award' class='right'><b>{usd(total_value)}</b><div class='muted'>Offer {usd(listing.offer_cents)} • Award {usd(listing.chore_award_cents)}</div></td>"
            f"<td data-label='Completed'>{submitted_at}</td>"
            "<td data-label='Actions' class='right'>"
            "<form class='inline' method='post' action='/admin/marketplace/payout'>"
            f"<input type='hidden' name='listing_id' value='{listing.id}'>"
            "<input type='hidden' name='redirect' value='/admin?section=payouts'>"
            "<input name='amount' type='text' data-money placeholder='override $ (optional)' style='max-width:150px'>"
            "<input name='reason' type='text' placeholder='reason (optional)' style='max-width:200px'>"
            "<button type='submit'>Approve</button></form> "
            "<form class='inline' method='post' action='/admin/marketplace/deny' style='margin-left:6px;' onsubmit='return confirm(\"Deny this marketplace payout?\");'>"
            f"<input type='hidden' name='listing_id' value='{listing.id}'>"
            "<input type='hidden' name='redirect' value='/admin?section=payouts'>"
            "<input name='reason' type='text' placeholder='reason (optional)' style='max-width:200px'>"
            "<button type='submit' class='danger'>Deny</button></form>"
            "</td></tr>"
        )
    global_groups: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for claim, claimant, chore in global_pending:
        key = (claim.chore_id, claim.period_key)
        entry = global_groups.setdefault(key, {"chore": chore, "claims": []})
        entry["claims"].append({"claim": claim, "child": claimant})
    multi_modals: List[str] = []
    for (chore_id_val, period_key), data in global_groups.items():
        chore = data["chore"]
        claim_entries = data["claims"]
        participants = ", ".join(html_escape(item["child"].name) for item in claim_entries)
        latest_submitted = max((item["claim"].submitted_at for item in claim_entries), default=None)
        submitted_display = latest_submitted.strftime("%Y-%m-%d %H:%M") if latest_submitted else ""
        key = (chore_id_val, period_key)
        approved_existing = approved_lookup.get(key, [])
        if len(claim_entries) == 1:
            entry = claim_entries[0]
            claim = entry["claim"]
            child_obj = entry["child"]
            pending_rows_parts.append(
                "<tr>"
                f"<td data-label='Kid'><b>{html_escape(child_obj.name)}</b><div class='muted'>{child_obj.kid_id}</div></td>"
                f"<td data-label='Chore'><b>{html_escape(chore.name)}</b><div class='muted'>Free-for-all ({html_escape(period_key)})</div></td>"
                f"<td data-label='Award' class='right'><b>{usd(chore.award_cents)}</b></td>"
                f"<td data-label='Completed'>{submitted_display}</td>"
                "<td data-label='Actions'>"
                "<form class='stacked-form' method='post' action='/admin/global_chore/claims'>"
                f"<input type='hidden' name='chore_id' value='{chore_id_val}'>"
                f"<input type='hidden' name='period_key' value='{html_escape(period_key)}'>"
                f"<input type='hidden' name='claim_ids' value='{claim.id}'>"
                "<input type='hidden' name='redirect' value='/admin?section=payouts'>"
                f"<input name='amount_{claim.id}' type='text' data-money placeholder='override $ (optional)'>"
                "<input name='reason' type='text' placeholder='reason (optional)'>"
                "<div class='actions'>"
                "<button type='submit' name='decision' value='approve'>Payout</button>"
                "<button type='submit' name='decision' value='reject' class='danger'>Deny</button>"
                "</div></form></td></tr>"
            )
            continue
        safe_period = re.sub(r"[^a-z0-9]+", "-", period_key.lower()) or "period"
        modal_id = f"modal-{chore_id_val}-{safe_period}"
        remaining_award = max(0, chore.award_cents - sum(cl.award_cents for cl in approved_existing))
        remaining_slots = max(0, chore.max_claimants - len(approved_existing))
        modal_rows = "".join(
            "<tr>"
            f"<td data-label='Select'><input type='checkbox' name='claim_ids' value='{item['claim'].id}' checked></td>"
            f"<td data-label='Kid'><b>{html_escape(item['child'].name)}</b><div class='muted'>{item['child'].kid_id}</div></td>"
            f"<td data-label='Submitted'>{item['claim'].submitted_at.strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td data-label='Override ($)' class='right'><input name='amount_{item['claim'].id}' type='text' data-money placeholder='optional'></td>"
            "</tr>"
            for item in claim_entries
        )
        if not modal_rows:
            modal_rows = "<tr><td colspan='4' class='muted'>(no pending claims)</td></tr>"
        multi_modals.append(
            "<div id='" + modal_id + "' class='modal-overlay'><div class='modal-card'>"
            + f"<div class='modal-head'><h3>Free-for-all — {html_escape(chore.name)}</h3><a href='#' class='pill'>Close</a></div>"
            + f"<p class='muted'>Period {html_escape(period_key)} • Award {usd(chore.award_cents)} • Max winners {chore.max_claimants}</p>"
            + f"<p class='muted'>Approved so far: {len(approved_existing)} • Slots left {remaining_slots} • Remaining award {usd(remaining_award)}</p>"
            + "<form method='post' action='/admin/global_chore/claims' class='stacked-form'>"
            + f"<input type='hidden' name='chore_id' value='{chore_id_val}'>"
            + f"<input type='hidden' name='period_key' value='{html_escape(period_key)}'>"
            + "<input type='hidden' name='redirect' value='/admin?section=payouts'>"
            + "<table><tr><th>Select</th><th>Kid</th><th>Submitted</th><th>Override ($)</th></tr>"
            + modal_rows
            + "</table>"
            + "<input name='reason' type='text' placeholder='reason (optional)'>"
            + "<div class='actions'>"
            + "<button type='submit' name='decision' value='approve'>Payout Selected</button>"
            + "<button type='submit' name='decision' value='reject' class='danger'>Deny Selected</button>"
            + "<a href='#' class='button-link secondary'>Cancel</a>"
            + "</div></form></div></div>"
        )
        pending_rows_parts.append(
            "<tr>"
            "<td data-label='Kid'><b>Multiple kids</b><div class='muted'>"
            + html_escape(participants or "No names")
            + "</div></td>"
            f"<td data-label='Chore'><b>{html_escape(chore.name)}</b><div class='muted'>Free-for-all ({html_escape(period_key)})</div></td>"
            f"<td data-label='Award' class='right'><b>{usd(chore.award_cents)}</b></td>"
            f"<td data-label='Completed'>{submitted_display}</td>"
            f"<td data-label='Actions' class='right'><a href='#{modal_id}' class='button-link'>Manage</a></td>"
            "</tr>"
        )
    pending_rows = "".join(pending_rows_parts) or "<tr><td colspan='5' class='muted'>(no pending)</td></tr>"
    if admin_privs.can_manage_payouts:
        pending_card = (
            "<div class='card'>"
            "<h3>Pending Payouts</h3>"
            "<table><tr><th>Kid</th><th>Chore</th><th>Award</th><th>Completed</th><th>Actions</th></tr>"
            f"{pending_rows}</table>"
            "<p class='muted' style='margin-top:6px;'>Audit trail: <a href='/admin/audit'>Pending vs Paid</a></p>"
            "</div>"
        )
    else:
        readonly_rows: List[str] = []
        for inst, chore, child in pending:
            submitted = inst.completed_at.strftime("%Y-%m-%d %H:%M") if inst.completed_at else ""
            readonly_rows.append(
                "<tr>"
                f"<td data-label='Kid'><b>{html_escape(child.name)}</b><div class='muted'>{child.kid_id}</div></td>"
                f"<td data-label='Chore'><b>{html_escape(chore.name)}</b><div class='muted'>{chore.type}</div></td>"
                f"<td data-label='Award' class='right'><b>{usd(chore.award_cents)}</b></td>"
                f"<td data-label='Completed'>{submitted}</td>"
                "</tr>"
            )
        for claim, claimant, chore in global_pending:
            submitted = claim.submitted_at.strftime("%Y-%m-%d %H:%M") if claim.submitted_at else ""
            readonly_rows.append(
                "<tr>"
                f"<td data-label='Kid'><b>{html_escape(claimant.name)}</b><div class='muted'>{claimant.kid_id}</div></td>"
                f"<td data-label='Chore'><b>{html_escape(chore.name)}</b><div class='muted'>Free-for-all ({html_escape(claim.period_key)})</div></td>"
                f"<td data-label='Award' class='right'><b>{usd(chore.award_cents)}</b></td>"
                f"<td data-label='Completed'>{submitted}</td>"
                "</tr>"
            )
        for listing in marketplace_submitted:
            worker = (
                all_kids_by_id.get(listing.claimed_by)
                if listing.claimed_by
                else None
            )
            owner = all_kids_by_id.get(listing.owner_kid_id)
            worker_name = html_escape(worker.name) if worker else html_escape(listing.claimed_by or "Unknown")
            owner_name = html_escape(owner.name) if owner else html_escape(listing.owner_kid_id)
            readonly_rows.append(
                "<tr>"
                f"<td data-label='Kid'><b>{worker_name}</b><div class='muted'>{html_escape(listing.claimed_by or '')}</div></td>"
                f"<td data-label='Chore'><b>{html_escape(listing.chore_name)}</b><div class='muted'>Marketplace from {owner_name}</div></td>"
                f"<td data-label='Award' class='right'><b>{usd(listing.offer_cents + listing.chore_award_cents)}</b></td>"
                f"<td data-label='Completed'>{_format_market_ts(listing.submitted_at)}</td>"
                "</tr>"
            )
        readonly_table = "".join(readonly_rows) or "<tr><td colspan='4' class='muted'>(no pending)</td></tr>"
        pending_card = (
            "<div class='card'>"
            "<h3>Pending Payouts</h3>"
            "<p class='muted'>View-only — contact a full administrator to approve or deny payouts.</p>"
            "<table><tr><th>Kid</th><th>Chore</th><th>Award</th><th>Completed</th></tr>"
            f"{readonly_table}</table>"
            "</div>"
        )
        multi_modals = []
    total_assets_c = total_cash_c + total_market_value_c + total_cd_value_c
    kids_count = len(kids)
    pending_payout_count = len(pending) + len(marketplace_submitted)
    global_pending_count = len(global_pending)
    money_request_count = len(pending_money_requests)
    goals_attention_count = len(needs)

    def fmt_signed(value: int) -> str:
        return f"{'+' if value >= 0 else ''}{usd(value)}"

    avg_cash_c = total_cash_c // kids_count if kids_count else 0
    avg_assets_c = total_assets_c // kids_count if kids_count else 0
    top_summary = max(
        portfolio_summaries.values(),
        key=lambda entry: entry["total_assets_c"],
        default=None,
    )
    quick_cards: List[str] = [
        (
            "<div class='stat-card'>"
            "<div class='stat-card__label'>Total assets</div>"
            f"<div class='stat-card__value'>{usd(total_assets_c)}</div>"
            f"<div class='stat-card__meta'>Cash {usd(total_cash_c)} • Markets {usd(total_market_value_c)} • CDs {usd(total_cd_value_c)}</div>"
            "</div>"
        ),
        (
            "<div class='stat-card'>"
            "<div class='stat-card__label'>Market exposure</div>"
            f"<div class='stat-card__value'>{usd(total_market_value_c)}</div>"
            f"<div class='stat-card__meta'>Unrealized {fmt_signed(total_unrealized_c)} • Realized {fmt_signed(total_realized_c)}</div>"
            "</div>"
        ),
        (
            "<div class='stat-card'>"
            "<div class='stat-card__label'>Action items</div>"
            f"<div class='stat-card__value'>{pending_payout_count + global_pending_count + money_request_count}</div>"
            f"<div class='stat-card__meta'>Payouts {pending_payout_count + global_pending_count} • Requests {money_request_count} • Goals {goals_attention_count}</div>"
            "</div>"
        ),
        (
            "<div class='stat-card'>"
            "<div class='stat-card__label'>Cash buffer</div>"
            f"<div class='stat-card__value'>{usd(total_cash_c)}</div>"
            f"<div class='stat-card__meta'>Avg per kid {usd(avg_cash_c)} • CDs ready {total_cd_ready}/{total_active_certificates}</div>"
            "</div>"
        ),
    ]
    if top_summary:
        top_kid = top_summary["kid"]
        quick_cards.append(
            "<div class='stat-card'>"
            "<div class='stat-card__label'>Top portfolio</div>"
            f"<div class='stat-card__value'>{usd(top_summary['total_assets_c'])}</div>"
            f"<div class='stat-card__meta'>{html_escape(top_kid.name)} ({top_kid.kid_id}) • Avg assets {usd(avg_assets_c)}</div>"
            "</div>"
        )
    overview_quick_html = "<div class='overview-stats-grid'>" + "".join(quick_cards) + "</div>"

    analytics_days = 7
    today_admin = moment_admin.date()

    def admin_day_range(days: int) -> List[date]:
        start = today_admin - timedelta(days=days - 1)
        return [start + timedelta(days=offset) for offset in range(days)]

    def admin_series(series_map: Dict[date, float]) -> List[Tuple[date, float]]:
        return [(day, float(series_map.get(day, 0))) for day in admin_day_range(analytics_days)]

    def admin_trend(series: Sequence[Tuple[date, float]], formatter: Callable[[float], str], bar_class: str) -> str:
        if not series:
            return "<div class='muted' style='margin-top:12px;'>No data yet.</div>"
        max_value = max((value for _, value in series), default=0.0)
        bars: List[str] = []
        for day, value in series:
            height_pct = 8.0 if max_value <= 0 else max(8.0, (value / max_value) * 100)
            label = day.strftime("%a")
            bars.append(
                f"<div class='trend-bars__bar {bar_class}' style='height:{height_pct:.0f}%;'>"
                + f"<span class='trend-bars__value'>{formatter(value)}</span>"
                + f"<span class='trend-bars__label'>{label}</span>"
                + "</div>"
            )
        return "<div class='trend-bars'>" + "".join(bars) + "</div>"

    payout_map: Dict[date, int] = {}
    chore_count_map: Dict[date, int] = {}
    interest_map: Dict[date, int] = {}
    for event in analytics_events:
        if not event.timestamp:
            continue
        day = event.timestamp.date()
        reason = (event.reason or "").lower()
        amount = event.change_cents or 0
        if reason.startswith("chore:") or reason.startswith("global_chore:"):
            payout_map[day] = payout_map.get(day, 0) + max(0, amount)
            if amount >= 0:
                chore_count_map[day] = chore_count_map.get(day, 0) + 1
        if "interest" in reason or reason.startswith("invest_cd_mature"):
            interest_map[day] = interest_map.get(day, 0) + max(0, amount)

    payout_series = admin_series(payout_map)
    chore_series = admin_series(chore_count_map)
    interest_series = admin_series(interest_map)
    payout_total_amount = int(sum(value for _, value in payout_series))
    chore_total = int(sum(value for _, value in chore_series))
    interest_total_amount = int(sum(value for _, value in interest_series))

    approval_hours: List[float] = []
    approval_by_day: Dict[date, List[float]] = {}
    for instance, event in approval_pairs:
        if not instance.completed_at or not event.timestamp:
            continue
        delta_hours = (event.timestamp - instance.completed_at).total_seconds() / 3600.0
        if delta_hours < 0:
            continue
        approval_hours.append(delta_hours)
        approval_by_day.setdefault(event.timestamp.date(), []).append(delta_hours)
    for claim in approved_global_claims:
        if not claim.submitted_at or not claim.approved_at:
            continue
        delta_hours = (claim.approved_at - claim.submitted_at).total_seconds() / 3600.0
        if delta_hours < 0:
            continue
        approval_hours.append(delta_hours)
        approval_by_day.setdefault(claim.approved_at.date(), []).append(delta_hours)

    approval_series = [
        (day, sum(values) / len(values) if values else 0.0)
        for day, values in ((d, approval_by_day.get(d, [])) for d in admin_day_range(analytics_days))
    ]
    overall_avg_hours = sum(approval_hours) / len(approval_hours) if approval_hours else 0.0
    overall_avg_display = f"{overall_avg_hours:.1f} hrs" if approval_hours else "—"

    goal_saved_total = sum(goal.saved_cents for goal in goals_all)
    jar_segments = [
        ("cash", "Spending", total_cash_c),
        ("goals", "Goals", goal_saved_total),
        ("market", "Investments", total_market_value_c),
        ("cd", "Certificates", total_cd_value_c),
    ]
    jar_total = sum(segment[2] for segment in jar_segments)
    if jar_total > 0:
        jar_bar_segments: List[str] = []
        jar_legend_items: List[str] = []
        for key, label, value in jar_segments:
            pct = (value / jar_total) * 100 if jar_total else 0
            jar_bar_segments.append(
                f"<div class='jar-bar__segment jar-bar__segment--{key}' style='width:{pct:.2f}%;' title='{label}: {usd(value)} ({pct:.1f}%)'></div>"
            )
            jar_legend_items.append(
                f"<div class='jar-legend__item'><span class='jar-legend__swatch jar-bar__segment--{key}'></span>{label}: {usd(value)} ({pct:.1f}%)</div>"
            )
        jar_bar_html = "<div class='jar-bar'>" + "".join(jar_bar_segments) + "</div>" + "<div class='jar-legend'>" + "".join(jar_legend_items) + "</div>"
    else:
        jar_bar_html = "<div class='muted' style='margin-top:12px;'>No balances tracked yet.</div>"

    analytics_card = (
        "<div class='card analytics-card'>"
        "<h3>Operations analytics</h3>"
        "<div class='muted'>Last 7 days of payouts, approvals, and interest.</div>"
        "<div class='analytics-grid'>"
        "<div>"
        "<div class='insight-label'>Total payouts</div>"
        f"<div class='insight-value'>{usd(payout_total_amount)}</div>"
        f"{admin_trend(payout_series, lambda v: usd(int(v)), 'trend-bars__bar--payout')}"
        "</div>"
        "<div>"
        "<div class='insight-label'>Chores completed</div>"
        f"<div class='insight-value'>{chore_total}</div>"
        f"{admin_trend(chore_series, lambda v: f'{int(v)}', 'trend-bars__bar--completed')}"
        "</div>"
        "<div>"
        "<div class='insight-label'>Avg approval time</div>"
        f"<div class='insight-value'>{overall_avg_display}</div>"
        f"{admin_trend(approval_series, lambda v: f'{v:.1f}h', 'trend-bars__bar--approval')}"
        "</div>"
        "<div>"
        "<div class='insight-label'>Interest posted</div>"
        f"<div class='insight-value'>{usd(interest_total_amount)}</div>"
        f"{admin_trend(interest_series, lambda v: usd(int(v)), 'trend-bars__bar--interest')}"
        "</div>"
        "<div>"
        "<div class='insight-label'>Jar distribution</div>"
        f"{jar_bar_html}"
        "</div>"
        "</div>"
        "</div>"
    )

    payout_preview_items: List[str] = []
    for inst, chore, child in pending[:4]:
        payout_preview_items.append(
            "<li>"
            + f"<b>{html_escape(child.name)}</b> • {usd(chore.award_cents)}"
            + f" <span class='muted'>({html_escape(chore.name)})</span>"
            + "</li>"
        )
    payout_section = (
        "<h4 style='margin-top:12px;'>Chore payouts</h4>"
        + (
            "<ul style='margin:6px 0 0 18px;'>" + "".join(payout_preview_items) + "</ul>"
            if payout_preview_items
            else "<div class='muted'>All caught up.</div>"
        )
        + "<a href='/admin?section=payouts' class='button-link secondary' style='margin-top:6px;'>Open payouts</a>"
    )
    global_summary: Dict[int, Dict[str, Any]] = {}
    for claim, claimant, chore in global_pending:
        entry = global_summary.setdefault(chore.id, {"chore": chore, "count": 0})
        entry["count"] += 1
    global_preview_items: List[str] = []
    for entry in list(global_summary.values())[:3]:
        chore = entry["chore"]
        count = entry["count"]
        global_preview_items.append(
            f"<li><b>{html_escape(chore.name)}</b> • {count} claim{'s' if count != 1 else ''}</li>"
        )
    global_section = (
        "<h4 style='margin-top:12px;'>Free-for-all claims</h4>"
        + (
            "<ul style='margin:6px 0 0 18px;'>" + "".join(global_preview_items) + "</ul>"
            if global_preview_items
            else "<div class='muted'>No submissions waiting.</div>"
        )
        + "<a href='/admin?section=payouts' class='button-link secondary' style='margin-top:6px;'>Review claims</a>"
    )
    request_preview_items: List[str] = []
    for req in pending_money_requests[:4]:
        sender = kids_by_id.get(req.from_kid_id)
        recipient = kids_by_id.get(req.to_kid_id)
        sender_label = html_escape(sender.name) if sender else html_escape(req.from_kid_id)
        recipient_label = html_escape(recipient.name) if recipient else html_escape(req.to_kid_id)
        reason_note = f" <span class='muted'>({html_escape(req.reason)})</span>" if req.reason else ""
        request_preview_items.append(
            f"<li><b>{sender_label}</b> → {recipient_label} • {usd(req.amount_cents)}{reason_note}</li>"
        )
    requests_section = (
        "<h4 style='margin-top:12px;'>Money requests</h4>"
        + (
            "<ul style='margin:6px 0 0 18px;'>" + "".join(request_preview_items) + "</ul>"
            if request_preview_items
            else "<div class='muted'>No pending requests.</div>"
        )
        + "<a href='/admin?section=accounts' class='button-link secondary' style='margin-top:6px;'>Manage transfers</a>"
    )
    goal_preview_items: List[str] = []
    for goal, child in needs[:4]:
        goal_preview_items.append(
            f"<li><b>{html_escape(child.name)}</b> • {html_escape(goal.name)} ({usd(goal.saved_cents)} / {usd(goal.target_cents)})</li>"
        )
    goals_section = (
        "<h4 style='margin-top:12px;'>Goals needing review</h4>"
        + (
            "<ul style='margin:6px 0 0 18px;'>" + "".join(goal_preview_items) + "</ul>"
            if goal_preview_items
            else "<div class='muted'>No goals ready for action.</div>"
        )
        + "<a href='/admin?section=goals' class='button-link secondary' style='margin-top:6px;'>Open goals</a>"
    )
    overview_actions_card = (
        "<div class='card'>"
        "<h3>Action items</h3>"
        "<div class='muted'>Keep payouts, requests, and goals moving.</div>"
        f"{payout_section}{global_section}{requests_section}{goals_section}"
        "</div>"
    )

    def format_ratio(value: float) -> str:
        return f"{value * 100:.1f}%" if value else "0.0%"

    def risk_score(summary: Dict[str, Any]) -> int:
        market_ratio = summary.get("market_ratio", 0.0)
        cash_ratio = summary.get("cash_ratio", 0.0)
        largest_pct = summary.get("largest_position_pct", 0.0)
        holding_count = summary.get("holding_count", 0)
        score = 0
        if market_ratio > 0.65:
            score += 1
        if cash_ratio < 0.1 and market_ratio > 0.2:
            score += 1
        if largest_pct > 0.5:
            score += 1
        if holding_count <= 1 and market_ratio > 0.25:
            score += 1
        return score


    def risk_badge(summary: Dict[str, Any]) -> str:
        score = risk_score(summary)
        labels = ["Very low", "Low", "Balanced", "Elevated", "High"]
        colors = ["#0f766e", "#15803d", "#2563eb", "#ea580c", "#b91c1c"]
        idx = min(score, len(labels) - 1)
        return (
            f"<span class='pill' style='background:{colors[idx]}; color:#fff;'>"
            + labels[idx]
            + "</span>"
        )

    sorted_summaries = sorted(
        portfolio_summaries.values(),
        key=lambda entry: entry["total_assets_c"],
        reverse=True,
    )
    leader_rows = "".join(
        (
            "<tr>"
            + f"<td data-label='Kid'><b>{html_escape(summary['kid'].name)}</b><div class='muted'>{summary['kid'].kid_id}</div></td>"
            + f"<td data-label='Assets' class='right'>{usd(summary['total_assets_c'])}</td>"
            + f"<td data-label='Cash %' class='right'>{format_ratio(summary['cash_ratio'])}</td>"
            + f"<td data-label='Markets %' class='right'>{format_ratio(summary['market_ratio'])}</td>"
            + f"<td data-label='CD %' class='right'>{format_ratio(summary['cd_ratio'])}</td>"
            + f"<td data-label='Net P/L' class='right'>{fmt_signed(summary['total_pl_c'])}</td>"
            + "<td data-label='Risk'>"
            + risk_badge(summary)
            + f"<div class='muted'>{summary['holding_count']} holding{'s' if summary['holding_count'] != 1 else ''}</div>"
            + "</td>"
            + "</tr>"
        )
        for summary in sorted_summaries[:5]
    ) or "<tr><td colspan='7' class='muted'>No investing activity yet.</td></tr>"
    portfolio_preview_card = (
        "<div class='card'>"
        "<h3>Portfolio highlights</h3>"
        "<div class='muted'>Diversification and risk snapshot across all kids.</div>"
        f"<table><tr><th>Kid</th><th>Assets</th><th>Cash %</th><th>Markets %</th><th>CD %</th><th>Net P/L</th><th>Risk</th></tr>{leader_rows}</table>"
        "</div>"
    )
    overview_summary_card = (
        "<div class='card'>"
        "<h3>Command center</h3>"
        f"<div class='muted'>Realtime snapshot across {kids_count} kid{'s' if kids_count != 1 else ''}.</div>"
        f"{overview_quick_html}"
        "</div>"
    )
    overview_content = overview_summary_card + analytics_card + overview_actions_card + portfolio_preview_card
    children_overview_rows = "".join(
        (
            "<tr>"
            f"<td data-label='Child'><b>{html_escape(child.name)}</b><div class='muted'>{child.kid_id}</div></td>"
            f"<td data-label='Level'>L{child.level}</td>"
            f"<td data-label='Streak'>{child.streak_days} day{'s' if child.streak_days != 1 else ''}</td>"
            f"<td data-label='Balance' class='right'>{usd(child.balance_cents)}</td>"
            f"<td data-label='Actions' class='right'><a href='/admin?section=children&child={child.kid_id}' class='button-link secondary'>Expand</a></td>"
            "</tr>"
        )
        for child in kids
    ) or "<tr><td colspan='5' class='muted'>(no kids yet)</td></tr>"
    children_overview_card = (
        "<div class='card'>"
        "<h3>Children Overview</h3>"
        "<table><tr><th>Child</th><th>Level</th><th>Streak</th><th>Balance</th><th>Actions</th></tr>"
        f"{children_overview_rows}</table>"
        "</div>"
    )
    child_detail_card = ""
    if selected_child:
        child_obj = kids_by_id.get(selected_child)
        if child_obj:
            action_links: List[str] = [
                f"<a href='/admin/kiosk?kid_id={child_obj.kid_id}' class='button-link secondary'>Kiosk</a>",
                f"<a href='/admin/kiosk_full?kid_id={child_obj.kid_id}' class='button-link secondary'>Kiosk (auto)</a>",
            ]
            if admin_privs.can_manage_chores:
                action_links.append(
                    f"<a href='/admin/chores?kid_id={child_obj.kid_id}' class='button-link secondary'>Manage chores</a>"
                )
            if admin_privs.can_manage_allowance:
                action_links.append(
                    f"<a href='/admin/goals?kid_id={child_obj.kid_id}' class='button-link secondary'>Goals</a>"
                )
            action_links.append(
                f"<a href='/admin/statement?kid_id={child_obj.kid_id}' class='button-link secondary'>Statement</a>"
            )
            detail_forms: List[str] = []
            if admin_privs.can_create_accounts:
                detail_forms.append(
                    f"<form method='post' action='/admin/set_allowance' class='stacked-form'><input type='hidden' name='kid_id' value='{child_obj.kid_id}'><label>Allowance (dollars / week)</label><input name='allowance' type='text' data-money value='{dollars_value(child_obj.allowance_cents)}'><button type='submit'>Save Allowance</button></form>"
                )
                detail_forms.append(
                    f"<form method='post' action='/admin/set_kid_pin' class='stacked-form'><input type='hidden' name='kid_id' value='{child_obj.kid_id}'><label>Set kid PIN</label><input name='new_pin' placeholder='e.g. 4321'><button type='submit'>Set PIN</button></form>"
                )
            if admin_privs.can_delete_accounts:
                detail_forms.append(
                    f"<form method='post' action='/delete_kid' class='stacked-form' onsubmit='return confirm(\"Delete kid and all events?\");'><input type='hidden' name='kid_id' value='{child_obj.kid_id}'><label>Parent PIN (confirm)</label><input name='pin' placeholder='parent PIN'><button type='submit' class='danger'>Delete Kid</button></form>"
                )
            if detail_forms:
                forms_html = (
                    "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin-top:12px;'>"
                    + "".join(detail_forms)
                    + "</div>"
                )
            else:
                forms_html = (
                    "<div class='muted' style='margin-top:12px;'>No account actions available for this admin.</div>"
                )
            child_detail_card = (
                "<div class='card'>"
                f"<h3>{html_escape(child_obj.name)} — Details</h3>"
                f"<div class='muted'>{child_obj.kid_id} • Level {child_obj.level} • Streak {child_obj.streak_days} day{'s' if child_obj.streak_days != 1 else ''}</div>"
                f"<div style='margin-top:6px;'><b>Balance:</b> {usd(child_obj.balance_cents)}</div>"
                "<div class='actions' style='margin-top:12px; flex-wrap:wrap; gap:8px;'>"
                + "".join(action_links)
                + "</div>"
                + forms_html
                + "<p class='muted' style='margin-top:10px;'><a href='/admin?section=children'>← Back to overview</a></p>"
                + "</div>"
            )
    children_content = children_overview_card + child_detail_card
    filtered_admin_events = filter_events(
        events,
        search=admin_events_query,
        direction=admin_events_dir,
        kid_lookup=kids_by_id,
        kid_filter=admin_events_kid,
    )
    event_rows = "".join(
        (
            "<tr><td data-label='When'>"
            + event.timestamp.strftime("%Y-%m-%d %H:%M")
            + "</td><td data-label='Kid'>"
            + html_escape(event.child_id)
            + "</td><td data-label='Δ Amount' class='right'>"
            + ("+" if event.change_cents >= 0 else "")
            + usd(event.change_cents)
            + "</td><td data-label='Reason'>"
            + html_escape(event.reason)
            + "</td></tr>"
        )
        for event in filtered_admin_events
    ) or "<tr><td colspan='4' class='muted'>No events matched these filters.</td></tr>"
    if filtered_admin_events:
        if admin_events_query or admin_events_dir != "all" or admin_events_kid:
            events_summary_html = (
                f"<div class='muted' style='margin-bottom:8px;'>Showing {len(filtered_admin_events)} of {len(events)} recent events.</div>"
            )
        else:
            events_summary_html = (
                f"<div class='muted' style='margin-bottom:8px;'>Latest {len(filtered_admin_events)} events.</div>"
            )
    else:
        events_summary_html = (
            f"<div class='muted' style='margin-bottom:8px;'>No matches found across {len(events)} recent events.</div>"
        )
    events_search_value = html_escape(admin_events_query)
    events_dir_select = "".join(
        f"<option value='{value}'{' selected' if admin_events_dir == value else ''}>{label}</option>"
        for value, label in [
            ("all", "All activity"),
            ("credit", "Credits"),
            ("debit", "Debits"),
            ("zero", "Zero change"),
        ]
    )
    kid_option_bits = ["<option value=''>All kids</option>"]
    for kid in kids:
        selected_attr = " selected" if admin_events_kid.lower() == kid.kid_id.lower() else ""
        kid_option_bits.append(
            f"<option value='{kid.kid_id}'{selected_attr}>{html_escape(kid.name)} ({kid.kid_id})</option>"
        )
    events_kid_select = "".join(kid_option_bits)
    events_reset_html = (
        "<a href='/admin?section=events' class='button-link secondary' style='margin-top:6px;'>Reset</a>"
        if admin_events_query or admin_events_dir != "all" or admin_events_kid
        else ""
    )
    events_card = (
        "<div class='card'>"
        "<h3>Recent Events</h3>"
        "<form method='get' action='/admin' class='stacked-form' style='margin-bottom:12px;'>"
        "<input type='hidden' name='section' value='events'>"
        "<label>Search</label>"
        f"<input name='events_search' placeholder='Search reason, amount, or date' value='{events_search_value}'>"
        "<label>Type</label>"
        f"<select name='events_dir'>{events_dir_select}</select>"
        "<label>Kid</label>"
        f"<select name='events_kid'>{events_kid_select}</select>"
        "<button type='submit'>Apply</button>"
        f"{events_reset_html}"
        "</form>"
        f"{events_summary_html}"
        "<p class='muted'>Need a CSV? <a href='/admin/ledger.csv'>Download ledger</a></p>"
        f"<table><tr><th>When</th><th>Kid</th><th>Δ Amount</th><th>Reason</th></tr>{event_rows}</table>"
        "</div>"
    )
    create_kid_card = (
        "<div class='card' id='create-kid'>"
        "<h3>Create Kid</h3>"
        "<form method='post' action='/create_kid' class='stacked-form'>"
        "<label>kid_id</label><input name='kid_id' placeholder='alex01' required>"
        "<label>Name</label><input name='name' placeholder='Alex' required>"
        "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px;'>"
        "<div><label>Starting (dollars)</label><input name='starting' type='text' data-money value='0.00'></div>"
        "<div><label>Allowance (dollars / week)</label><input name='allowance' type='text' data-money value='0.00'></div>"
        "</div>"
        "<label>Set kid PIN (optional)</label><input name='kid_pin' placeholder='e.g. 4321'>"
        "<button type='submit'>Create Kid</button>"
        "</form>"
        "</div>"
    )
    credit_card = (
        "<div class='card'>"
        "<h3>Credit / Debit</h3>"
        "<form method='post' action='/adjust_balance' class='stacked-form'>"
        f"<label>kid_id</label><select name='kid_id' required>{kid_options_html}</select>"
        "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px;'>"
        "<div><label>Amount (dollars)</label><input name='amount' type='text' data-money value='1.00'></div>"
        "<div><label>Type</label><select name='kind'><option value='credit'>Credit (chore)</option><option value='debit'>Debit (redeem)</option></select></div>"
        "</div>"
        "<label>Reason</label><input name='reason' placeholder='chore / redeem'>"
        "<button type='submit'>Apply</button>"
        "</form>"
        "</div>"
    )
    transfer_card = (
        "<div class='card'>"
        "<h3>Family Transfer</h3>"
        "<form method='post' action='/admin/transfer' class='stacked-form'>"
        f"<label>From</label><select name='from_kid' required>{kid_options_html}</select>"
        f"<label>To</label><select name='to_kid' required>{kid_options_html}</select>"
        "<label>Amount (dollars)</label><input name='amount' type='text' data-money value='1.00'>"
        "<label>Note</label><input name='note' placeholder='optional note'>"
        "<button type='submit'>Transfer</button>"
        "</form>"
        "</div>"
    )
    account_cards: List[str] = []
    if admin_privs.can_create_accounts:
        account_cards.append(create_kid_card)
    if admin_privs.can_adjust_balances:
        account_cards.append(credit_card)
    if admin_privs.can_transfer_funds:
        account_cards.append(transfer_card)
    accounts_content = (
        "".join(account_cards)
        if account_cards
        else "<div class='card'><p class='muted'>No account tools available.</p></div>"
    )
    if admin_privs.can_manage_prizes:
        prize_rows = "".join(
            (
                f"<tr><td data-label='Prize'><b>{html_escape(prize.name)}</b><div class='muted'>{html_escape(prize.notes or '')}</div></td>"
                f"<td data-label='Cost' class='right'>{usd(prize.cost_cents)}</td>"
                f"<td data-label='Actions' class='right'><form method='post' action='/delete_prize' class='inline' onsubmit=\"return confirm('Delete this prize?');\"><input type='hidden' name='prize_id' value='{prize.id}'><button type='submit' class='danger'>Delete</button></form></td></tr>"
            )
            for prize in prizes
        ) or "<tr><td colspan='3' class='muted'>(no prizes yet)</td></tr>"
        prizes_card = (
            "<div class='card'>"
            "<h3>Prizes</h3>"
            "<form method='post' action='/add_prize' class='stacked-form'>"
            "<label>Name</label><input name='name' placeholder='Ice cream' required>"
            "<label>Cost (dollars)</label><input name='cost' type='text' data-money value='1.00'>"
            "<label>Notes</label><input name='notes' placeholder='One serving'>"
            "<button type='submit'>Add Prize</button>"
            "</form>"
            f"<table style='margin-top:10px;'><tr><th>Prize</th><th>Cost</th><th>Actions</th></tr>{prize_rows}</table>"
            "</div>"
        )
    else:
        prize_rows = "".join(
            (
                f"<tr><td data-label='Prize'><b>{html_escape(prize.name)}</b><div class='muted'>{html_escape(prize.notes or '')}</div></td>"
                f"<td data-label='Cost' class='right'>{usd(prize.cost_cents)}</td>"
            )
            for prize in prizes
        ) or "<tr><td colspan='2' class='muted'>(no prizes yet)</td></tr>"
        prizes_card = (
            "<div class='card'>"
            "<h3>Prizes</h3>"
            "<p class='muted'>Prize catalog is view-only for this admin.</p>"
            f"<table style='margin-top:10px;'><tr><th>Prize</th><th>Cost</th></tr>{prize_rows}</table>"
            "</div>"
        )
    cd_rates_pct = {code: cd_rates_bps.get(code, DEFAULT_CD_RATE_BPS) / 100 for code, _, _ in CD_TERM_OPTIONS}
    active_cd_total = sum(certificate_value_cents(cert, at=moment_admin) for cert in active_certs)
    active_cd_count = len(active_certs)
    ready_cd = sum(1 for cert in active_certs if moment_admin >= certificate_maturity_date(cert))
    rate_summary = " • ".join(f"{label}: {cd_rates_pct[code]:.2f}%" for code, label, _ in CD_TERM_OPTIONS)
    rate_field_blocks = []
    for idx, (code, label, _) in enumerate(CD_TERM_OPTIONS):
        label_style = " style='margin-top:6px;'" if idx else ""
        rate_field_blocks.append(
            f"        <label{label_style}>{label} rate (% APR)</label>\n"
            f"        <input name='rate_{code}' type='number' step='0.01' min='0' value='{cd_rates_pct[code]:.2f}' required>\n"
        )
    rate_fields_html = "".join(rate_field_blocks)
    penalty_summary = ", ".join(
        f"{label}: {cd_penalty_days.get(code, 0)} day{'s' if cd_penalty_days.get(code, 0) != 1 else ''}"
        for code, label, _ in CD_TERM_OPTIONS
    )
    penalty_field_blocks = []
    for idx, (code, label, _) in enumerate(CD_TERM_OPTIONS):
        label_style = " style='margin-top:6px;'" if idx else ""
        penalty_field_blocks.append(
            f"        <label{label_style}>{label} penalty (days of interest)</label>\n"
            f"        <input name='penalty_{code}' type='number' min='0' step='1' value='{cd_penalty_days.get(code, 0)}' required>\n"
        )
    penalty_fields_html = "".join(penalty_field_blocks)
    ready_note = (
        f"<div class='muted' style='margin-top:4px;'>{ready_cd} certificate{'s' if ready_cd != 1 else ''} ready to cash out.</div>"
        if ready_cd
        else "<div class='muted' style='margin-top:4px;'>Kids manage certificates from their investing page.</div>"
    )
    summary_rows = "".join(
        (
            "<tr>"
            + f"<td data-label='Kid'><b>{html_escape(summary['kid'].name)}</b><div class='muted'>{summary['kid'].kid_id}</div></td>"
            + f"<td data-label='Total' class='right'>{usd(summary['total_assets_c'])}</td>"
            + f"<td data-label='Cash' class='right'>{usd(summary['cash_cents'])}</td>"
            + f"<td data-label='Markets' class='right'>{usd(summary['total_market_c'])}</td>"
            + f"<td data-label='CDs' class='right'>{usd(summary['cd_value_c'])}</td>"
            + f"<td data-label='Unrealized' class='right'>{fmt_signed(summary['unrealized_pl_c'])}</td>"
            + f"<td data-label='Realized' class='right'>{fmt_signed(summary['realized_pl_c'])}</td>"
            + "<td data-label='Risk'>"
            + risk_badge(summary)
            + f"<div class='muted'>Largest position {format_ratio(summary['largest_position_pct'])}</div>"
            + "</td>"
            + "</tr>"
        )
        for summary in sorted_summaries
    ) or "<tr><td colspan='8' class='muted'>No investing data available.</td></tr>"
    investing_summary_card = (
        "<div class='card'>"
        "<h3>Investing analytics</h3>"
        "<div class='muted'>Monitor cash reserves, market exposure, and risk by kid.</div>"
        f"<div style='margin-top:6px;'>Active certificates: <b>{active_cd_count}</b> worth <b>{usd(active_cd_total)}</b></div>"
        f"<div>CD rates: <b>{rate_summary}</b></div>"
        f"<div>Early withdrawal penalties: <b>{penalty_summary}</b></div>"
        f"{ready_note}"
        f"<table style='margin-top:10px;'><tr><th>Kid</th><th>Total</th><th>Cash</th><th>Markets</th><th>CDs</th><th>Unrealized</th><th>Realized</th><th>Risk</th></tr>{summary_rows}</table>"
        "<p class='muted' style='margin-top:6px;'>Risk scores consider concentration, cash buffers, and diversification.</p>"
        "</div>"
    )
    analytics_points: List[str] = []
    if sorted_summaries:
        highest_risk_entry = max(sorted_summaries, key=risk_score)
        analytics_points.append(
            f"<li><b>{html_escape(highest_risk_entry['kid'].name)}</b> carries the highest risk {risk_badge(highest_risk_entry)} with {format_ratio(highest_risk_entry['largest_position_pct'])} in the largest position.</li>"
        )
        best_return_entry = max(sorted_summaries, key=lambda entry: entry.get("total_pl_c", 0))
        analytics_points.append(
            f"<li><b>{html_escape(best_return_entry['kid'].name)}</b> leads performance at {fmt_signed(best_return_entry['total_pl_c'])} net P/L.</li>"
        )
        strongest_cash = max(sorted_summaries, key=lambda entry: entry.get("cash_ratio", 0.0))
        analytics_points.append(
            f"<li><b>{html_escape(strongest_cash['kid'].name)}</b> maintains the largest cash buffer at {format_ratio(strongest_cash['cash_ratio'])}.</li>"
        )
    analytics_html = (
        "<ul class='muted' style='margin:8px 0 12px 18px; list-style:disc;'>" + "".join(analytics_points) + "</ul>"
    ) if analytics_points else "<p class='muted' style='margin-top:8px;'>No investing data available yet.</p>"
    if admin_privs.can_manage_investing:
        settings_forms = (
            "<h4>CD rate settings</h4>"
            "<form method='post' action='/admin/certificates/rate' style='margin-top:10px;'>"
            "  <p class='muted'>Annual percentage rates for newly opened certificates.</p>"
            f"{rate_fields_html}"
            "  <button type='submit' style='margin-top:8px;'>Save Rates</button>"
            "</form>"
            "<form method='post' action='/admin/certificates/penalty' style='margin-top:10px;'>"
            "  <p class='muted'>Days of interest forfeited when cashing out before maturity.</p>"
            f"{penalty_fields_html}"
            "  <button type='submit' style='margin-top:8px;'>Save Penalties</button>"
            "</form>"
        )
    else:
        settings_forms = (
            "<div class='muted' style='margin-top:12px;'>Investing settings are view-only for this admin.</div>"
        )
    investing_controls_card = (
        "<div class='card'>"
        "<h3>Portfolio tools &amp; controls</h3>"
        "<div class='muted'>Use these insights to gauge performance and adjust certificate settings.</div>"
        f"{analytics_html}"
        f"{settings_forms}"
        "</div>"
    )
    detail_cards: List[str] = []
    for summary in sorted_summaries:
        kid = summary["kid"]
        total_assets = summary["total_assets_c"]
        holdings_rows = "".join(
            (
                "<tr>"
                + f"<td data-label='Symbol'><b>{html_escape(holding['symbol'])}</b></td>"
                + f"<td data-label='Shares'>{holding['metrics']['shares']:.4f}</td>"
                + f"<td data-label='Price' class='right'>{usd(holding['metrics']['price_c'])}</td>"
                + f"<td data-label='Value' class='right'>{usd(holding['metrics']['market_value_c'])}</td>"
                + f"<td data-label='Invested' class='right'>{usd(holding['metrics']['invested_cost_c'])}</td>"
                + f"<td data-label='Unrealized' class='right'>{fmt_signed(holding['metrics']['unrealized_pl_c'])}</td>"
                + f"<td data-label='Realized' class='right'>{fmt_signed(holding['metrics']['realized_pl_c'])}</td>"
                + (
                    f"<td data-label='Return %' class='right'>{((holding['metrics']['unrealized_pl_c'] + holding['metrics']['realized_pl_c']) / holding['metrics']['invested_cost_c'] * 100):.2f}%</td>"
                    if holding["metrics"]["invested_cost_c"]
                    else "<td data-label='Return %' class='right'>—</td>"
                )
                + "</tr>"
            )
            for holding in summary["holdings"]
        ) or "<tr><td colspan='8' class='muted'>No tracked markets yet.</td></tr>"
        certificate_rows = "".join(
            (
                "<tr>"
                + f"<td data-label='Principal'>{usd(cert['principal_cents'])}</td>"
                + f"<td data-label='Value' class='right'>{usd(cert['value_cents'])}</td>"
                + f"<td data-label='Rate'>{cert['rate_bps'] / 100:.2f}%</td>"
                + f"<td data-label='Term'>{html_escape(cert['term_label'])}</td>"
                + f"<td data-label='Status'>{'Matured' if cert['matured'] else 'Growing'}</td>"
                + f"<td data-label='Matures' class='right'>{cert['matures_at']:%Y-%m-%d}</td>"
                + "</tr>"
            )
            for cert in summary["certificates"]
        ) or "<tr><td colspan='6' class='muted'>No active certificates.</td></tr>"
        cd_ready_note = ""
        if summary["cd_ready"]:
            ready_count = summary["cd_ready"]
            cd_ready_note = f" {ready_count} certificate{'s' if ready_count != 1 else ''} ready to cash out."
        detail_cards.append(
            "<div class='card'>"
            + f"<h3>{html_escape(kid.name)} — Portfolio</h3>"
            + f"<div class='muted'>Kid ID: {kid.kid_id}</div>"
            + f"<div style='margin-top:6px;'><b>Total assets:</b> {usd(total_assets)} • Cash {usd(summary['cash_cents'])} • Markets {usd(summary['total_market_c'])} • CDs {usd(summary['cd_value_c'])}</div>"
            + f"<div style='margin-top:6px;'>{risk_badge(summary)} <span class='muted' style='margin-left:8px;'>Cash {format_ratio(summary['cash_ratio'])} • Markets {format_ratio(summary['market_ratio'])} • CDs {format_ratio(summary['cd_ratio'])}</span></div>"
            + f"<p class='muted' style='margin-top:6px;'>Unrealized {fmt_signed(summary['unrealized_pl_c'])} • Realized {fmt_signed(summary['realized_pl_c'])} • Largest position {format_ratio(summary['largest_position_pct'])}.{cd_ready_note}</p>"
            + "<h4 style='margin-top:10px;'>Stock &amp; fund holdings</h4>"
            + f"<table><tr><th>Symbol</th><th>Shares</th><th>Price</th><th>Value</th><th>Invested</th><th>Unrealized</th><th>Realized</th><th>Return %</th></tr>{holdings_rows}</table>"
            + "<h4 style='margin-top:12px;'>Certificates of Deposit</h4>"
            + f"<table><tr><th>Principal</th><th>Value</th><th>Rate</th><th>Term</th><th>Status</th><th>Matures</th></tr>{certificate_rows}</table>"
            + "</div>"
        )
    investing_card = investing_summary_card + investing_controls_card + "".join(detail_cards)
    current_display = moment_admin
    mode_value = time_settings.get("mode", TIME_MODE_AUTO)
    offset_value = time_settings.get("offset", 0)
    manual_raw = time_settings.get("manual") or ""
    manual_display = ""
    if manual_raw:
        try:
            manual_dt = datetime.fromisoformat(manual_raw)
            manual_display = manual_dt.strftime("%Y-%m-%dT%H:%M")
        except ValueError:
            manual_display = manual_raw
    if admin_privs.can_manage_time:
        time_card = (
            "<div class='card'>"
            "<h3>Time Controls</h3>"
            f"<div class='muted'>Current app time: {current_display.strftime('%Y-%m-%d %H:%M:%S')}</div>"
            "<form method='post' action='/admin/time_settings' class='stacked-form'>"
            f"<label>Mode</label><select name='mode'><option value='{TIME_MODE_AUTO}' {'selected' if mode_value == TIME_MODE_AUTO else ''}>Auto (system clock)</option><option value='{TIME_MODE_MANUAL}' {'selected' if mode_value == TIME_MODE_MANUAL else ''}>Manual override</option></select>"
            f"<label>Offset minutes (auto mode)</label><input name='offset' type='number' step='1' value='{offset_value}'>"
            f"<label>Manual date &amp; time</label><input name='manual_datetime' type='datetime-local' value='{manual_display}'>"
            "<div class='muted'>Manual time only applies when manual mode is selected. Leave blank to keep the previous manual value.</div>"
            "<button type='submit'>Save Time Settings</button>"
            "</form>"
            "</div>"
        )
    else:
        time_card = (
            "<div class='card'>"
            "<h3>Time Controls</h3>"
            f"<div class='muted'>Current app time: {current_display.strftime('%Y-%m-%d %H:%M:%S')}</div>"
            "<p class='muted' style='margin-top:6px;'>You do not have permission to change time settings.</p>"
            "</div>"
        )
    if admin_privs.can_manage_allowance:
        rules_card = (
            "<div class='card'>"
            "<h3>Allowance Rules</h3>"
            "<form method='post' action='/admin/rules' class='stacked-form'>"
            f"<label><input type='checkbox' name='bonus_all' {'checked' if bonus_on_all else ''}> Bonus if all chores complete</label>"
            f"<input name='bonus' type='text' data-money value='{dollars_value(bonus_cents)}' placeholder='bonus $'>"
            f"<label><input type='checkbox' name='penalty_miss' {'checked' if penalty_on_miss else ''}> Penalty if chores missed</label>"
            f"<input name='penalty' type='text' data-money value='{dollars_value(penalty_cents)}' placeholder='penalty $'>"
            "<button type='submit'>Save Rules</button>"
            "</form>"
            "<p class='muted' style='margin-top:6px;'>Rules apply when weekly allowance runs (first admin view each Sunday).</p>"
            "</div>"
        )
    else:
        bonus_text = "Enabled" if bonus_on_all else "Disabled"
        penalty_text = "Enabled" if penalty_on_miss else "Disabled"
        rules_card = (
            "<div class='card'>"
            "<h3>Allowance Rules</h3>"
            f"<div>Bonus if all chores complete: <b>{bonus_text}</b> ({usd(bonus_cents)})</div>"
            f"<div style='margin-top:4px;'>Penalty if chores missed: <b>{penalty_text}</b> ({usd(penalty_cents)})</div>"
            "<p class='muted' style='margin-top:8px;'>Contact a full administrator to adjust these settings.</p>"
            "</div>"
        )
    permission_options = [
        ("perm_payouts", "can_manage_payouts", "Approve/deny payouts"),
        ("perm_chores", "can_manage_chores", "Manage chores"),
        ("perm_time", "can_manage_time", "Time controls"),
        ("perm_allowance", "can_manage_allowance", "Allowance rules & goals"),
        ("perm_prizes", "can_manage_prizes", "Manage prize catalog"),
        ("perm_create_accounts", "can_create_accounts", "Create kid accounts / pins"),
        ("perm_delete_accounts", "can_delete_accounts", "Delete kid accounts"),
        ("perm_adjust_balances", "can_adjust_balances", "Credit / debit balances"),
        ("perm_transfer", "can_transfer_funds", "Transfer between kids"),
        ("perm_create_admins", "can_create_admins", "Create admins"),
        ("perm_delete_admins", "can_delete_admins", "Delete admins"),
        ("perm_change_pins", "can_change_admin_pins", "Change admin PINs"),
        ("perm_investing", "can_manage_investing", "Manage investing controls"),
    ]
    admin_list_items: List[str] = []
    privilege_form_blocks: List[str] = []
    current_admin_role = (role or "").lower()
    can_manage_privileges = current_admin_role == "dad"
    for admin in parent_admins:
        role_key = admin["role"]
        label = html_escape(admin["label"])
        privileges = privileges_by_role.get(role_key, AdminPrivileges.default(role_key))
        if privileges.is_all_kids:
            kid_scope_text = "All kids"
        elif privileges.kid_ids:
            kid_labels = []
            for kid_id in privileges.kid_ids:
                kid_obj = all_kids_by_id.get(kid_id)
                if kid_obj:
                    kid_labels.append(f"{html_escape(kid_obj.name)} ({kid_id})")
                else:
                    kid_labels.append(html_escape(kid_id))
            kid_scope_text = ", ".join(kid_labels)
        else:
            kid_scope_text = "No kids selected"
        limit_bits: List[str] = []
        if privileges.max_credit_cents is not None:
            limit_bits.append(f"credit ≤ {usd(privileges.max_credit_cents)}")
        if privileges.max_debit_cents is not None:
            limit_bits.append(f"debit ≤ {usd(privileges.max_debit_cents)}")
        limits_summary = ", ".join(limit_bits) if limit_bits else "None"
        enabled_permissions = [
            label_text for _, attr, label_text in permission_options if getattr(privileges, attr)
        ]
        permission_summary = ", ".join(enabled_permissions) if enabled_permissions else "None"
        delete_button = ""
        if role_key not in DEFAULT_PARENT_ROLES and admin_privs.can_delete_admins:
            delete_button = (
                "<form method='post' action='/admin/delete_parent_admin' class='inline' "
                "style='margin-left:8px;'>"
                f"<input type='hidden' name='role' value='{role_key}'>"
                "<button type='submit' class='danger secondary'>Delete</button>"
                "</form>"
            )
        admin_list_items.append(
            "<li>"
            + f"{label} <span class='muted'>({role_key})</span>"
            + (" <span class='pill'>Default</span>" if role_key in DEFAULT_PARENT_ROLES else "")
            + delete_button
            + "<div class='muted' style='margin-top:4px;'>Kid access: "
            + kid_scope_text
            + "</div>"
            + "<div class='muted'>Limits: "
            + limits_summary
            + "</div>"
            + "<div class='muted'>Permissions: "
            + permission_summary
            + "</div>"
            + "</li>"
        )
        if can_manage_privileges and role_key not in DEFAULT_PARENT_ROLES:
            checked_ids = {kid.lower() for kid in privileges.kid_ids}
            kid_checkboxes = "".join(
                f"<label class='checkbox'><input type='checkbox' name='kid_ids' value='{kid.kid_id}'"
                + (" checked" if kid.kid_id.lower() in checked_ids else "")
                + f"> {html_escape(kid.name)} ({kid.kid_id})</label>"
                for kid in all_kids
            ) or "<div class='muted'>No kids available yet.</div>"
            max_credit_value = html_escape(dollars_value(privileges.max_credit_cents) if privileges.max_credit_cents is not None else "")
            max_debit_value = html_escape(dollars_value(privileges.max_debit_cents) if privileges.max_debit_cents is not None else "")
            permission_checkboxes = "".join(
                "<label class='checkbox'><input type='checkbox' name='"
                + field
                + "' value='1'"
                + (" checked" if getattr(privileges, attr) else "")
                + f"> {label_text}</label>"
                for field, attr, label_text in permission_options
            )
            privilege_form_blocks.append(
                "<form method='post' action='/admin/update_privileges' class='stacked-form' style='margin-top:12px;'>"
                + f"<h4>{label} — Privileges</h4>"
                + f"<input type='hidden' name='role' value='{role_key}'>"
                + "<label>Kid access</label>"
                + f"<select name='kid_scope'><option value='all'{' selected' if privileges.is_all_kids else ''}>All kids</option><option value='custom'{' selected' if not privileges.is_all_kids else ''}>Choose specific kids</option></select>"
                + "<div class='muted' style='margin-top:4px;'>Global chores are always visible to all admins.</div>"
                + "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:6px; margin-top:6px;'>"
                + kid_checkboxes
                + "</div>"
                + "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px; margin-top:10px;'>"
                + f"<div><label>Max credit ($)</label><input name='max_credit' type='text' data-money value='{max_credit_value}' placeholder='no limit'></div>"
                + f"<div><label>Max debit ($)</label><input name='max_debit' type='text' data-money value='{max_debit_value}' placeholder='no limit'></div>"
                + "</div>"
                + "<div class='muted' style='margin-top:4px;'>Leave limits blank for unlimited approvals.</div>"
                + "<label style='margin-top:10px;'>Permissions</label>"
                + "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:6px;'>"
                + permission_checkboxes
                + "</div>"
                + "<button type='submit' style='margin-top:10px;'>Save Privileges</button>"
                + "</form>"
            )
    admin_list_html = "".join(admin_list_items) or "<li class='muted'>(no admins yet)</li>"
    privilege_forms_html = "".join(privilege_form_blocks)
    if not can_manage_privileges:
        privilege_note = "<p class='muted' style='margin-top:12px;'>Only the default Dad administrator can change privileges.</p>"
    elif not privilege_form_blocks:
        privilege_note = "<p class='muted' style='margin-top:12px;'>Add another admin to configure custom privileges.</p>"
    else:
        privilege_note = ""
    pin_form_html = ""
    if admin_privs.can_change_admin_pins:
        pin_form_html = (
            "<form method='post' action='/admin/set_parent_pin' class='stacked-form' style='margin-top:12px;'>"
            "<h4>Update PIN</h4>"
            f"<select name='role'>{parent_options_html}</select>"
            "<label>New PIN</label><input name='new_pin' type='password' placeholder='****' autocomplete='new-password' required>"
            "<label>Confirm PIN</label><input name='confirm_pin' type='password' placeholder='****' autocomplete='new-password' required>"
            "<button type='submit'>Set PIN</button>"
            "</form>"
        )
    elif parent_admins:
        pin_form_html = "<p class='muted' style='margin-top:12px;'>You do not have permission to change admin PINs.</p>"
    add_admin_form_html = ""
    if admin_privs.can_create_admins:
        add_admin_form_html = (
            "<form method='post' action='/admin/add_parent_admin' class='stacked-form' style='margin-top:12px;'>"
            "<h4>Add another admin</h4>"
            "<label>Name</label><input name='label' placeholder='Grandma' required>"
            "<label>PIN</label><input name='pin' type='password' placeholder='****' required>"
            "<label>Confirm PIN</label><input name='confirm_pin' type='password' placeholder='****' required>"
            "<button type='submit'>Add Admin</button>"
            "</form>"
        )
    else:
        add_admin_form_html = "<p class='muted' style='margin-top:12px;'>You do not have permission to add new admins.</p>"
    parent_admins_card = (
        "<div class='card'>"
        "<h3>Parent Admins</h3>"
        f"<ul class='admin-list'>{admin_list_html}</ul>"
        + privilege_note
        + privilege_forms_html
        + pin_form_html
        + add_admin_form_html
        + "</div>"
    )

    def _market_status_badge(listing: MarketplaceListing) -> str:
        label, bg, fg = status_styles_market.get(
            listing.status, (listing.status.title(), "#e2e8f0", "#334155")
        )
        return f"<span class='pill' style='background:{bg}; color:{fg};'>{label}</span>"

    marketplace_rows: List[str] = []
    for listing in marketplace_listings[:60]:
        owner = all_kids_by_id.get(listing.owner_kid_id)
        owner_name = (
            html_escape(owner.name)
            if owner
            else html_escape(listing.owner_kid_id)
        )
        claimer = all_kids_by_id.get(listing.claimed_by) if listing.claimed_by else None
        claimer_name = (
            html_escape(claimer.name)
            if claimer
            else (html_escape(listing.claimed_by) if listing.claimed_by else "")
        )
        status_html = _market_status_badge(listing)
        if listing.status == MARKETPLACE_STATUS_CLAIMED and claimer_name:
            status_html += f"<div class='muted'>By {claimer_name}</div><div class='muted'>{_format_market_ts(listing.claimed_at)}</div>"
        elif listing.status == MARKETPLACE_STATUS_SUBMITTED:
            status_html += f"<div class='muted'>Submitted {_format_market_ts(listing.submitted_at)}</div>"
            if claimer_name:
                status_html += f"<div class='muted'>By {claimer_name}</div>"

        elif listing.status == MARKETPLACE_STATUS_COMPLETED:
            status_html += f"<div class='muted'>Completed {_format_market_ts(listing.completed_at)}</div>"
        elif listing.status == MARKETPLACE_STATUS_CANCELLED:
            status_html += f"<div class='muted'>Cancelled {_format_market_ts(listing.cancelled_at)}</div>"
        elif listing.status == MARKETPLACE_STATUS_REJECTED:
            status_html += f"<div class='muted'>Rejected {_format_market_ts(listing.completed_at)}</div>"
            if listing.payout_note:
                status_html += f"<div class='muted'>{html_escape(listing.payout_note)}</div>"
        total_value_c = listing.final_payout_cents or (
            listing.offer_cents + listing.chore_award_cents
        )
        total_value = usd(total_value_c)

        marketplace_rows.append(
            "<tr>"
            f"<td data-label='Created'>{_format_market_ts(listing.created_at)}</td>"
            f"<td data-label='Owner'><b>{owner_name}</b><div class='muted'>{html_escape(listing.chore_name)}</div></td>"
            f"<td data-label='Offer' class='right'>{usd(listing.offer_cents)}<div class='muted'>Award {usd(listing.chore_award_cents)}</div></td>"
            f"<td data-label='Total' class='right'>{total_value}</td>"
            f"<td data-label='Status'>{status_html}</td>"
            "</tr>"
        )
    marketplace_table = (
        "".join(marketplace_rows)
        or "<tr><td colspan='5' class='muted'>No marketplace activity recorded yet.</td></tr>"
    )
    marketplace_summary = (
        f"Open {len(marketplace_open)} • Claimed {len(marketplace_claimed)}"
        f" • Submitted {len(marketplace_submitted)}"
        f" • Completed {len(marketplace_completed)} • Cancelled {len(marketplace_cancelled)}"
        f" • Rejected {len(marketplace_rejected)}"

    )
    marketplace_card = (
        "<div class='card'>"
        "<h3>Chore Marketplace</h3>"
        f"<div class='muted'>{marketplace_summary}</div>"
        f"<div class='muted' style='margin-bottom:8px;'>Escrow {usd(escrow_total_c)} • Lifetime payouts {usd(payout_total_c)}</div>"
        f"<table><tr><th>Created</th><th>Owner</th><th>Offer</th><th>Total</th><th>Status</th></tr>{marketplace_table}</table>"
        "</div>"
    )
    weekday_selector = "".join(
        f"<label style='margin-right:6px;'><input type='checkbox' name='weekdays' value='{day}'> {label}</label>"
        for day, label in WEEKDAY_OPTIONS
    )
    chores_card = (
        "<div class='card'>"
        "<h3>Add a Chore</h3>"
        "<form method='post' action='/admin/chores/create' class='stacked-form'>"
        f"<label>kid_id</label><select name='kid_id' required>{kid_options_html}<option value='{GLOBAL_CHORE_KID_ID}'>Global (Free-for-all)</option></select>"
        "<label>Name</label><input name='name' placeholder='Take out trash' required>"
        "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:8px;'>"
        "<div><label>Type</label><select name='type'><option value='daily'>Daily</option><option value='weekly'>Weekly</option><option value='monthly'>Monthly</option><option value='special'>Special</option></select></div>"
        "<div><label>Award (dollars)</label><input name='award' type='text' data-money value='0.50'></div>"
        "<div><label>Max claimants (global)</label><input name='max_claimants' type='number' min='1' value='1'></div>"
        "</div>"
        "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:8px;'>"
        "<div><label>Start Date (optional)</label><input name='start_date' type='date'></div>"
        "<div><label>End Date (optional)</label><input name='end_date' type='date'></div>"
        "</div>"
        "<div style='margin-top:6px;'><div class='muted' style='margin-bottom:4px;'>Weekdays (optional)</div><div>"
        f"{weekday_selector}"
        "</div></div>"
        "<label>Specific dates (comma separated)</label><input name='specific_dates' placeholder='YYYY-MM-DD,YYYY-MM-DD'>"
        "<label>Notes</label><input name='notes' placeholder='Any details'>"
        "<p class='muted'>Global chores appear for all kids under “Free-for-all”. Use max claimants to set how many kids can share the reward per period.</p>"
        "<button type='submit'>Add Chore</button>"
        "</form>"
        "</div>"
    )
    sections: List[Tuple[str, str, str, str]] = [
        ("overview", "Command center", overview_content, ""),
        ("goals", "Goals needing action", goals_card, ""),
        ("payouts", "Pending payouts", pending_card, "".join(multi_modals)),
        ("children", "Children overview", children_content, ""),
        ("events", "Recent events", events_card, ""),
        ("accounts", "Account tools", accounts_content, ""),
        ("investing", "Portfolios & analytics", investing_card, ""),
        ("chores", "Chore publishing", chores_card, ""),
        ("marketplace", "Marketplace", marketplace_card, ""),
        ("prizes", "Prizes", prizes_card, ""),
        ("rules", "Allowance rules", rules_card, ""),
        ("time", "Time controls", time_card, ""),
        ("admins", "Parent admins", parent_admins_card, ""),
    ]
    sections_map = {key: {"label": label, "content": content, "extra": extra} for key, label, content, extra in sections}
    if selected_section not in sections_map:
        selected_section = "overview"
    sidebar_links = "".join(
        (
            f"<a href='/admin?section={key}' class='{ 'active' if key == selected_section else ''}'>{html_escape(cfg['label'])}</a>"
        )
        for key, cfg in sections_map.items()
    )
    selected_content = sections_map[selected_section]["content"]
    if notice_html:
        selected_content = notice_html + selected_content
    extra_html = sections_map[selected_section].get("extra", "")
    admin_pref_controls = preference_controls_html(request)
    inner = (
        "<div class='topbar'><h3>Admin Portal</h3><div style='display:flex; flex-direction:column; gap:6px; align-items:flex-end;'>"
        + "<div>"
        + _role_badge(role)
        + "<form method='post' action='/admin/logout' style='display:inline-block; margin-left:8px;'><button type='submit' class='pill'>Logout</button></form>"
        + "</div>"
        + admin_pref_controls
        + "</div></div>"
        + "<div class='layout'><nav class='sidebar'>"
        + sidebar_links
        + "</nav><div class='content'>"
        + selected_content
        + "</div></div>"
    )
    return render_page(request, "Admin", inner + extra_html)
@app.get("/admin/kiosk", response_class=HTMLResponse)
def admin_kiosk(request: Request, kid_id: str = Query(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return render_page(request, "Kiosk", "<div class='card'>Kid not found.</div>")
        chores = list_chore_instances_for_kid(kid_id)
        events = session.exec(
            select(Event)
            .where(Event.child_id == kid_id)
            .order_by(desc(Event.timestamp))
            .limit(10)
        ).all()
    event_rows = "".join(
        f"<tr><td data-label='When'>{event.timestamp.strftime('%b %d, %I:%M %p')}</td>"
        f"<td data-label='Δ Amount' class='right'>{'+' if event.change_cents>=0 else ''}{usd(event.change_cents)}</td>"
        f"<td data-label='Reason'>{event.reason}</td></tr>"
        for event in events
    ) or "<tr><td>(no events)</td></tr>"
    chore_cards = "".join(
        f"<div class='card'><b>{chore.name}</b> <span class='muted'>({chore.type})</span> "
        f"<span class='pill' style='margin-left:8px;'>{(inst.status if inst else 'available').title()}</span> "
        f"<span class='muted' style='margin-left:8px;'>+{usd(chore.award_cents)}</span></div>"
        for chore, inst in chores
    ) or "<div class='muted'>(no chores)</div>"
    inner = f"""
    <div class='card kiosk'>
      <div><div class='name'>{child.name}</div><div class='muted'>{child.kid_id} • L{child.level} • Streak {child.streak_days}</div></div>
      <div class='balance'>{usd(child.balance_cents)}</div>
    </div>
    <div class='grid'>
      <div class='card'><h3>Chores</h3>{chore_cards}</div>
      <div class='card'><h3>Recent Activity</h3><table><tr><th>When</th><th>Δ Amount</th><th>Reason</th></tr>{event_rows}</table></div>
    </div>
    """
    return render_page(request, f"Kiosk — {child.name}", inner)


@app.get("/admin/kiosk_full", response_class=HTMLResponse)
def admin_kiosk_full(request: Request, kid_id: str = Query(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return render_page(request, "Kiosk", "<div class='card'>Kid not found.</div>")
        chores = list_chore_instances_for_kid(kid_id)
    head = "<meta http-equiv='refresh' content='10'>"
    chore_cards = "".join(
        f"<div class='card'><b>{chore.name}</b> <span class='muted'>({chore.type})</span>"
        f" <span class='pill' style='margin-left:8px;'>{(inst.status if inst else 'available').title()}</span></div>"
        for chore, inst in chores
    ) or "<div class='muted'>(no chores)</div>"
    inner = f"""
    <div class='card kiosk'>
      <div><div class='name'>{child.name} <span class='muted'>{child.kid_id}</span></div></div>
      <div class='balance'>{usd(child.balance_cents)}</div>
    </div>
    <div class='card'><h3>Chores</h3>{chore_cards}<p class='muted' style='margin-top:6px;'>Auto-refresh every 10 seconds.</p></div>
    """
    return render_page(request, f"Kiosk — {child.name}", inner, head_extra=head)

@app.get("/admin/chores", response_class=HTMLResponse)
def admin_manage_chores(request: Request, kid_id: str = Query(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_chores", redirect="/admin?section=chores"
        )
    ) is not None:
        return redirect
    if kid_id != GLOBAL_CHORE_KID_ID:
        if (
            denied := ensure_admin_kid_access(
                request, kid_id, redirect="/admin?section=chores"
            )
        ) is not None:
            return denied
    is_global = kid_id == GLOBAL_CHORE_KID_ID
    with Session(engine) as session:
        pending_claim_rows: List[Tuple[GlobalChoreClaim, Child, Chore]] = []
        recent_claim_rows: List[Tuple[GlobalChoreClaim, Child, Chore]] = []
        approved_lookup: Dict[Tuple[int, str], List[GlobalChoreClaim]] = {}
        if is_global:
            child = None
            chores = session.exec(
                select(Chore)
                .where(Chore.kid_id == GLOBAL_CHORE_KID_ID)
                .order_by(desc(Chore.created_at))
            ).all()
            if chores:
                approved_rows = session.exec(
                    select(GlobalChoreClaim)
                    .where(GlobalChoreClaim.chore_id.in_([ch.id for ch in chores]))
                    .where(GlobalChoreClaim.status == GLOBAL_CHORE_STATUS_APPROVED)
                ).all()
                for claim in approved_rows:
                    approved_lookup.setdefault((claim.chore_id, claim.period_key), []).append(claim)
            pending_claim_rows = session.exec(
                select(GlobalChoreClaim, Child, Chore)
                .where(Chore.id == GlobalChoreClaim.chore_id)
                .where(Chore.kid_id == GLOBAL_CHORE_KID_ID)
                .where(Child.kid_id == GlobalChoreClaim.kid_id)
                .where(GlobalChoreClaim.status == GLOBAL_CHORE_STATUS_PENDING)
                .order_by(GlobalChoreClaim.period_key, GlobalChoreClaim.submitted_at)
            ).all()
            recent_claim_rows = session.exec(
                select(GlobalChoreClaim, Child, Chore)
                .where(Chore.id == GlobalChoreClaim.chore_id)
                .where(Chore.kid_id == GLOBAL_CHORE_KID_ID)
                .where(Child.kid_id == GlobalChoreClaim.kid_id)
                .where(GlobalChoreClaim.status != GLOBAL_CHORE_STATUS_PENDING)
                .order_by(desc(GlobalChoreClaim.submitted_at))
                .limit(20)
            ).all()
        else:
            child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
            if not child:
                return render_page(request, "Chores", "<div class='card'>Kid not found.</div>")
            chores = session.exec(
                select(Chore)
                .where(Chore.kid_id == kid_id)
                .order_by(desc(Chore.created_at))
            ).all()
    rows_parts: List[str] = []
    for chore in chores:
        form_id = f"chore-form-{chore.id}"
        selected_weekdays = chore_weekdays(chore)
        weekday_controls = "".join(
            f"<label class='chore-schedule__weekday'><input type='checkbox' name='weekdays' value='{day}'{' checked' if day in selected_weekdays else ''} form='{form_id}'> {label}</label>"
            for day, label in WEEKDAY_OPTIONS
        )
        specific_value = html_escape(chore.specific_dates or "")
        start_value = chore.start_date.isoformat() if chore.start_date else ""
        end_value = chore.end_date.isoformat() if chore.end_date else ""
        name_value = html_escape(chore.name)
        notes_value = html_escape(chore.notes or "")
        schedule_html = (
            "<div class='chore-schedule'>"
            "<div class='chore-schedule__dates'>"
            f"<input name='start_date' type='date' value='{start_value}' form='{form_id}'>"
            f"<input name='end_date' type='date' value='{end_value}' form='{form_id}'>"
            "</div>"
            f"<div class='chore-schedule__weekdays'>{weekday_controls}</div>"
            f"<input name='specific_dates' placeholder='YYYY-MM-DD,YYYY-MM-DD' value='{specific_value}' form='{form_id}'>"
            "</div>"
        )
        is_global_chore = chore.kid_id == GLOBAL_CHORE_KID_ID
        chore_type = normalize_chore_type(chore.type, is_global=is_global_chore)
        if is_global_chore:
            type_options = ["daily", "weekly", "monthly"]
        else:
            type_options = ["daily", "weekly", "monthly", "special"]
        type_select = "".join(
            f"<option value='{opt}' {'selected' if chore_type == opt else ''}>{opt}</option>"
            for opt in type_options
        )
        action_items = [f"<button type='submit' form='{form_id}'>Save</button>"]
        if not is_global_chore:
            action_items.append(
                "<form method='post' action='/admin/chore_make_available_now' class='chore-row__action-form'>"
                f"<input type='hidden' name='chore_id' value='{chore.id}'>"
                "<button type='submit'>Make Available Now</button>"
                "</form>"
            )
        action_items.append(
            (
                "<form method='post' action='/admin/chores/deactivate' class='chore-row__action-form'>"
                f"<input type='hidden' name='chore_id' value='{chore.id}'><button type='submit' class='danger'>Deactivate</button></form>"
            )
            if chore.active
            else
            (
                "<form method='post' action='/admin/chores/activate' class='chore-row__action-form'>"
                f"<input type='hidden' name='chore_id' value='{chore.id}'><button type='submit'>Activate</button></form>"
            )
        )
        action_html = "<div class='chore-row__actions'>" + "".join(action_items) + "</div>"
        rows_parts.append(
            "<tr>"
            f"<td data-label='Name'><form id='{form_id}' method='post' action='/admin/chores/update'></form>"
            f"<input type='hidden' name='chore_id' value='{chore.id}' form='{form_id}'>"
            f"<input name='name' value='{name_value}' form='{form_id}'></td>"
            f"<td data-label='Type'><select name='type' form='{form_id}'>{type_select}</select></td>"
            f"<td data-label='Award ($)' class='right'><input name='award' type='text' data-money value='{dollars_value(chore.award_cents)}' form='{form_id}' class='chore-field--compact'></td>"
            f"<td data-label='Max Spots'><input name='max_claimants' type='number' min='1' value='{max(1, chore.max_claimants)}' form='{form_id}' class='chore-field--compact'></td>"
            f"<td data-label='Schedule'>{schedule_html}</td>"
            f"<td data-label='Notes'><input name='notes' value='{notes_value}' form='{form_id}'></td>"
            f"<td data-label='Status'><span class='pill'>{'Active' if chore.active else 'Inactive'}</span></td>"
            f"<td data-label='Actions' class='right'>{action_html}</td>"
            "</tr>"
        )
    rows = "".join(rows_parts) or "<tr><td colspan='8' class='muted'>(no chores yet)</td></tr>"
    if is_global:
        heading = "Manage Global Chores"
        badge = f"<span class='pill' style='margin-left:8px;'>{GLOBAL_CHORE_KID_ID}</span>"
        note_html = "<p class='muted' style='margin-top:6px;'>Global chores appear for all kids under “Free-for-all”. Use the controls below to approve or reject submissions.</p>"
    else:
        heading = f"Manage Chores — {child.name}"
        badge = f"<span class='pill' style='margin-left:8px;'>{child.kid_id}</span>"
        note_html = "<p class='muted' style='margin-top:6px;'>“Make Available Now” republishes the chore for the current period (within its active window).</p>"
    chores_table = f"""
    <div class='card'>
      <table class='chore-table'>
        <tr><th>Name</th><th>Type</th><th>Award ($)</th><th>Max Spots</th><th>Schedule</th><th>Notes</th><th>Status</th><th>Actions</th></tr>
        {rows}
      </table>
      {note_html}
    </div>
    """
    pending_html = ""
    history_html = ""
    if is_global:
        pending_groups: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for claim, claimant, chore in pending_claim_rows:
            key = (claim.chore_id, claim.period_key)
            entry = pending_groups.setdefault(key, {"chore": chore, "claims": []})
            entry["claims"].append((claim, claimant))
        if pending_groups:
            group_blocks: List[str] = []
            for (chore_id_val, period_key), data in pending_groups.items():
                chore = data["chore"]
                claims = data["claims"]
                approved_list = approved_lookup.get((chore_id_val, period_key), [])
                approved_total = sum(cl.award_cents for cl in approved_list)
                remaining_award = max(0, chore.award_cents - approved_total)
                remaining_slots = max(0, chore.max_claimants - len(approved_list))
                table_rows = "".join(
                    "<tr>"
                    f"<td data-label='Select'><input type='checkbox' name='claim_ids' value='{claim.id}'></td>"
                    f"<td data-label='Kid'><b>{html_escape(claimant.name)}</b><div class='muted'>{claimant.kid_id}</div></td>"
                    f"<td data-label='Submitted'>{claim.submitted_at.strftime('%Y-%m-%d %H:%M')}</td>"
                    f"<td data-label='Override ($)' class='right'><input name='amount_{claim.id}' type='text' data-money placeholder='optional'></td>"
                    "</tr>"
                    for claim, claimant in claims
                )
                table_rows = table_rows or "<tr><td colspan='4' class='muted'>(no pending claims)</td></tr>"
                group_blocks.append(
                    "<div style='margin-top:12px; padding:12px; border-radius:12px; border:1px solid #1f2937;'>"
                    f"<div style='font-weight:600;'>{html_escape(chore.name)} — {period_key}</div>"
                    f"<div class='muted' style='margin-top:4px;'>Max {chore.max_claimants} kids • Approved {len(approved_list)} • Slots left {remaining_slots} • Remaining award {usd(remaining_award)}</div>"
                    f"<form method='post' action='/admin/global_chore/claims' style='margin-top:8px;'>"
                    f"<input type='hidden' name='chore_id' value='{chore_id_val}'>"
                    f"<input type='hidden' name='period_key' value='{period_key}'>"
                    "<input type='hidden' name='redirect' value='/admin?section=payouts'>"
                    f"<table><tr><th>Select</th><th>Kid</th><th>Submitted</th><th>Override ($)</th></tr>{table_rows}</table>"
                    f"<div style='display:flex; gap:8px; flex-wrap:wrap; margin-top:8px;'>"
                    f"<button type='submit' name='decision' value='approve'>Approve Selected</button>"
                    f"<button type='submit' name='decision' value='reject' class='danger'>Reject Selected</button>"
                    "</div>"
                    "</form>"
                    "</div>"
                )
            pending_html = """
    <div class='card'>
      <h3>Pending Free-for-all Claims</h3>
      <div class='muted'>Select kids to approve or reject. If no override is provided, rewards are split evenly among approved kids.</div>
      {blocks}
    </div>
            """.format(blocks="".join(group_blocks))
        else:
            pending_html = """
    <div class='card'>
      <h3>Pending Free-for-all Claims</h3>
      <p class='muted'>No pending submissions right now.</p>
    </div>
            """
        history_rows = "".join(
            "<tr>"
            f"<td data-label='When'>{(claim.approved_at or claim.submitted_at).strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td data-label='Chore'><b>{html_escape(chore.name)}</b></td>"
            f"<td data-label='Kid'><b>{html_escape(child.name)}</b><div class='muted'>{child.kid_id}</div></td>"
            f"<td data-label='Period'>{claim.period_key}</td>"
            f"<td data-label='Result'>{claim.status.title()}</td>"
            f"<td data-label='Award' class='right'>{usd(claim.award_cents)}</td>"
            "</tr>"
            for claim, child, chore in recent_claim_rows
        ) or "<tr><td colspan='6' class='muted'>(no recent activity)</td></tr>"
        history_html = f"""
    <div class='card'>
      <h3>Recent Free-for-all Decisions</h3>
      <table><tr><th>When</th><th>Chore</th><th>Kid</th><th>Period</th><th>Result</th><th>Award</th></tr>{history_rows}</table>
    </div>
    """
    topbar = f"""
    <div class='topbar'><h3>{heading} {badge}</h3>
      <a href='/admin'><button>Back</button></a>
    </div>
    """
    inner = f"""
    {topbar}
    {chores_table}
    {pending_html}
    {history_html}
    """
    return render_page(request, "Manage Chores", inner)


@app.post("/admin/chores/create")
def admin_chore_create(
    request: Request,
    kid_id: str = Form(...),
    name: str = Form(...),
    type: str = Form(...),
    award: str = Form(...),
    max_claimants: str = Form("1"),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    notes: str = Form(""),
    weekdays: List[str] = Form([]),
    specific_dates: str = Form(""),
):
    if (
        redirect := require_admin_permission(
            request, "can_manage_chores", redirect="/admin?section=chores"
        )
    ) is not None:
        return redirect
    award_c = to_cents_from_dollars_str(award, 0)
    try:
        max_claims_value = int((max_claimants or "1").strip())
    except ValueError:
        max_claims_value = 1
    max_claims_value = max(1, max_claims_value)
    weekday_csv = serialize_weekday_selection(weekdays) if weekdays else None
    dates_csv = serialize_specific_dates(specific_dates) if specific_dates else None
    kid_value = (kid_id or "").strip()
    if kid_value and kid_value != GLOBAL_CHORE_KID_ID:
        if (
            denied := ensure_admin_kid_access(
                request, kid_value, redirect="/admin?section=chores"
            )
        ) is not None:
            return denied
    normalized_type = normalize_chore_type(type, is_global=kid_value == GLOBAL_CHORE_KID_ID)
    kid_label = "Free-for-all" if kid_value == GLOBAL_CHORE_KID_ID else kid_value or "Unknown"
    with Session(engine) as session:
        if kid_value and kid_value != GLOBAL_CHORE_KID_ID:
            target_child = session.exec(select(Child).where(Child.kid_id == kid_value)).first()
            if target_child:
                kid_label = target_child.name
        chore = Chore(
            kid_id=kid_value,
            name=name.strip(),
            type=normalized_type,
            award_cents=award_c,
            notes=notes.strip() or None,
            start_date=date.fromisoformat(start_date) if start_date else None,
            end_date=date.fromisoformat(end_date) if end_date else None,
            max_claimants=max_claims_value,
            weekdays=weekday_csv,
            specific_dates=dates_csv,
        )
        session.add(chore)
        session.commit()
    set_admin_notice(
        request,
        f"Added chore '{name.strip()}' for {html_escape(kid_label)}.",
        "success",
    )
    return RedirectResponse("/admin?section=chores", status_code=302)


@app.post("/admin/chores/update")
def admin_chore_update(
    request: Request,
    chore_id: int = Form(...),
    name: str = Form(...),
    type: str = Form(...),
    award: str = Form(...),
    max_claimants: str = Form("1"),
    start_date: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    notes: str = Form(""),
    weekdays: List[str] = Form([]),
    specific_dates: str = Form(""),
):
    if (
        redirect := require_admin_permission(
            request, "can_manage_chores", redirect="/admin?section=chores"
        )
    ) is not None:
        return redirect
    award_c = to_cents_from_dollars_str(award, 0)
    try:
        max_claims_value = int((max_claimants or "1").strip())
    except ValueError:
        max_claims_value = 1
    max_claims_value = max(1, max_claims_value)
    weekday_csv = serialize_weekday_selection(weekdays) if weekdays else None
    dates_csv = serialize_specific_dates(specific_dates) if specific_dates else None
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return RedirectResponse("/admin", status_code=302)
        if chore.kid_id and chore.kid_id != GLOBAL_CHORE_KID_ID:
            if (
                denied := ensure_admin_kid_access(
                    request, chore.kid_id, redirect="/admin?section=chores"
                )
            ) is not None:
                return denied
        normalized_type = normalize_chore_type(type, is_global=chore.kid_id == GLOBAL_CHORE_KID_ID)
        target_kid = chore.kid_id
        chore.name = name.strip()
        chore.type = normalized_type
        chore.award_cents = award_c
        chore.notes = notes.strip() or None
        chore.start_date = date.fromisoformat(start_date) if start_date else None
        chore.end_date = date.fromisoformat(end_date) if end_date else None
        chore.max_claimants = max_claims_value
        chore.weekdays = weekday_csv
        chore.specific_dates = dates_csv
        session.add(chore)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={target_kid}", status_code=302)


@app.post("/admin/global_chore/claims")
async def admin_global_chore_claims(request: Request):
    form = await request.form()
    decision = (form.get("decision") or "approve").strip().lower()
    chore_id_raw = form.get("chore_id") or "0"
    period_key = (form.get("period_key") or "").strip()
    reason_text = (form.get("reason") or "").strip()
    redirect_target = (form.get("redirect") or "").strip()
    if not redirect_target or not redirect_target.startswith("/"):
        redirect_target = f"/admin/chores?kid_id={GLOBAL_CHORE_KID_ID}"
    if (
        redirect_response := require_admin_permission(
            request, "can_manage_payouts", redirect=redirect_target
        )
    ) is not None:
        return redirect_response
    try:
        chore_id = int(chore_id_raw)
    except ValueError:
        chore_id = 0
    claim_ids_raw = form.getlist("claim_ids") if hasattr(form, "getlist") else []
    selected_ids: List[int] = []
    for raw in claim_ids_raw:
        try:
            selected_ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not selected_ids:
        body = "<div class='card'><p style='color:#f87171;'>Select at least one submission first.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
        return render_page(request, "Approve Global Chores", body, status_code=400)
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if (
            not chore
            or chore.kid_id != GLOBAL_CHORE_KID_ID
            or not period_key
        ):
            body = "<div class='card'><p style='color:#f87171;'>Could not find that Free-for-all chore.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
            return render_page(request, "Approve Global Chores", body, status_code=404)
        claims = session.exec(
            select(GlobalChoreClaim)
            .where(GlobalChoreClaim.id.in_(selected_ids))
        ).all()
        if len(claims) != len(selected_ids):
            body = "<div class='card'><p style='color:#f87171;'>Some submissions could not be loaded.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
            return render_page(request, "Approve Global Chores", body, status_code=400)
        claims.sort(key=lambda c: selected_ids.index(c.id))
        for claim in claims:
            if claim.chore_id != chore.id or claim.period_key != period_key:
                body = "<div class='card'><p style='color:#f87171;'>Selected claims do not match this chore/period.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
                return render_page(request, "Approve Global Chores", body, status_code=400)
            if claim.status != GLOBAL_CHORE_STATUS_PENDING:
                body = "<div class='card'><p style='color:#f87171;'>Only pending submissions can be processed.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
                return render_page(request, "Approve Global Chores", body, status_code=400)
            if (
                denied := ensure_admin_kid_access(
                    request, claim.kid_id, redirect=redirect_target
                )
            ) is not None:
                return denied
        approved_existing = session.exec(
            select(GlobalChoreClaim)
            .where(GlobalChoreClaim.chore_id == chore.id)
            .where(GlobalChoreClaim.period_key == period_key)
            .where(GlobalChoreClaim.status == GLOBAL_CHORE_STATUS_APPROVED)
        ).all()
        approved_total = sum(cl.award_cents for cl in approved_existing)
        remaining_slots = max(0, chore.max_claimants - len(approved_existing))
        now = datetime.utcnow()
        role = admin_role(request) or "admin"
        if decision == "reject":
            for claim in claims:
                claim.status = GLOBAL_CHORE_STATUS_REJECTED
                claim.approved_at = now
                claim.approved_by = role
                if reason_text:
                    claim.notes = reason_text
                session.add(claim)
                child = session.exec(select(Child).where(Child.kid_id == claim.kid_id)).first()
                if child:
                    child.updated_at = now
                    session.add(child)
                    rejection_reason = f"global_chore_denied:{chore.name}:{period_key}"
                    if reason_text:
                        rejection_reason += f" ({reason_text})"
                    session.add(
                        Event(
                            child_id=child.kid_id,
                            change_cents=0,
                            reason=rejection_reason,
                        )
                    )
            session.commit()
            set_admin_notice(
                request,
                f"Rejected {len(claims)} Free-for-all submission{'s' if len(claims) != 1 else ''} for {html_escape(chore.name)}.",
                "success",
            )
            return RedirectResponse(redirect_target, status_code=302)
        if len(claims) > remaining_slots:
            body = "<div class='card'><p style='color:#f87171;'>Not enough spots remain to approve that many kids.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
            return render_page(request, "Approve Global Chores", body, status_code=400)
        remaining_award = max(0, chore.award_cents - approved_total)
        overrides: Dict[int, int] = {}
        override_total = 0
        for claim in claims:
            raw_amount = (form.get(f"amount_{claim.id}") or "").strip()
            if not raw_amount:
                continue
            cents = to_cents_from_dollars_str(raw_amount, 0)
            if cents <= 0:
                continue
            overrides[claim.id] = cents
            override_total += cents
        if override_total > remaining_award:
            body = "<div class='card'><p style='color:#f87171;'>Override amounts exceed the remaining reward.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
            return render_page(request, "Approve Global Chores", body, status_code=400)
        auto_claims = [claim for claim in claims if claim.id not in overrides]
        auto_award_pool = remaining_award - override_total
        share_map: Dict[int, int] = dict(overrides)
        share_count = len(auto_claims)
        if share_count > 0:
            if auto_award_pool < 0:
                auto_award_pool = 0
            base_share = auto_award_pool // share_count
            remainder = auto_award_pool % share_count
            for idx, claim in enumerate(auto_claims):
                share_map[claim.id] = base_share + (1 if idx < remainder else 0)
        payout_total = sum(share_map.values())
        if payout_total > remaining_award:
            body = "<div class='card'><p style='color:#f87171;'>Calculated rewards exceed the remaining amount.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
            return render_page(request, "Approve Global Chores", body, status_code=400)
        for award_value in share_map.values():
            if award_value <= 0:
                continue
            if (
                limit_redirect := ensure_admin_amount_within_limits(
                    request, award_value, "credit", redirect=redirect_target
                )
            ) is not None:
                return limit_redirect
        for claim in claims:
            award_cents = share_map.get(claim.id, 0)
            claim.status = GLOBAL_CHORE_STATUS_APPROVED
            claim.award_cents = award_cents
            claim.approved_at = now
            claim.approved_by = role
            if reason_text:
                claim.notes = reason_text
            session.add(claim)
            child = session.exec(select(Child).where(Child.kid_id == claim.kid_id)).first()
            if child:
                if award_cents > 0:
                    child.balance_cents += award_cents
                    _update_gamification(child, award_cents)
                child.updated_at = now
                reason = f"global_chore:{chore.name}:{period_key}"
                if reason_text:
                    reason += f" ({reason_text})"
                event = Event(child_id=child.kid_id, change_cents=award_cents, reason=reason)
                session.add(event)
            session.add(child)
        session.commit()
    set_admin_notice(
        request,
        f"Approved {len(claims)} Free-for-all submission{'s' if len(claims) != 1 else ''} for {html_escape(chore.name)} totaling {usd(sum(share_map.values()))}.",
        "success",
    )
    return RedirectResponse(redirect_target, status_code=302)


@app.post("/admin/chores/activate")
def admin_chore_activate(request: Request, chore_id: int = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_chores", redirect="/admin?section=chores"
        )
    ) is not None:
        return redirect
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return RedirectResponse("/admin", status_code=302)
        if chore.kid_id and chore.kid_id != GLOBAL_CHORE_KID_ID:
            if (
                denied := ensure_admin_kid_access(
                    request, chore.kid_id, redirect="/admin?section=chores"
                )
            ) is not None:
                return denied
        target_kid = chore.kid_id
        chore.active = True
        session.add(chore)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={target_kid}", status_code=302)


@app.post("/admin/chores/deactivate")
def admin_chore_deactivate(request: Request, chore_id: int = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_chores", redirect="/admin?section=chores"
        )
    ) is not None:
        return redirect
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return RedirectResponse("/admin", status_code=302)
        if chore.kid_id and chore.kid_id != GLOBAL_CHORE_KID_ID:
            if (
                denied := ensure_admin_kid_access(
                    request, chore.kid_id, redirect="/admin?section=chores"
                )
            ) is not None:
                return denied
        target_kid = chore.kid_id
        chore.active = False
        session.add(chore)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={target_kid}", status_code=302)


@app.post("/admin/chore_make_available_now")
def chore_make_available_now(request: Request, chore_id: int = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_chores", redirect="/admin?section=chores"
        )
    ) is not None:
        return redirect
    moment = now_local()
    today = moment.date()
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return render_page(request, "Admin", "<div class='card danger'>Chore not found.</div>")
        if chore.kid_id and chore.kid_id != GLOBAL_CHORE_KID_ID:
            if (
                denied := ensure_admin_kid_access(
                    request, chore.kid_id, redirect="/admin?section=chores"
                )
            ) is not None:
                return denied
        kid_id = chore.kid_id
        if not is_chore_in_window(chore, today):
            return RedirectResponse(f"/admin/chores?kid_id={kid_id}", status_code=302)
        chore_type = normalize_chore_type(chore.type)
        pk = "SPECIAL" if chore_type == "special" else period_key_for(chore_type, moment)
        instance = ChoreInstance(chore_id=chore.id, period_key=pk, status="available")
        session.add(instance)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={kid_id}", status_code=302)

@app.get("/admin/goals", response_class=HTMLResponse)
def admin_goals(request: Request, kid_id: str = Query(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_allowance", redirect="/admin?section=goals"
        )
    ) is not None:
        return redirect
    if (
        denied := ensure_admin_kid_access(
            request, kid_id, redirect="/admin?section=goals"
        )
    ) is not None:
        return denied
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return render_page(request, "Goals", "<div class='card'>Kid not found.</div>")
        goals = session.exec(select(Goal).where(Goal.kid_id == kid_id).order_by(desc(Goal.created_at))).all()
    rows = "".join(
        f"<tr>"
        f"<td data-label='Goal'><b>{goal.name}</b>" + (" <span class='pill' title='Goal reached'>Reached</span>" if goal.saved_cents >= goal.target_cents else "") + "</td>"
        f"<td data-label='Saved' class='right'>{usd(goal.saved_cents)} / {usd(goal.target_cents)}"
        f"<div class='muted'>{format_percent(percent_complete(goal.saved_cents, goal.target_cents))} complete</div></td>"
        f"<td data-label='Actions' class='right'>"
        f"<form class='inline' method='post' action='/admin/goal_update'>"
        f"<input type='hidden' name='goal_id' value='{goal.id}'>"
        f"<input type='hidden' name='kid_id' value='{kid_id}'>"
        f"<input name='name' value='{goal.name}' style='max-width:160px'>"
        f"<input name='target' type='text' data-money value='{dollars_value(goal.target_cents)}' style='max-width:140px'>"
        f"<button type='submit'>Update</button></form> "
        f"<form class='inline' method='post' action='/admin/goal_return_funds' style='margin-left:6px;'>"
        f"<input type='hidden' name='goal_id' value='{goal.id}'>"
        f"<input type='hidden' name='kid_id' value='{kid_id}'>"
        f"<button type='submit' class='danger'>Return Funds</button></form> "
        f"<form class='inline' method='post' action='/admin/goal_delete' onsubmit=\"return confirm('Delete goal and refund saved funds?');\" style='margin-left:6px;'>"
        f"<input type='hidden' name='goal_id' value='{goal.id}'>"
        f"<input type='hidden' name='kid_id' value='{kid_id}'>"
        f"<button type='submit' class='danger'>Delete</button></form>"
        f"</td></tr>"
        for goal in goals
    ) or "<tr><td>(no goals)</td></tr>"
    inner = f"""
    <div class='topbar'><h3>Goals — {child.name} <span class='pill' style='margin-left:8px;'>{child.kid_id}</span></h3>
      <a href='/admin'><button>Back</button></a>
    </div>
    <div class='card'>
      <form method='post' action='/admin/goal_create' class='inline'>
        <input type='hidden' name='kid_id' value='{kid_id}'>
        <input name='name' placeholder='e.g. Lego set' required>
        <input name='target' type='text' data-money placeholder='target $' required>
        <button type='submit'>Create Goal</button>
      </form>
      <table style='margin-top:8px;'><tr><th>Goal</th><th>Saved</th><th>Actions</th></tr>{rows}</table>
    </div>
    """
    return render_page(request, "Goals", inner)


@app.get("/admin/statement", response_class=HTMLResponse)
def admin_statement(request: Request, kid_id: str = Query(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    cd_rates_bps = {code: DEFAULT_CD_RATE_BPS for code, _, _ in CD_TERM_OPTIONS}
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return render_page(request, "Statement", "<div class='card'>Kid not found.</div>")
        events = session.exec(
            select(Event)
            .where(Event.child_id == kid_id)
            .order_by(desc(Event.timestamp))
            .limit(50)
        ).all()
        goals = session.exec(select(Goal).where(Goal.kid_id == kid_id).order_by(desc(Goal.created_at))).all()
        certificates = session.exec(
            select(Certificate)
            .where(Certificate.kid_id == kid_id)
            .order_by(desc(Certificate.opened_at))
        ).all()
        cd_rates_bps = get_all_cd_rate_bps(session)
    moment = datetime.utcnow()
    instruments = list_market_instruments_for_kid(kid_id)
    holding_details: List[Dict[str, Any]] = []
    total_market_c = 0
    total_invested_c = 0
    total_unrealized_c = 0
    total_realized_c = 0
    for inst in instruments:
        metrics = compute_holdings_metrics(kid_id, inst.symbol)
        holding_details.append(
            {
                "symbol": inst.symbol,
                "name": inst.name or inst.symbol,
                "kind": inst.kind,
                "metrics": metrics,
            }
        )
        total_market_c += metrics["market_value_c"]
        total_invested_c += metrics["invested_cost_c"]
        total_unrealized_c += metrics["unrealized_pl_c"]
        total_realized_c += metrics["realized_pl_c"]
    goal_rows = "".join(
        f"<tr><td data-label='Goal'><b>{goal.name}</b></td>"
        f"<td data-label='Saved' class='right'>{usd(goal.saved_cents)} / {usd(goal.target_cents)}</td>"
        f"<td data-label='Progress' class='right'>{format_percent(percent_complete(goal.saved_cents, goal.target_cents))}</td>"
        f"<td data-label='Status'>{'Reached' if goal.saved_cents >= goal.target_cents else 'In progress'}</td></tr>"
        for goal in goals
    ) or "<tr><td>(no goals yet)</td></tr>"
    reward_events = [
        event
        for event in events
        if isinstance(event.reason, str) and event.reason.startswith("prize:")
    ][:10]
    reward_rows = "".join(
        f"<tr><td data-label='When'>{event.timestamp.strftime('%Y-%m-%d')}</td>"
        f"<td data-label='Reward'>{event.reason.split(':', 1)[1]}</td>"
        f"<td data-label='Cost' class='right'>{usd(-event.change_cents)}</td></tr>"
        for event in reward_events
    ) or "<tr><td>(no rewards)</td></tr>"
    activity_rows = "".join(
        f"<tr><td data-label='When'>{event.timestamp.strftime('%Y-%m-%d %H:%M')}</td>"
        f"<td data-label='Δ Amount' class='right'>{'+' if event.change_cents>=0 else ''}{usd(event.change_cents)}</td>"
        f"<td data-label='Reason'>{event.reason}</td></tr>"
        for event in events
    ) or "<tr><td>(no events)</td></tr>"
    def fmt_signed(value: int) -> str:
        return f"{'+' if value >= 0 else ''}{usd(value)}"

    holding_items: List[str] = []
    for holding in holding_details:
        metrics = holding.get("metrics", {})
        symbol = html_escape(holding.get("symbol", "?"))
        name = html_escape(holding.get("name") or symbol)
        shares = metrics.get("shares", 0.0)
        price_c = metrics.get("price_c", 0)
        value_c = metrics.get("market_value_c", 0)
        invested_c = metrics.get("invested_cost_c", 0)
        unrealized_c = metrics.get("unrealized_pl_c", 0)
        realized_c = metrics.get("realized_pl_c", 0)
        total_pl = unrealized_c + realized_c
        if invested_c:
            pct = (total_pl / invested_c) * 100
            pct_text = f" • Return {pct:.1f}%"
        else:
            pct_text = ""
        holding_items.append(
            "<li>"
            + f"<div style='display:flex; flex-direction:column; gap:2px;'>"
            + f"<div><b>{symbol}</b> <span class='muted'>{name}</span></div>"
            + f"<div class='muted'>Shares {shares:.4f} @ {usd(price_c)}</div>"
            + f"<div>Value {usd(value_c)} • Invested {usd(invested_c)} • P/L {fmt_signed(total_pl)}{pct_text}</div>"
            + f"<div class='muted'>Unrealized {fmt_signed(unrealized_c)} • Realized {fmt_signed(realized_c)}</div>"
            + "</div>"
            + "</li>"
        )
    holdings_html = (
        "<ul class='portfolio-holdings'>" + "".join(holding_items) + "</ul>"
    ) if holding_items else "<div class='muted'>No tracked holdings yet.</div>"

    cert_rows = ""
    active_cd_total = 0
    active_cd_count = 0
    ready_cd = 0
    certificate_items: List[str] = []
    for certificate in certificates:
        value_c = certificate_value_cents(certificate, at=moment)
        maturity = certificate_maturity_date(certificate)
        progress_pct = certificate_progress_percent(certificate, at=moment)
        rate_display = certificate.rate_bps / 100
        if certificate.matured_at:
            status = f"Cashed out on {certificate.matured_at.strftime('%Y-%m-%d')}"
            progress_pct = 100.0
        elif moment >= maturity:
            status = "Matured — ready to cash out"
            ready_cd += 1
        else:
            status = f"Matures {maturity:%Y-%m-%d}"
        if certificate.matured_at is None:
            active_cd_total += value_c
            active_cd_count += 1
        certificate_items.append(
            "<li>"
            + f"<div style='display:flex; flex-direction:column; gap:2px;'>"
            + f"<div><b>{usd(certificate.principal_cents)}</b> principal • {rate_display:.2f}% ({certificate_term_label(certificate)})</div>"
            + f"<div>Value {usd(value_c)} • Progress {format_percent(progress_pct)}</div>"
            + f"<div class='muted'>{status}</div>"
            + "</div>"
            + "</li>"
        )
    certificates_html = (
        "<ul class='portfolio-cds'>" + "".join(certificate_items) + "</ul>"
    ) if certificate_items else "<div class='muted'>(no certificates)</div>"
    cd_rates_pct = {
        code: cd_rates_bps.get(code, DEFAULT_CD_RATE_BPS) / 100 for code, _, _ in CD_TERM_OPTIONS
    }
    cd_rates_summary = ", ".join(
        f"{label} {cd_rates_pct[code]:.2f}%" for code, label, _ in CD_TERM_OPTIONS
    )
    ready_note = (
        f"<div class='muted' style='margin-top:4px;'>{ready_cd} certificate{'s' if ready_cd != 1 else ''} ready to cash out.</div>"
        if ready_cd
        else "<div class='muted' style='margin-top:4px;'>All active certificates are still growing.</div>"
    )
    snapshot_card = f"""
    <div class='card'>
      <h3>Account Snapshot</h3>
      <div><b>Balance:</b> {usd(child.balance_cents)}</div>
      <div><b>Weekly allowance:</b> {usd(child.allowance_cents)}</div>
      <div><b>Level:</b> {child.level} • Streak: {child.streak_days} days</div>
      <div style='margin-top:6px;'><b>Badges:</b> {_badges_html(child.badges)}</div>
      <div class='muted' style='margin-top:6px;'>Last updated {(child.updated_at or datetime.utcnow()):%Y-%m-%d %H:%M}</div>
    </div>
    """
    net_pl_c = total_unrealized_c + total_realized_c
    total_assets_c = child.balance_cents + total_market_c + active_cd_total
    total_return_pct = (net_pl_c / total_invested_c * 100) if total_invested_c else None
    return_line = (
        f"Total return {total_return_pct:.2f}%"
        if total_return_pct is not None
        else "Total return —"
    )
    investing_card = f"""
    <div class='card'>
      <h3>Investing Overview</h3>
      <div><b>Total assets:</b> {usd(total_assets_c)}</div>
      <div><b>Cash:</b> {usd(child.balance_cents)}</div>
      <div><b>Markets:</b> {usd(total_market_c)} • CDs {usd(active_cd_total)}</div>
      <div><b>P/L:</b> {fmt_signed(total_unrealized_c)} unrealized, {fmt_signed(total_realized_c)} realized • {return_line}</div>
      <h4 style='margin-top:12px;'>Stock &amp; fund holdings</h4>
      {holdings_html}
      <h4 style='margin-top:12px;'>Certificates of Deposit</h4>
      <div class='muted'>Current rates: {cd_rates_summary}</div>
      <div class='muted'>Active certificates: {active_cd_count} worth {usd(active_cd_total)}.</div>
      {ready_note}
      {certificates_html}
    </div>
    """
    goals_card = f"""
    <div class='card'>
      <h3>Savings Goals</h3>
      <table><tr><th>Goal</th><th>Saved</th><th>Progress</th><th>Status</th></tr>{goal_rows}</table>
    </div>
    """
    rewards_card = f"""
    <div class='card'>
      <h3>Rewards &amp; Prizes</h3>
      <table><tr><th>When</th><th>Reward</th><th>Cost</th></tr>{reward_rows}</table>
    </div>
    """
    activity_card = f"""
    <div class='card'>
      <h3>Recent Activity</h3>
      <table><tr><th>When</th><th>Δ Amount</th><th>Reason</th></tr>{activity_rows}</table>
    </div>
    """
    inner = f"""
    <div class='topbar'><h3>Account Statement — {child.name} <span class='pill' style='margin-left:8px;'>{child.kid_id}</span></h3>
      <a href='/admin'><button>Back</button></a>
    </div>
    <div class='grid'>
      {snapshot_card}
      {investing_card}
    </div>
    <div class='grid'>
      {goals_card}
      {rewards_card}
    </div>
    {activity_card}
    """
    return render_page(request, "Account Statement", inner)


@app.post("/admin/goal_create")
def admin_goal_create(request: Request, kid_id: str = Form(...), name: str = Form(...), target: str = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_allowance", redirect="/admin?section=goals"
        )
    ) is not None:
        return redirect
    if (
        denied := ensure_admin_kid_access(
            request, kid_id, redirect="/admin?section=goals"
        )
    ) is not None:
        return denied
    target_c = to_cents_from_dollars_str(target, 0)
    with Session(engine) as session:
        session.add(Goal(kid_id=kid_id, name=name.strip(), target_cents=target_c))
        session.commit()
    return RedirectResponse(f"/admin/goals?kid_id={kid_id}", status_code=302)


@app.post("/admin/goal_update")
def admin_goal_update(request: Request, goal_id: int = Form(...), kid_id: str = Form(...), name: str = Form(...), target: str = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_allowance", redirect="/admin?section=goals"
        )
    ) is not None:
        return redirect
    if (
        denied := ensure_admin_kid_access(
            request, kid_id, redirect="/admin?section=goals"
        )
    ) is not None:
        return denied
    target_c = to_cents_from_dollars_str(target, 0)
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        if not goal or goal.kid_id != kid_id:
            return RedirectResponse(f"/admin/goals?kid_id={kid_id}", status_code=302)
        goal.name = name.strip()
        goal.target_cents = max(0, target_c)
        if goal.saved_cents >= goal.target_cents and goal.achieved_at is None:
            goal.achieved_at = datetime.utcnow()
            session.add(Event(child_id=kid_id, change_cents=0, reason=f"goal_reached:{goal.name}"))
        session.add(goal)
        session.commit()
    return RedirectResponse(f"/admin/goals?kid_id={kid_id}", status_code=302)


@app.post("/admin/goal_return_funds")
def admin_goal_return_funds(request: Request, goal_id: int = Form(...), kid_id: Optional[str] = Form(None)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_allowance", redirect="/admin?section=goals"
        )
    ) is not None:
        return redirect
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        if not goal:
            return RedirectResponse("/admin", status_code=302)
        kid = kid_id or goal.kid_id
        child = session.exec(select(Child).where(Child.kid_id == kid)).first()
        if not child:
            return RedirectResponse("/admin", status_code=302)
        redirect_target = f"/admin/goals?kid_id={kid}"
        if (
            denied := ensure_admin_kid_access(
                request, kid, redirect=redirect_target
            )
        ) is not None:
            return denied
        if goal.saved_cents > 0:
            if (
                limit_redirect := ensure_admin_amount_within_limits(
                    request, goal.saved_cents, "credit", redirect=redirect_target
                )
            ) is not None:
                return limit_redirect
            child.balance_cents += goal.saved_cents
            session.add(Event(child_id=kid, change_cents=goal.saved_cents, reason=f"goal_refund_admin:{goal.name}"))
            goal.saved_cents = 0
            child.updated_at = datetime.utcnow()
            session.add(child)
            session.add(goal)
            session.commit()
    if kid_id:
        return RedirectResponse(f"/admin/goals?kid_id={kid_id}", status_code=302)
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/goal_delete")
def admin_goal_delete(request: Request, goal_id: int = Form(...), kid_id: Optional[str] = Form(None)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_allowance", redirect="/admin?section=goals"
        )
    ) is not None:
        return redirect
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        if not goal:
            return RedirectResponse("/admin", status_code=302)
        target_kid = kid_id or goal.kid_id
        if (
            denied := ensure_admin_kid_access(
                request, target_kid, redirect=f"/admin/goals?kid_id={target_kid}"
            )
        ) is not None:
            return denied
        child = session.exec(select(Child).where(Child.kid_id == goal.kid_id)).first()
        if child and goal.saved_cents > 0:
            redirect_target = f"/admin/goals?kid_id={goal.kid_id}"
            if (
                limit_redirect := ensure_admin_amount_within_limits(
                    request, goal.saved_cents, "credit", redirect=redirect_target
                )
            ) is not None:
                return limit_redirect
            child.balance_cents += goal.saved_cents
            session.add(Event(child_id=goal.kid_id, change_cents=goal.saved_cents, reason=f"goal_refund_delete_admin:{goal.name}"))
            child.updated_at = datetime.utcnow()
            session.add(child)
        session.delete(goal)
        session.commit()
    if kid_id:
        return RedirectResponse(f"/admin/goals?kid_id={kid_id}", status_code=302)
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/goal_grant")
def admin_goal_grant(request: Request, goal_id: int = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_allowance", redirect="/admin?section=goals"
        )
    ) is not None:
        return redirect
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        if not goal:
            return RedirectResponse("/admin", status_code=302)
        if (
            denied := ensure_admin_kid_access(
                request, goal.kid_id, redirect=f"/admin/goals?kid_id={goal.kid_id}"
            )
        ) is not None:
            return denied
        session.add(Event(child_id=goal.kid_id, change_cents=0, reason=f"goal_granted:{goal.name}"))
        session.delete(goal)
        session.commit()
    return RedirectResponse("/admin", status_code=302)

@app.post("/create_kid")
def create_kid(request: Request, kid_id: str = Form(...), name: str = Form(...), starting: str = Form("0.00"), allowance: str = Form("0.00"), kid_pin: str = Form("")):
    if (
        redirect := require_admin_permission(
            request, "can_create_accounts", redirect="/admin?section=accounts"
        )
    ) is not None:
        return redirect
    starting_c = to_cents_from_dollars_str(starting, 0)
    allowance_c = to_cents_from_dollars_str(allowance, 0)
    with Session(engine) as session:
        if session.exec(select(Child).where(Child.kid_id == kid_id)).first():
            body = "<div class='card'><p style='color:#ff6b6b;'>kid_id exists.</p><p><a href='/admin'>Back</a></p></div>"
            return render_page(request, "Admin", body)
        child = Child(
            kid_id=kid_id.strip(),
            name=name.strip(),
            balance_cents=starting_c,
            allowance_cents=allowance_c,
            kid_pin=(kid_pin or "").strip(),
        )
        session.add(child)
        if starting_c:
            session.add(Event(child_id=kid_id.strip(), change_cents=starting_c, reason="starting_balance"))
        session.commit()
    set_admin_notice(request, f"Created kid profile {html_escape(name.strip())}.", "success")
    return RedirectResponse("/admin?section=accounts#create-kid", status_code=302)


@app.post("/delete_kid")
def delete_kid(request: Request, kid_id: str = Form(...), pin: str = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_delete_accounts", redirect="/admin?section=accounts"
        )
    ) is not None:
        return redirect
    with Session(engine) as session:
        if resolve_admin_role(pin, session=session) is None:
            body = "<div class='card'><p style='color:#ff6b6b;'>Incorrect parent PIN.</p><p><a href='/admin'>Back</a></p></div>"
            return render_page(request, "Admin", body)
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return RedirectResponse("/admin?section=accounts", status_code=302)
        if (
            denied := ensure_admin_kid_access(
                request, child.kid_id, redirect="/admin?section=accounts"
            )
        ) is not None:
            return denied
        kid_name = child.name
        for event in session.exec(select(Event).where(Event.child_id == kid_id)).all():
            session.delete(event)
        for goal in session.exec(select(Goal).where(Goal.kid_id == kid_id)).all():
            session.delete(goal)
        chore_ids = [ch.id for ch in session.exec(select(Chore).where(Chore.kid_id == kid_id)).all()]
        if chore_ids:
            for inst in session.exec(select(ChoreInstance).where(ChoreInstance.chore_id.in_(chore_ids))).all():
                session.delete(inst)
            for chore_id in chore_ids:
                chore = session.get(Chore, chore_id)
                if chore:
                    session.delete(chore)
        for tx in session.exec(select(InvestmentTx).where(InvestmentTx.kid_id == kid_id)).all():
            session.delete(tx)
        holding = session.exec(select(Investment).where(Investment.kid_id == kid_id)).first()
        if holding:
            session.delete(holding)
        for certificate in session.exec(select(Certificate).where(Certificate.kid_id == kid_id)).all():
            session.delete(certificate)
        for link in session.exec(
            select(KidMarketInstrument).where(KidMarketInstrument.kid_id == kid_id)
        ).all():
            session.delete(link)
        session.delete(child)
        session.commit()
    set_admin_notice(
        request,
        f"Deleted kid profile {html_escape(kid_name)} ({html_escape(kid_id)}).",
        "success",
    )
    return RedirectResponse("/admin?section=children", status_code=302)


@app.post("/admin/set_parent_pin")
def admin_set_parent_pin(
    request: Request,
    role: str = Form(...),
    new_pin: str = Form(...),
    confirm_pin: str = Form(...),
):
    if (
        redirect := require_admin_permission(
            request, "can_change_admin_pins", redirect="/admin?section=admins"
        )
    ) is not None:
        return redirect
    normalized_role = (role or "").lower()
    available_roles = {admin["role"] for admin in all_parent_admins()}
    if normalized_role not in available_roles:
        body = "<div class='card'><p style='color:#ff6b6b;'>Select an existing admin before updating the PIN.</p><p><a href='/admin'>Back</a></p></div>"
        return render_page(request, "Admin", body)
    pin_value = (new_pin or "").strip()
    confirmation = (confirm_pin or "").strip()
    if not pin_value:
        body = "<div class='card'><p style='color:#ff6b6b;'>Enter a new PIN before saving.</p><p><a href='/admin'>Back</a></p></div>"
        return render_page(request, "Admin", body)
    if pin_value != confirmation:
        body = "<div class='card'><p style='color:#ff6b6b;'>Confirmation PIN does not match.</p><p><a href='/admin'>Back</a></p></div>"
        return render_page(request, "Admin", body)
    set_parent_pin(normalized_role, pin_value)
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/add_parent_admin")
def admin_add_parent_admin(
    request: Request,
    label: str = Form(...),
    pin: str = Form(...),
    confirm_pin: str = Form(...),
):
    if (
        redirect := require_admin_permission(
            request, "can_create_admins", redirect="/admin?section=admins"
        )
    ) is not None:
        return redirect
    display_name = (label or "").strip()
    pin_value = (pin or "").strip()
    confirmation = (confirm_pin or "").strip()
    if not display_name:
        body = "<div class='card'><p style='color:#ff6b6b;'>Enter a name for the new admin.</p><p><a href='/admin'>Back</a></p></div>"
        return render_page(request, "Admin", body)
    if not pin_value:
        body = "<div class='card'><p style='color:#ff6b6b;'>Enter a PIN for the new admin.</p><p><a href='/admin'>Back</a></p></div>"
        return render_page(request, "Admin", body)
    if pin_value != confirmation:
        body = "<div class='card'><p style='color:#ff6b6b;'>Confirmation PIN does not match.</p><p><a href='/admin'>Back</a></p></div>"
        return render_page(request, "Admin", body)
    base_key = _normalize_parent_role_key(display_name)
    with Session(engine) as session:
        existing = {admin["role"] for admin in all_parent_admins(session)}
        slug = base_key
        suffix = 1
        while slug in existing:
            suffix += 1
            slug = f"{base_key}{suffix}"
        extras = _load_extra_parent_admins(session)
        extras = [entry for entry in extras if entry["role"] not in DEFAULT_PARENT_ROLES]
        extras.append({"role": slug, "label": display_name})
        MetaDAO.set(session, EXTRA_PARENT_ADMINS_KEY, json.dumps(extras))
        MetaDAO.set(session, _parent_pin_meta_key(slug), pin_value)
        save_admin_privileges(session, AdminPrivileges.default(slug))
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/delete_parent_admin")
def admin_delete_parent_admin(request: Request, role: str = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_delete_admins", redirect="/admin?section=admins"
        )
    ) is not None:
        return redirect
    normalized_role = (role or "").strip().lower()
    if not normalized_role:
        body = "<div class='card'><p style='color:#ff6b6b;'>Select an admin to delete first.</p><p><a href='/admin'>Back</a></p></div>"
        return render_page(request, "Admin", body, status_code=400)
    if normalized_role in DEFAULT_PARENT_ROLES:
        body = "<div class='card'><p style='color:#ff6b6b;'>Default admins cannot be removed.</p><p><a href='/admin'>Back</a></p></div>"
        return render_page(request, "Admin", body, status_code=400)
    with Session(engine) as session:
        extras = _load_extra_parent_admins(session)
        filtered = [entry for entry in extras if entry["role"] != normalized_role]
        if len(filtered) == len(extras):
            body = "<div class='card'><p style='color:#ff6b6b;'>Could not find that admin account.</p><p><a href='/admin'>Back</a></p></div>"
            return render_page(request, "Admin", body, status_code=404)
        MetaDAO.set(session, EXTRA_PARENT_ADMINS_KEY, json.dumps(filtered))
        pin_key = _parent_pin_meta_key(normalized_role)
        existing_pin = session.get(MetaKV, pin_key)
        if existing_pin:
            session.delete(existing_pin)
        delete_admin_privileges(session, normalized_role)
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/update_privileges")
async def admin_update_privileges(request: Request):
    if (redirect := require_admin(request)) is not None:
        return redirect
    if (admin_role(request) or "").lower() != "dad":
        return admin_forbidden(
            request,
            "Only the default Dad administrator can update privileges.",
            "/admin?section=admins",
        )
    form = await request.form()
    target_role = (form.get("role") or "").strip().lower()
    if not target_role:
        return admin_forbidden(
            request,
            "Select an admin before saving privileges.",
            "/admin?section=admins",
        )
    if target_role in DEFAULT_PARENT_ROLES:
        return admin_forbidden(
            request,
            "Default admins always retain full access.",
            "/admin?section=admins",
        )
    kid_scope_raw = (form.get("kid_scope") or "all").strip().lower()
    kid_scope = kid_scope_raw if kid_scope_raw in {"all", "custom"} else "all"
    kid_ids: List[str] = []
    if kid_scope == "custom":
        if hasattr(form, "getlist"):
            kid_ids = [kid.strip() for kid in form.getlist("kid_ids") if kid and kid.strip()]
        else:
            raw_ids = form.get("kid_ids") or ""
            kid_ids = [kid.strip() for kid in raw_ids.split(",") if kid.strip()]

    def parse_limit(field_name: str) -> Optional[int]:
        raw_value = (form.get(field_name) or "").strip()
        if not raw_value:
            return None
        cents = to_cents_from_dollars_str(raw_value, 0)
        return cents if cents > 0 else None

    def checkbox_enabled(field_name: str) -> bool:
        value = form.get(field_name)
        if value is None:
            return False
        return str(value).lower() in {"1", "true", "on", "yes"}

    max_credit_c = parse_limit("max_credit")
    max_debit_c = parse_limit("max_debit")
    permission_fields = {
        "can_manage_payouts": checkbox_enabled("perm_payouts"),
        "can_manage_chores": checkbox_enabled("perm_chores"),
        "can_manage_time": checkbox_enabled("perm_time"),
        "can_manage_allowance": checkbox_enabled("perm_allowance"),
        "can_manage_prizes": checkbox_enabled("perm_prizes"),
        "can_create_accounts": checkbox_enabled("perm_create_accounts"),
        "can_delete_accounts": checkbox_enabled("perm_delete_accounts"),
        "can_adjust_balances": checkbox_enabled("perm_adjust_balances"),
        "can_transfer_funds": checkbox_enabled("perm_transfer"),
        "can_create_admins": checkbox_enabled("perm_create_admins"),
        "can_delete_admins": checkbox_enabled("perm_delete_admins"),
        "can_change_admin_pins": checkbox_enabled("perm_change_pins"),
        "can_manage_investing": checkbox_enabled("perm_investing"),
    }
    with Session(engine) as session:
        existing_roles = {admin["role"] for admin in all_parent_admins(session)}
        if target_role not in existing_roles:
            return admin_forbidden(
                request,
                "That admin account could not be found.",
                "/admin?section=admins",
            )
        privileges = AdminPrivileges(
            role=target_role,
            kid_scope=kid_scope,
            kid_ids=kid_ids,
            max_credit_cents=max_credit_c,
            max_debit_cents=max_debit_c,
            **permission_fields,
        )
        save_admin_privileges(session, privileges)
        session.commit()
    set_admin_notice(request, "Updated admin privileges.", "success")
    return RedirectResponse("/admin?section=admins", status_code=302)


@app.post("/admin/set_allowance")
def admin_set_allowance(request: Request, kid_id: str = Form(...), allowance: str = Form("0.00")):
    if (
        redirect := require_admin_permission(
            request, "can_create_accounts", redirect="/admin?section=accounts"
        )
    ) is not None:
        return redirect
    allowance_c = to_cents_from_dollars_str(allowance, 0)
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return render_page(request, "Admin", "<div class='card'>Child not found.</div>")
        if (
            denied := ensure_admin_kid_access(
                request, child.kid_id, redirect="/admin?section=accounts"
            )
        ) is not None:
            return denied
        child.allowance_cents = allowance_c
        child.updated_at = datetime.utcnow()
        session.add(child)
        session.commit()
    set_admin_notice(request, "Updated allowance.", "success")
    return RedirectResponse("/admin?section=accounts", status_code=302)


@app.post("/admin/set_kid_pin")
def set_kid_pin(request: Request, kid_id: str = Form(...), new_pin: str = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_create_accounts", redirect="/admin?section=accounts"
        )
    ) is not None:
        return redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return render_page(request, "Admin", "<div class='card'>Child not found.</div>")
        if (
            denied := ensure_admin_kid_access(
                request, child.kid_id, redirect="/admin?section=accounts"
            )
        ) is not None:
            return denied
        child.kid_pin = (new_pin or "").strip()
        child.updated_at = datetime.utcnow()
        session.add(child)
        session.commit()
    set_admin_notice(request, "Updated kid PIN.", "success")
    return RedirectResponse("/admin?section=accounts", status_code=302)


@app.post("/adjust_balance")
def adjust_balance(request: Request, kid_id: str = Form(...), amount: str = Form("0.00"), kind: str = Form(...), reason: str = Form("")):
    if (
        redirect := require_admin_permission(
            request, "can_adjust_balances", redirect="/admin?section=accounts"
        )
    ) is not None:
        return redirect
    amount_c = to_cents_from_dollars_str(amount, 0)
    kind = (kind or "").lower()
    if kind not in {"credit", "debit"}:
        return render_page(request, "Admin", "<div class='card'>Invalid type.</div>")
    if (
        limit_redirect := ensure_admin_amount_within_limits(
            request, amount_c, kind, redirect="/admin?section=accounts"
        )
    ) is not None:
        return limit_redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return render_page(request, "Admin", "<div class='card'>Child not found.</div>")
        if (
            denied := ensure_admin_kid_access(
                request, child.kid_id, redirect="/admin?section=accounts"
            )
        ) is not None:
            return denied
        if kind == "credit":
            child.balance_cents += amount_c
            session.add(Event(child_id=kid_id, change_cents=amount_c, reason=reason or "credit"))
        else:
            if amount_c > child.balance_cents:
                return render_page(request, "Admin", "<div class='card'>Insufficient funds.</div>")
            child.balance_cents -= amount_c
            session.add(Event(child_id=kid_id, change_cents=-amount_c, reason=reason or "debit"))
            child.updated_at = datetime.utcnow()
        session.add(child)
        session.commit()
    action = "Credited" if kind == "credit" else "Debited"
    set_admin_notice(
        request,
        f"{action} {usd(amount_c)} for {html_escape(kid_id)}.",
        "success",
    )
    return RedirectResponse("/admin?section=accounts", status_code=302)


@app.post("/admin/transfer")
def admin_transfer(
    request: Request,
    from_kid: str = Form(...),
    to_kid: str = Form(...),
    amount: str = Form("0.00"),
    note: str = Form(""),
):
    if (
        redirect := require_admin_permission(
            request, "can_transfer_funds", redirect="/admin?section=accounts"
        )
    ) is not None:
        return redirect
    from_kid = (from_kid or "").strip()
    to_kid = (to_kid or "").strip()
    if not from_kid or not to_kid or from_kid == to_kid:
        body = "<div class='card'><p style='color:#ff6b6b;'>Choose two different kids for a transfer.</p><p><a href='/admin'>Back</a></p></div>"
        return render_page(request, "Admin", body, status_code=400)
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        return RedirectResponse("/admin?section=accounts", status_code=302)
    if (
        limit_redirect := ensure_admin_amount_within_limits(
            request, amount_c, "debit", redirect="/admin?section=accounts"
        )
    ) is not None:
        return limit_redirect
    if (
        limit_redirect := ensure_admin_amount_within_limits(
            request, amount_c, "credit", redirect="/admin?section=accounts"
        )
    ) is not None:
        return limit_redirect
    with Session(engine) as session:
        sender = session.exec(select(Child).where(Child.kid_id == from_kid)).first()
        recipient = session.exec(select(Child).where(Child.kid_id == to_kid)).first()
        if not sender or not recipient:
            body = "<div class='card'><p style='color:#ff6b6b;'>Child not found.</p><p><a href='/admin'>Back</a></p></div>"
            return render_page(request, "Admin", body, status_code=404)
        if (
            denied := ensure_admin_kid_access(
                request, sender.kid_id, redirect="/admin?section=accounts"
            )
        ) is not None:
            return denied
        if (
            denied := ensure_admin_kid_access(
                request, recipient.kid_id, redirect="/admin?section=accounts"
            )
        ) is not None:
            return denied
        if amount_c > sender.balance_cents:
            body = "<div class='card'><p style='color:#ff6b6b;'>Insufficient funds for this transfer.</p><p><a href='/admin'>Back</a></p></div>"
            return render_page(request, "Admin", body, status_code=400)
        sender.balance_cents -= amount_c
        recipient.balance_cents += amount_c
        sender.updated_at = datetime.utcnow()
        recipient.updated_at = datetime.utcnow()
        note_text = (note or "").strip()
        reason_out = f"transfer_to:{to_kid}" + (f" ({note_text})" if note_text else "")
        reason_in = f"transfer_from:{from_kid}" + (f" ({note_text})" if note_text else "")
        session.add(sender)
        session.add(recipient)
        session.add(Event(child_id=from_kid, change_cents=-amount_c, reason=reason_out))
        session.add(Event(child_id=to_kid, change_cents=amount_c, reason=reason_in))
        session.commit()
    set_admin_notice(
        request,
        f"Transferred {usd(amount_c)} from {html_escape(from_kid)} to {html_escape(to_kid)}.",
        "success",
    )
    return RedirectResponse("/admin?section=accounts", status_code=302)


@app.post("/add_prize")
def add_prize(request: Request, name: str = Form(...), cost: str = Form("0.00"), notes: str = Form("")):
    if (
        redirect := require_admin_permission(
            request, "can_manage_prizes", redirect="/admin?section=prizes"
        )
    ) is not None:
        return redirect
    cost_c = to_cents_from_dollars_str(cost, 0)
    with Session(engine) as session:
        prize = Prize(name=name.strip(), cost_cents=cost_c, notes=notes.strip() or None)
        session.add(prize)
        session.commit()
    set_admin_notice(request, f"Added prize {html_escape(name.strip())}.", "success")
    return RedirectResponse("/admin?section=prizes", status_code=302)


@app.post("/redeem_prize")
def redeem_prize(request: Request, kid_id: str = Form(...), prize_id: int = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_prizes", redirect="/admin?section=prizes"
        )
    ) is not None:
        return redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        prize = session.get(Prize, prize_id)
        if not child or not prize:
            return render_page(request, "Admin", "<div class='card'>Child or prize not found.</div>")
        if (
            denied := ensure_admin_kid_access(
                request, child.kid_id, redirect="/admin?section=prizes"
            )
        ) is not None:
            return denied
        if prize.cost_cents > child.balance_cents:
            return render_page(request, "Admin", "<div class='card'>Insufficient funds for prize.</div>")
        child.balance_cents -= prize.cost_cents
        child.updated_at = datetime.utcnow()
        session.add(Event(child_id=kid_id, change_cents=-prize.cost_cents, reason=f"prize:{prize.name}"))
        session.add(child)
        session.commit()
    set_admin_notice(request, "Redeemed prize.", "success")
    return RedirectResponse("/admin?section=prizes", status_code=302)


@app.post("/delete_prize")
def delete_prize(request: Request, prize_id: int = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_prizes", redirect="/admin?section=prizes"
        )
    ) is not None:
        return redirect
    with Session(engine) as session:
        prize = session.get(Prize, prize_id)
        if not prize:
            return render_page(request, "Admin", "<div class='card'>Prize not found.</div>")
        session.delete(prize)
        session.commit()
    set_admin_notice(request, "Deleted prize.", "success")
    return RedirectResponse("/admin?section=prizes", status_code=302)

@app.post("/admin/chore_deny")
def admin_chore_deny(
    request: Request,
    instance_id: int = Form(...),
    redirect: str = Form("/admin?section=payouts"),
):
    redirect_target = (redirect or "").strip()
    if not redirect_target or not redirect_target.startswith("/"):
        redirect_target = "/admin?section=payouts"
    if (
        auth_redirect := require_admin_permission(
            request, "can_manage_payouts", redirect=redirect_target
        )
    ) is not None:
        return auth_redirect
    with Session(engine) as session:
        instance = session.get(ChoreInstance, instance_id)
        if not instance or instance.status != "pending":
            set_admin_notice(request, "That chore is no longer pending.", "error")
            return RedirectResponse(redirect_target, status_code=302)
        instance.status = "available"
        instance.completed_at = None
        chore = session.get(Chore, instance.chore_id)
        if chore and chore.kid_id:
            if (
                denied := ensure_admin_kid_access(
                    request, chore.kid_id, redirect=redirect_target
                )
            ) is not None:
                return denied
        session.add(instance)
        session.commit()
    set_admin_notice(request, "Moved chore back to Available for review later.", "success")
    return RedirectResponse(redirect_target, status_code=302)


@app.post("/admin/chore_payout")
def admin_chore_payout(
    request: Request,
    instance_id: int = Form(...),
    amount: str = Form(""),
    reason: str = Form(""),
    redirect: str = Form("/admin?section=payouts"),
):
    redirect_target = (redirect or "").strip()
    if not redirect_target or not redirect_target.startswith("/"):
        redirect_target = "/admin?section=payouts"
    if (
        auth_redirect := require_admin_permission(
            request, "can_manage_payouts", redirect=redirect_target
        )
    ) is not None:
        return auth_redirect
    child_name = ""
    chore_name = ""
    payout_c = 0
    with Session(engine) as session:
        instance = session.get(ChoreInstance, instance_id)
        if not instance or instance.status != "pending":
            set_admin_notice(request, "That chore is no longer pending.", "error")
            return RedirectResponse(redirect_target, status_code=302)
        chore = session.get(Chore, instance.chore_id)
        if not chore:
            set_admin_notice(request, "Chore information was missing.", "error")
            return RedirectResponse(redirect_target, status_code=302)
        child = session.exec(select(Child).where(Child.kid_id == chore.kid_id)).first()
        if not child:
            set_admin_notice(request, "Kid account could not be found.", "error")
            return RedirectResponse(redirect_target, status_code=302)
        if (
            denied := ensure_admin_kid_access(
                request, child.kid_id, redirect=redirect_target
            )
        ) is not None:
            return denied
        raw_amount = (amount or "").strip()
        if not raw_amount:
            payout_c = chore.award_cents
        else:
            parsed_amount = to_cents_from_dollars_str(raw_amount, chore.award_cents)
            payout_c = chore.award_cents if parsed_amount <= 0 else parsed_amount
        if (
            limit_redirect := ensure_admin_amount_within_limits(
                request, payout_c, "credit", redirect=redirect_target
            )
        ) is not None:
            return limit_redirect
        moment = _time_provider()
        child.balance_cents += payout_c
        child.updated_at = moment
        _update_gamification(child, payout_c)
        reason_clean = (reason or "").strip()
        reason_text = f"chore:{chore.name}" + (f" ({reason_clean})" if reason_clean else "")
        event = Event(child_id=child.kid_id, change_cents=payout_c, reason=reason_text)
        session.add(event)
        session.add(child)
        session.commit()
        instance.status = "paid"
        instance.paid_event_id = event.id
        session.add(instance)
        session.commit()
        child_name = child.name
        chore_name = chore.name
    if child_name and chore_name:
        set_admin_notice(
            request,
            f"Paid {usd(payout_c)} to {html_escape(child_name)} for {html_escape(chore_name)}.",
            "success",
        )
    return RedirectResponse(redirect_target, status_code=302)


@app.post("/admin/marketplace/payout")
def admin_marketplace_payout(
    request: Request,
    listing_id: int = Form(...),
    amount: str = Form(""),
    reason: str = Form(""),
    redirect: str = Form("/admin?section=payouts"),
):
    redirect_target = (redirect or "").strip() or "/admin?section=payouts"
    if not redirect_target.startswith("/"):
        redirect_target = "/admin?section=payouts"
    if (
        auth_redirect := require_admin_permission(
            request, "can_manage_payouts", redirect=redirect_target
        )
    ) is not None:
        return auth_redirect
    actor = admin_role(request) or "guardian"
    with Session(engine) as session:
        listing = session.get(MarketplaceListing, listing_id)
        if not listing or listing.status != MARKETPLACE_STATUS_SUBMITTED:
            set_admin_notice(request, "That marketplace submission is not pending.", "error")
            return RedirectResponse(redirect_target, status_code=302)
        owner = session.exec(select(Child).where(Child.kid_id == listing.owner_kid_id)).first()
        worker = (
            session.exec(select(Child).where(Child.kid_id == listing.claimed_by)).first()
            if listing.claimed_by
            else None
        )
        chore = session.get(Chore, listing.chore_id)
        if not owner or not worker:
            set_admin_notice(request, "Could not locate kids for that marketplace payout.", "error")
            return RedirectResponse(redirect_target, status_code=302)
        award_cents = listing.chore_award_cents or (chore.award_cents if chore else 0)
        if award_cents < 0:
            award_cents = 0
        if listing.chore_award_cents != award_cents:
            listing.chore_award_cents = award_cents
        default_total_c = listing.offer_cents + award_cents
        raw_amount = (amount or "").strip()
        if raw_amount:
            override_c = to_cents_from_dollars_str(raw_amount, default_total_c)
            final_total_c = default_total_c if override_c <= 0 else override_c
        else:
            final_total_c = default_total_c
        offer_component_c = max(final_total_c - award_cents, 0)
        offer_delta_c = offer_component_c - listing.offer_cents
        if offer_delta_c > 0 and owner.balance_cents < offer_delta_c:
            set_admin_notice(
                request,
                f"{html_escape(owner.name)} does not have enough balance for that override.",
                "error",
            )
            return RedirectResponse(redirect_target, status_code=302)
        moment = datetime.utcnow()
        note_clean = (reason or "").strip() or None
        if offer_delta_c > 0:
            owner.balance_cents -= offer_delta_c
            session.add(
                Event(
                    child_id=owner.kid_id,
                    change_cents=-offer_delta_c,
                    reason=f"Marketplace offer increase: {listing.chore_name}",
                )
            )
        elif offer_delta_c < 0:
            refund_c = -offer_delta_c
            owner.balance_cents += refund_c
            session.add(
                Event(
                    child_id=owner.kid_id,
                    change_cents=refund_c,
                    reason=f"Marketplace offer refund: {listing.chore_name}",
                )
            )
        owner.updated_at = moment
        payout_event: Optional[Event] = None
        if final_total_c > 0:
            worker.balance_cents += final_total_c
            worker.updated_at = moment
            payout_reason = f"Marketplace payout: {listing.chore_name}"
            if note_clean:
                payout_reason += f" ({note_clean})"
            payout_event = Event(
                child_id=worker.kid_id,
                change_cents=final_total_c,
                reason=payout_reason,
            )
            session.add(payout_event)
            session.flush()
            listing.payout_event_id = payout_event.id
        else:
            listing.payout_event_id = None
        if chore:
            chore_type = normalize_chore_type(chore.type)
            period_key = (
                "SPECIAL"
                if chore_type == "special"
                else period_key_for(chore_type, now_local())
            )
            query = select(ChoreInstance).where(ChoreInstance.chore_id == chore.id)
            if chore_type != "special":
                query = query.where(ChoreInstance.period_key == period_key)
            inst = session.exec(query.order_by(desc(ChoreInstance.id))).first()
            if inst:
                inst.status = "paid"
                inst.completed_at = inst.completed_at or moment
                inst.paid_event_id = payout_event.id if payout_event else inst.paid_event_id
                session.add(inst)
        listing.status = MARKETPLACE_STATUS_COMPLETED
        listing.completed_at = moment
        listing.final_payout_cents = final_total_c
        listing.payout_note = note_clean
        listing.resolved_by = actor
        session.add(owner)
        session.add(worker)
        session.add(listing)
        session.commit()
    set_admin_notice(
        request,
        f"Paid {usd(final_total_c)} to {html_escape(worker.name)} for {html_escape(listing.chore_name)}.",
        "success",
    )
    return RedirectResponse(redirect_target, status_code=302)

@app.post("/admin/marketplace/deny")
def admin_marketplace_deny(
    request: Request,
    listing_id: int = Form(...),
    reason: str = Form(""),
    redirect: str = Form("/admin?section=payouts"),
):
    redirect_target = (redirect or "").strip() or "/admin?section=payouts"
    if not redirect_target.startswith("/"):
        redirect_target = "/admin?section=payouts"
    if (
        auth_redirect := require_admin_permission(
            request, "can_manage_payouts", redirect=redirect_target
        )
    ) is not None:
        return auth_redirect
    actor = admin_role(request) or "guardian"
    with Session(engine) as session:
        listing = session.get(MarketplaceListing, listing_id)
        if not listing or listing.status != MARKETPLACE_STATUS_SUBMITTED:
            set_admin_notice(request, "That marketplace submission is not pending.", "error")
            return RedirectResponse(redirect_target, status_code=302)
        owner = session.exec(select(Child).where(Child.kid_id == listing.owner_kid_id)).first()
        if not owner:
            set_admin_notice(request, "Could not locate the listing owner.", "error")
            return RedirectResponse(redirect_target, status_code=302)
        note_clean = (reason or "").strip() or None
        moment = datetime.utcnow()
        owner.balance_cents += listing.offer_cents
        owner.updated_at = moment
        session.add(
            Event(
                child_id=owner.kid_id,
                change_cents=listing.offer_cents,
                reason=f"Marketplace refund: {listing.chore_name}",
            )
        )
        chore = session.get(Chore, listing.chore_id)
        if chore:
            chore_type = normalize_chore_type(chore.type)
            period_key = (
                "SPECIAL"
                if chore_type == "special"
                else period_key_for(chore_type, now_local())
            )
            query = select(ChoreInstance).where(ChoreInstance.chore_id == chore.id)
            if chore_type != "special":
                query = query.where(ChoreInstance.period_key == period_key)
            inst = session.exec(query.order_by(desc(ChoreInstance.id))).first()
            if inst:
                inst.status = "available"
                inst.completed_at = None
                inst.paid_event_id = None
                session.add(inst)
        listing.status = MARKETPLACE_STATUS_REJECTED
        listing.completed_at = moment
        listing.final_payout_cents = 0
        listing.payout_note = note_clean
        listing.resolved_by = actor
        listing.payout_event_id = None
        session.add(owner)
        session.add(listing)
        session.commit()
    msg = "Marketplace submission denied. Funds returned to the owner."
    if note_clean:
        msg = f"{msg} ({html_escape(note_clean)})"
    set_admin_notice(
        request,
        msg,
        "success",
    )
    return RedirectResponse(redirect_target, status_code=302)

@app.post("/admin/rules")
def admin_rules(request: Request, bonus_all: Optional[str] = Form(None), bonus: str = Form("0.00"), penalty_miss: Optional[str] = Form(None), penalty: str = Form("0.00")):
    if (
        redirect := require_admin_permission(
            request, "can_manage_allowance", redirect="/admin?section=rules"
        )
    ) is not None:
        return redirect
    bonus_c = to_cents_from_dollars_str(bonus, 0)
    penalty_c = to_cents_from_dollars_str(penalty, 0)
    with Session(engine) as session:
        MetaDAO.set(session, "rule_bonus_all_complete", "1" if bonus_all else "0")
        MetaDAO.set(session, "rule_bonus_cents", str(bonus_c))
        MetaDAO.set(session, "rule_penalty_on_miss", "1" if penalty_miss else "0")
        MetaDAO.set(session, "rule_penalty_cents", str(penalty_c))
        session.commit()
    set_admin_notice(request, "Saved allowance rules.", "success")
    return RedirectResponse("/admin?section=rules", status_code=302)


@app.post("/admin/certificates/rate")
async def admin_set_certificate_rate(request: Request):
    if (
        redirect := require_admin_permission(
            request, "can_manage_investing", redirect="/admin?section=investing"
        )
    ) is not None:
        return redirect
    form = await request.form()
    invalid_terms: list[str] = []
    updates: Dict[str, int] = {}
    for code, label, _ in CD_TERM_OPTIONS:
        raw_value = (form.get(f"rate_{code}") or "").strip()
        if not raw_value:
            invalid_terms.append(label)
            continue
        try:
            rate_value = float(raw_value)
        except ValueError:
            invalid_terms.append(label)
            continue
        updates[code] = max(0, int(round(rate_value * 100)))
    if invalid_terms:
        details = ", ".join(invalid_terms)
        body = (
            "<div class='card'><p style='color:#ff6b6b;'>Enter numeric rates for: "
            f"{details}.</p><p><a href='/admin'>Back</a></p></div>"
        )
        return render_page(request, "Admin", body, status_code=400)
    if not updates:
        return RedirectResponse("/admin", status_code=302)
    with Session(engine) as session:
        for code, rate_bps in updates.items():
            MetaDAO.set(session, _cd_rate_meta_key(code), str(rate_bps))
        default_rate = updates.get(DEFAULT_CD_TERM_CODE)
        if default_rate is not None:
            MetaDAO.set(session, CD_RATE_KEY, str(default_rate))
        session.commit()
    set_admin_notice(request, "Updated CD rates.", "success")
    return RedirectResponse("/admin?section=investing", status_code=302)


@app.post("/admin/certificates/penalty")
async def admin_set_certificate_penalty(request: Request):
    if (
        redirect := require_admin_permission(
            request, "can_manage_investing", redirect="/admin?section=investing"
        )
    ) is not None:
        return redirect
    form = await request.form()
    updates: Dict[str, int] = {}
    invalid_terms: List[str] = []
    for code, label, _days in CD_TERM_OPTIONS:
        raw_value = (form.get(f"penalty_{code}") or "").strip()
        if raw_value == "":
            updates[code] = 0
            continue
        try:
            value = int(raw_value)
        except ValueError:
            invalid_terms.append(label)
            continue
        if value < 0:
            invalid_terms.append(label)
            continue
        updates[code] = value
    if invalid_terms:
        details = ", ".join(invalid_terms)
        body = (
            "<div class='card'><p style='color:#ff6b6b;'>Enter whole number penalties for: "
            f"{details}.</p><p><a href='/admin'>Back</a></p></div>"
        )
        return render_page(request, "Admin", body, status_code=400)
    with Session(engine) as session:
        for code, days in updates.items():
            MetaDAO.set(session, _cd_penalty_meta_key(code), str(days))
        default_days = updates.get(DEFAULT_CD_TERM_CODE)
        if default_days is not None:
            MetaDAO.set(session, CD_PENALTY_DAYS_KEY, str(default_days))
        session.commit()
    set_admin_notice(request, "Updated CD penalties.", "success")
    return RedirectResponse("/admin?section=investing", status_code=302)


@app.post("/admin/market_instruments/add")
def admin_market_instrument_add(
    request: Request,
    symbol: str = Form(...),
    name: str = Form(""),
    kind: str = Form(INSTRUMENT_KIND_STOCK),
):
    if (
        redirect := require_admin_permission(
            request, "can_manage_investing", redirect="/admin?section=investing"
        )
    ) is not None:
        return redirect
    add_market_instrument(symbol, name, kind)
    set_admin_notice(request, f"Added market instrument {html_escape(symbol)}.", "success")
    return RedirectResponse("/admin?section=investing", status_code=302)


@app.post("/admin/market_instruments/delete")
def admin_market_instrument_delete(request: Request, instrument_id: int = Form(...)):
    if (
        redirect := require_admin_permission(
            request, "can_manage_investing", redirect="/admin?section=investing"
        )
    ) is not None:
        return redirect
    delete_market_instrument(instrument_id)
    set_admin_notice(request, "Removed market instrument.", "success")
    return RedirectResponse("/admin?section=investing", status_code=302)


@app.post("/admin/time_settings")
def admin_time_settings(
    request: Request,
    mode: str = Form(TIME_MODE_AUTO),
    offset: str = Form("0"),
    manual_datetime: str = Form(""),
):
    if (redirect := require_admin(request)) is not None:
        return redirect
    raw_offset = (offset or "0").strip()
    try:
        offset_minutes = int(raw_offset)
    except ValueError:
        offset_minutes = 0
    manual_value = (manual_datetime or "").strip()
    set_time_settings(mode, offset_minutes, manual_value)
    return RedirectResponse("/admin", status_code=302)


@app.get("/admin/audit", response_class=HTMLResponse)
def admin_audit(request: Request):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        pending = session.exec(
            select(ChoreInstance, Chore, Child)
            .where(ChoreInstance.status == "pending")
            .where(ChoreInstance.chore_id == Chore.id)
            .where(Chore.kid_id == Child.kid_id)
        ).all()
        paid = session.exec(
            select(ChoreInstance, Chore, Child)
            .where(ChoreInstance.status == "paid")
            .where(ChoreInstance.chore_id == Chore.id)
            .where(Chore.kid_id == Child.kid_id)
            .order_by(desc(ChoreInstance.id))
            .limit(50)
        ).all()
    pending_rows = "".join(
        f"<tr><td>{child.name}</td><td>{chore.name}</td><td>{usd(chore.award_cents)}</td><td>{instance.completed_at}</td></tr>"
        for instance, chore, child in pending
    ) or "<tr><td>(none)</td></tr>"
    paid_rows = "".join(
        f"<tr><td>{child.name}</td><td>{chore.name}</td><td>{usd(chore.award_cents)}</td><td>{instance.completed_at}</td></tr>"
        for instance, chore, child in paid
    ) or "<tr><td>(none)</td></tr>"
    inner = f"""
    <div class='topbar'><h3>Chore Audit</h3><a href='/admin'><button>Back</button></a></div>
    <div class='grid'>
      <div class='card'><h3>Pending Approval</h3><table><tr><th>Kid</th><th>Chore</th><th>Award</th><th>Completed</th></tr>{pending_rows}</table></div>
      <div class='card'><h3>Recently Paid</h3><table><tr><th>Kid</th><th>Chore</th><th>Award</th><th>Completed</th></tr>{paid_rows}</table></div>
    </div>
    """
    return render_page(request, "Chore Audit", inner)


@app.get("/admin/ledger.csv")
def admin_ledger_csv(request: Request):
    if (redirect := require_admin(request)) is not None:
        return redirect
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "kid_id", "change", "reason"])
    with Session(engine) as session:
        for event in session.exec(select(Event).order_by(Event.timestamp)).all():
            writer.writerow([event.timestamp.isoformat(), event.child_id, event.change_cents / 100, event.reason])
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=ledger.csv"})
