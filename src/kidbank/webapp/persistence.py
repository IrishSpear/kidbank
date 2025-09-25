"""Persistence and SQLModel definitions for the KidBank web frontend."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlmodel import Field, Session, SQLModel, create_engine, desc, select

from .config import DEFAULT_GLOBAL_CHORE_TYPE, GLOBAL_CHORE_TYPES, SQLITE_FILE_NAME

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
    penalty_cents: int = 0
    penalty_last_date: Optional[date] = None
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
        if not _column_exists(raw, "chore", "penalty_cents"):
            raw.execute("ALTER TABLE chore ADD COLUMN penalty_cents INTEGER DEFAULT 0;")
        if not _column_exists(raw, "chore", "penalty_last_date"):
            raw.execute("ALTER TABLE chore ADD COLUMN penalty_last_date TEXT;")
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




__all__ = [
    "engine",
    "Child",
    "Event",
    "MoneyRequest",
    "Prize",
    "ChoreType",
    "Chore",
    "ChoreInstance",
    "CHORE_STATUS_PENDING_MARKETPLACE",
    "MARKETPLACE_STATUS_OPEN",
    "MARKETPLACE_STATUS_CLAIMED",
    "MARKETPLACE_STATUS_SUBMITTED",
    "MARKETPLACE_STATUS_COMPLETED",
    "MARKETPLACE_STATUS_CANCELLED",
    "MARKETPLACE_STATUS_REJECTED",
    "MarketplaceListing",
    "MetaKV",
    "Goal",
    "Certificate",
    "Investment",
    "InvestmentTx",
    "MarketInstrument",
    "KidMarketInstrument",
    "GlobalChoreClaim",
    "Lesson",
    "Quiz",
    "QuizAttempt",
    "GLOBAL_CHORE_KID_ID",
    "GLOBAL_CHORE_STATUS_PENDING",
    "GLOBAL_CHORE_STATUS_APPROVED",
    "GLOBAL_CHORE_STATUS_REJECTED",
    "INSTRUMENT_KIND_STOCK",
    "INSTRUMENT_KIND_CRYPTO",
    "DEFAULT_MARKET_SYMBOL",
    "TIME_MODE_AUTO",
    "TIME_MODE_MANUAL",
    "TIME_META_MODE_KEY",
    "TIME_META_MANUAL_KEY",
    "TIME_META_OFFSET_KEY",
    "TIME_META_MANUAL_REF_KEY",
    "create_db_and_tables",
    "run_migrations"
]
