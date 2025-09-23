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

import csv
import io
import json
import math
import os
import re
import sqlite3
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, timedelta
from html import escape as html_escape
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request as URLRequest, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
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
    status: str = "available"  # available|pending|paid
    completed_at: Optional[datetime] = None
    paid_event_id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


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


def kid_authed(request: Request) -> Optional[str]:
    return request.session.get("kid_authed")


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


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------
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
        button{ width:100%; }
        .kiosk{flex-direction:column; align-items:flex-start}
        .kiosk .balance{font-size:42px}
        .modal-card{padding:16px; border-radius:10px; max-height:96vh;}
      }
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


def frame(title: str, inner: str, head_extra: str = "") -> str:
    return (
        f"<html><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'>{head_extra}<title>{title}</title>"
        f"{base_styles()}{money_mask_js()}</head><body>{inner}</body></html>"
    )


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


def list_chore_instances_for_kid(kid_id: str) -> List[Tuple[Chore, Optional[ChoreInstance]]]:
    ensure_instances_for_kid(kid_id)
    moment = now_local()
    today = moment.date()
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
                current = next((i for i in insts if i.status in {"available", "pending"}), None)
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


def list_market_instruments(session: Session | None = None) -> List[MarketInstrument]:
    def _load(active: Session) -> List[MarketInstrument]:
        return (
            active.exec(select(MarketInstrument).order_by(MarketInstrument.symbol)).all()
        )

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
    marker = f"<circle cx='{x_positions[-1]:.2f}' cy='{y_positions[-1]:.2f}' r='3.5' fill='{color}'/>"
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
        + marker
        + "</svg>"
    )


# Ensure core market instruments exist after migrations
ensure_default_instrument()


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

# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Kid Bank")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")


# ---------------------------------------------------------------------------
# Kid-facing routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def landing() -> HTMLResponse:
    inner = """
    <div class='grid'>
      <div class='card'>
        <h3>Kid Sign-In</h3>
        <form method='post' action='/kid/login'>
          <label>kid_id</label><input name='kid_id' placeholder='e.g. alex01' required>
          <label style='margin-top:8px;'>PIN</label><input name='kid_pin' placeholder='your PIN' required>
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
    return HTMLResponse(frame("Kid Bank — Sign In", inner))


@app.post("/kid/login", response_class=HTMLResponse)
def kid_login(request: Request, kid_id: str = Form(...), kid_pin: str = Form(...)) -> HTMLResponse:
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child or (child.kid_pin or "") != (kid_pin or ""):
            body = "<div class='card'><p style='color:#ff6b6b;'>Invalid kid_id or PIN.</p><p><a href='/'>Back</a></p></div>"
            return HTMLResponse(frame("Kid Login", body))
    request.session["kid_authed"] = kid_id
    return RedirectResponse("/kid", status_code=302)


@app.get("/kid", response_class=HTMLResponse)
def kid_home(request: Request, section: str = Query("overview")) -> HTMLResponse:
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    selected_section = (section or "overview").strip().lower()
    moment = now_local()
    today = moment.date()
    global_infos: List[Dict[str, Any]] = []
    kid_global_claims: List[GlobalChoreClaim] = []
    global_chore_lookup: Dict[int, Chore] = {}
    try:
        others: List[Child] = []
        incoming_requests: List[MoneyRequest] = []
        outgoing_requests: List[MoneyRequest] = []
        with Session(engine) as session:
            child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
            if not child:
                request.session.pop("kid_authed", None)
                return RedirectResponse("/", status_code=302)
            chores = list_chore_instances_for_kid(kid_id)
            events = session.exec(
                select(Event)
                .where(Event.child_id == kid_id)
                .order_by(desc(Event.timestamp))
                .limit(20)
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
        event_rows = "".join(
            f"<tr><td data-label='When'>{event.timestamp.strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td data-label='Δ Amount' class='right'>{'+' if event.change_cents>=0 else ''}{usd(event.change_cents)}</td>"
            f"<td data-label='Reason'>{html_escape(event.reason)}</td></tr>"
            for event in events
        ) or "<tr><td>(no events)</td></tr>"
        chore_cards = ""
        for chore, inst in chores:
            status = inst.status if inst else "available"
            if status == "available":
                action = (
                    f"<form class='inline' method='post' action='/kid/checkoff'>"
                    f"<input type='hidden' name='chore_id' value='{chore.id}'>"
                    f"<button type='submit'>I did this (+{usd(chore.award_cents)})</button></form>"
                )
            elif status == "pending":
                action = "<span class='pill'>Pending</span>"
            else:
                action = "<span class='pill'>Paid</span>"
            window = ""
            if chore.start_date or chore.end_date:
                start_display = html_escape(str(chore.start_date)) if chore.start_date else "…"
                end_display = html_escape(str(chore.end_date)) if chore.end_date else "…"
                window = (
                    f"<div class='muted' style='margin-top:4px;'>Active: {start_display} → {end_display}</div>"
                )
            chore_name = html_escape(chore.name)
            chore_type = html_escape(chore.type)
            chore_cards += (
                f"<div class='card'><div><b>{chore_name}</b> <span class='muted'>({chore_type})</span></div>"
                f"{window}<div style='margin-top:6px;'>{action}</div></div>"
            )
        if not chore_cards:
            chore_cards = "<div class='muted'>(no chores yet)</div>"
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
            <h3>Money Moves</h3>
            <div class='muted'>Request money or share with siblings.</div>
            <div style='font-weight:600; margin-top:8px;'>Request Money</div>
            {request_form}
            <div style='font-weight:600; margin-top:14px;'>Send Money</div>
            {send_form}
            {pending_requests_section}
          </div>
        """
        available_chore_count = sum(1 for _, inst in chores if not inst or inst.status == "available")
        pending_chore_count = sum(1 for _, inst in chores if inst and inst.status == "pending")
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
        total_invested = usd(investing_snapshot.get("total_c", 0) if snapshot_ok else 0)
        quick_cards: List[str] = [
            (
                "<div class='card'><div class='muted'>Balance</div>"
                f"<div class='stat-value'>{usd(child.balance_cents)}</div>"
                f"<div class='muted' style='margin-top:4px;'>Allowance {usd(child.allowance_cents)} / week</div>"
                "</div>"
            ),
            (
                "<div class='card'><div class='muted'>Chores</div>"
                f"<div class='stat-value'>{available_chore_count} ready</div>"
                f"<div class='muted' style='margin-top:4px;'>Pending approval {pending_chore_count}</div>"
                "<a href='/kid?section=chores' class='button-link secondary' style='margin-top:10px;'>Open chores</a></div>"
            ),
            (
                "<div class='card'><div class='muted'>Free-for-all</div>"
                f"<div class='stat-value'>{open_global_count} open</div>"
                f"<div class='muted' style='margin-top:4px;'>Pending submissions {pending_global_count}</div>"
                "<a href='/kid?section=freeforall' class='button-link secondary' style='margin-top:10px;'>See chores</a></div>"
            ),
            (
                "<div class='card'><div class='muted'>Goals</div>"
                f"<div class='stat-value'>{len(goals)} active</div>"
                f"<div class='muted' style='margin-top:4px;'>Reached {achieved_goals}</div>"
                "<a href='/kid?section=goals' class='button-link secondary' style='margin-top:10px;'>View goals</a></div>"
            ),
            (
                "<div class='card'><div class='muted'>Money moves</div>"
                f"<div class='stat-value'>{incoming_count} requests</div>"
                f"<div class='muted' style='margin-top:4px;'>You sent {pending_outgoing_count} pending</div>"
                "<a href='/kid?section=money' class='button-link secondary' style='margin-top:10px;'>Go to money</a></div>"
            ),
        ]
        if snapshot_ok:
            quick_cards.append(
                "<div class='card'><div class='muted'>Investing</div>"
                f"<div class='stat-value'>{total_invested}</div>"
                f"<div class='muted' style='margin-top:4px;'>Markets {holdings_count} • CDs {cd_count}</div>"
                "<a href='/kid?section=investing' class='button-link secondary' style='margin-top:10px;'>View investing</a></div>"
            )
        else:
            quick_cards.append(
                "<div class='card'><div class='muted'>Investing</div><div class='stat-value'>—</div>"
                "<div class='muted' style='margin-top:4px;'>Data unavailable.</div>"
                "<a href='/kid?section=investing' class='button-link secondary' style='margin-top:10px;'>View investing</a></div>"
            )
        overview_quick_html = (
            "<div class='grid admin-top' style='margin-top:12px;'>" + "".join(quick_cards) + "</div>"
        )
        available_preview = [
            (chore, inst)
            for chore, inst in chores
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
                f"<li><b>{incoming_count}</b> money request{'s' if incoming_count != 1 else ''} waiting for you.</li>"
            )
        if pending_outgoing_count:
            highlight_items.append(
                f"<li>You have <b>{pending_outgoing_count}</b> sent request{'s' if pending_outgoing_count != 1 else ''} pending.</li>"
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
              <a href='/kid?section=money' class='button-link secondary'>Money moves</a>
              <a href='/kid?section=goals' class='button-link secondary'>Goals</a>
              <a href='/kid?section=activity' class='button-link secondary'>Recent activity</a>
            </div>
          </div>
        """
        kid_name_html = html_escape(child.name)
        kid_id_html = html_escape(child.kid_id)
        kiosk_card = f"""
          <div class='card kiosk'>
            <div>
              <div class='name'>{kid_name_html} <span class='muted'>({kid_id_html})</span></div>
              <div class='muted'>Level {child.level} • Streak {child.streak_days} day{'s' if child.streak_days != 1 else ''} • Badges: {_badges_html(child.badges)}</div>
            </div>
            <div class='balance'>{usd(child.balance_cents)}</div>
          </div>
        """
        overview_content = kiosk_card + overview_quick_html + highlights_card
        chores_content = f"""
          <div class='card'>
            <h3>My Chores</h3>
            {chore_cards}
          </div>
        """
        goals_content = f"""
          <div class='card'>
            <h3>My Goals</h3>
            <form method='post' action='/kid/goal_create' class='inline'>
              <input name='name' placeholder='e.g. Lego set' required>
              <input name='target' type='text' data-money placeholder='target $' required>
              <button type='submit'>Create Goal</button>
            </form>
            <table style='margin-top:8px;'><tr><th>Goal</th><th>Saved</th><th>Actions</th></tr>{goal_rows}</table>
          </div>
        """
        activity_content = f"""
          <div class='card'>
            <h3>Recent Activity</h3>
            <table><tr><th>When</th><th>Δ Amount</th><th>Reason</th></tr>{event_rows}</table>
          </div>
        """
        money_content = (notifications_html if notifications_html else "") + money_card
        sections: List[Tuple[str, str, str]] = [
            ("overview", "Overview", overview_content),
            ("chores", "My Chores", chores_content),
            ("freeforall", "Free-for-all", global_card),
            ("goals", "Goals", goals_content),
            ("money", "Money Moves", money_content),
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
        topbar = (
            "<div class='topbar'><h3>Kid Kiosk</h3><div style='display:flex; gap:8px; align-items:center; flex-wrap:wrap;'>"
            f"<span class='pill'>{kid_name_html} ({kid_id_html})</span>"
            + requests_badge
            + "<form method='post' action='/kid/logout' style='display:inline-block;'><button type='submit' class='pill'>Logout</button></form>"
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
        return HTMLResponse(frame(f"{child.name} — Kid", inner))
    except Exception:
        body = """
        <div class='card'>
          <h3>We hit a snag</h3>
          <p class='muted'>The kid dashboard ran into an error. Check server logs.</p>
          <a href='/'><button>Back to Sign In</button></a>
        </div>
        """
        return HTMLResponse(frame("Kid — Error", body))


def _kid_investing_snapshot(kid_id: str) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"ok": False}
    try:
        instruments = list_market_instruments()
        holdings: List[Dict[str, Any]] = []
        cd_total_c = 0
        cd_count = 0
        ready_count = 0
        cd_rates_bps = {code: DEFAULT_CD_RATE_BPS for code, _, _ in CD_TERM_OPTIONS}
        moment = datetime.utcnow()
        with Session(engine) as session:
            if not instruments:
                instruments = list_market_instruments(session)
            for instrument in instruments:
                holding = session.exec(
                    select(Investment)
                    .where(Investment.kid_id == kid_id, Investment.fund == _normalize_symbol(instrument.symbol))
                ).first()
                share_count = holding.shares if holding else 0.0
                price_c = market_price_cents(instrument.symbol)
                value_c = int(round(share_count * price_c))
                holdings.append(
                    {
                        "symbol": instrument.symbol,
                        "shares": share_count,
                        "price_c": price_c,
                        "value_c": value_c,
                    }
                )
            certificates = session.exec(
                select(Certificate)
                .where(Certificate.kid_id == kid_id)
                .where(Certificate.matured_at == None)  # noqa: E711
                .order_by(desc(Certificate.opened_at))
            ).all()
            cd_rates_bps = get_all_cd_rate_bps(session)
        if certificates:
            for certificate in certificates:
                cd_total_c += certificate_value_cents(certificate, at=moment)
                if moment >= certificate_maturity_date(certificate):
                    ready_count += 1
            cd_count = len(certificates)
        total_market_c = sum(item["value_c"] for item in holdings)
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
        return f"""
          <div class='card'>
            <h3>Investing</h3>
            <div class='muted'>Stocks &amp; certificates of deposit</div>
            <div style='margin-top:6px;'>{market_line}</div>
            <div style='margin-top:4px;'>{primary_line}</div>
            <div style='margin-top:4px;'>{cd_line}</div>
            <div class='muted' style='margin-top:4px;'>Total invested: <b>{usd(total_c)}</b> • CD rates {rate_summary}</div>
            <a href='/kid/invest'><button style='margin-top:8px;'>Open Investing Dashboard</button></a>
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
            return RedirectResponse("/kid", status_code=302)
        if not is_chore_in_window(chore, today):
            return RedirectResponse("/kid", status_code=302)
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
    return RedirectResponse("/kid", status_code=302)


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
            return RedirectResponse("/kid", status_code=302)
        period_key = global_chore_period_key(moment, chore)
        existing_claim = get_global_claim(session, chore.id, kid_id, period_key)
        if existing_claim:
            set_kid_notice(request, "You already submitted this Free-for-all chore for this period.", "error")
            return RedirectResponse("/kid", status_code=302)
        total_claims = count_global_claims(session, chore.id, period_key, include_pending=True)
        if total_claims >= chore.max_claimants:
            set_kid_notice(request, "All spots are taken for that chore.", "error")
            return RedirectResponse("/kid", status_code=302)
        claim = GlobalChoreClaim(
            chore_id=chore.id,
            kid_id=kid_id,
            period_key=period_key,
            status=GLOBAL_CHORE_STATUS_PENDING,
        )
        session.add(claim)
        session.commit()
    set_kid_notice(request, "Submitted your Free-for-all claim!", "success")
    return RedirectResponse("/kid", status_code=302)


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
        return RedirectResponse("/kid", status_code=302)
    if target == kid_id:
        set_kid_notice(request, "Choose someone else to ask for money.", "error")
        return RedirectResponse("/kid", status_code=302)
    if amount_c <= 0:
        set_kid_notice(request, "Enter an amount greater than zero to request money.", "error")
        return RedirectResponse("/kid", status_code=302)
    target_name = target
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            request.session.pop("kid_authed", None)
            return RedirectResponse("/", status_code=302)
        target_child = session.exec(select(Child).where(Child.kid_id == target)).first()
        if not target_child:
            set_kid_notice(request, "Could not find that kid.", "error")
            return RedirectResponse("/kid", status_code=302)
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
    return RedirectResponse("/kid", status_code=302)


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
        return RedirectResponse("/kid", status_code=302)
    with Session(engine) as session:
        money_request = session.get(MoneyRequest, request_id)
        if not money_request or money_request.to_kid_id != kid_id:
            set_kid_notice(request, "That request is no longer waiting on you.", "error")
            return RedirectResponse("/kid", status_code=302)
        if money_request.status != "pending":
            set_kid_notice(request, "That request has already been handled.", "error")
            return RedirectResponse("/kid", status_code=302)
        responder = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        requester = session.exec(select(Child).where(Child.kid_id == money_request.from_kid_id)).first()
        if not responder or not requester:
            set_kid_notice(request, "Could not process that request right now.", "error")
            return RedirectResponse("/kid", status_code=302)
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
            return RedirectResponse("/kid", status_code=302)
        if responder.balance_cents < money_request.amount_cents:
            set_kid_notice(request, "Not enough funds to accept this request right now.", "error")
            return RedirectResponse("/kid", status_code=302)
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
    return RedirectResponse("/kid", status_code=302)


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
        return RedirectResponse("/kid", status_code=302)
    if amount_c <= 0:
        set_kid_notice(request, "Enter an amount greater than zero to send money.", "error")
        return RedirectResponse("/kid", status_code=302)
    recipient_name = ""
    with Session(engine) as session:
        sender = session.exec(select(Child).where(Child.kid_id == from_kid)).first()
        if not sender:
            request.session.pop("kid_authed", None)
            return RedirectResponse("/", status_code=302)
        recipient = session.exec(select(Child).where(Child.kid_id == target)).first()
        if not recipient:
            set_kid_notice(request, "Could not find that kid.", "error")
            return RedirectResponse("/kid", status_code=302)
        if recipient.kid_id == sender.kid_id:
            set_kid_notice(request, "Choose someone else to send money to.", "error")
            return RedirectResponse("/kid", status_code=302)
        if sender.balance_cents < amount_c:
            set_kid_notice(request, "Not enough funds to send that amount.", "error")
            return RedirectResponse("/kid", status_code=302)
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
    return RedirectResponse("/kid", status_code=302)


@app.post("/kid/logout")
def kid_logout(request: Request):
    request.session.pop("kid_authed", None)
    return RedirectResponse("/", status_code=302)


@app.post("/kid/goal_create")
def kid_goal_create(request: Request, name: str = Form(...), target: str = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    target_c = to_cents_from_dollars_str(target, 0)
    with Session(engine) as session:
        session.add(Goal(kid_id=kid_id, name=name.strip(), target_cents=target_c))
        session.commit()
    return RedirectResponse("/kid", status_code=302)


@app.post("/kid/goal_deposit")
def kid_goal_deposit(request: Request, goal_id: int = Form(...), amount: str = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        return RedirectResponse("/kid", status_code=302)
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not goal or not child or goal.kid_id != kid_id:
            return RedirectResponse("/kid", status_code=302)
        if amount_c > child.balance_cents:
            return RedirectResponse("/kid", status_code=302)
        child.balance_cents -= amount_c
        goal.saved_cents += amount_c
        child.updated_at = datetime.utcnow()
        if goal.saved_cents >= goal.target_cents and goal.achieved_at is None:
            goal.achieved_at = datetime.utcnow()
            session.add(Event(child_id=kid_id, change_cents=0, reason=f"goal_reached:{goal.name}"))
        session.add(child)
        session.add(goal)
        session.add(Event(child_id=kid_id, change_cents=-amount_c, reason=f"goal_deposit:{goal.name}"))
        session.commit()
    return RedirectResponse("/kid", status_code=302)


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
            return RedirectResponse("/kid", status_code=302)
        if goal.saved_cents > 0:
            child.balance_cents += goal.saved_cents
            session.add(Event(child_id=kid_id, change_cents=goal.saved_cents, reason=f"goal_refund_delete:{goal.name}"))
        session.delete(goal)
        child.updated_at = datetime.utcnow()
        session.add(child)
        session.commit()
    return RedirectResponse("/kid", status_code=302)


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
        instruments = list_market_instruments()
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
            svg = detailed_history_chart_svg(history)
        else:
            svg = sparkline_svg_from_history(history)
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

        inner = f"""
        {notice_html}{search_card_html}
        {tabs_html}
        <div class='card'>
          <h3>Investing — {instrument_label}</h3>
          <div class='muted'>{instrument_symbol} • {kind_label}</div>
          <div style='margin-bottom:12px;'><b>Available Balance:</b> {usd(balance_c)}</div>
          <div class='grid' style='grid-template-columns:1fr 1fr; gap:12px;'>
            <div class='card'>
              <div><b>Current Price</b></div>
              <div style='font-size:28px; font-weight:800; margin-top:6px;'>{usd(metrics['price_c'])}</div>
              <div class='muted'>{unit_label}</div>
              <div style='margin-top:8px;'>{svg}</div>
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
        """
        return HTMLResponse(frame(f"Investing — {instrument_label}", inner))
    except Exception:
        body = """
        <div class='card'>
          <h3>Investing</h3>
          <p class='muted'>The investing dashboard hit an error. Check server logs.</p>
          <a href='/kid'><button>Back</button></a>
        </div>
        """
        return HTMLResponse(frame("Investing — Error", body))
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
    set_kid_notice(request, f"Tracking {instrument.symbol}.", "success")
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
    return HTMLResponse(frame("Sell Certificate", inner))


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
    with Session(engine) as session:
        instrument = session.exec(
            select(MarketInstrument).where(MarketInstrument.symbol == normalized_symbol)
        ).first()
        if not instrument:
            set_kid_notice(request, "That market is no longer tracked.", "error")
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        holdings = session.exec(
            select(Investment).where(Investment.fund == normalized_symbol)
        ).all()
        active_holders = [h for h in holdings if h.shares > 1e-6]
        own_active = next((h for h in active_holders if h.kid_id == kid_id), None)
        if own_active is not None:
            set_kid_notice(request, "Sell your shares before removing this stock.", "error")
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        other_active = [h for h in active_holders if h.kid_id != kid_id]
        if other_active:
            set_kid_notice(
                request,
                "Another kid is still invested in this stock, so it can't be removed yet.",
                "error",
            )
            return RedirectResponse(
                f"/kid/invest?symbol={symbol}&range={next_range}&chart={chart_mode}",
                status_code=302,
            )
        for holding in holdings:
            if holding.kid_id == kid_id:
                session.delete(holding)
        session.delete(instrument)
        session.commit()
    remaining = list_market_instruments()
    fallback_symbol = symbol
    if not any(_normalize_symbol(inst.symbol) == normalized_symbol for inst in remaining):
        fallback_symbol = (
            remaining[0].symbol if remaining else DEFAULT_MARKET_SYMBOL
        )
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
    return HTMLResponse(frame("Parent Login", inner))


@app.post("/admin/login")
def admin_login(request: Request, pin: str = Form(...)):
    with Session(engine) as session:
        role = resolve_admin_role(pin, session=session)
    if not role:
        body = "<div class='card'><p style='color:#ff6b6b;'>Incorrect PIN.</p><p><a href='/admin/login'>Try again</a></p></div>"
        return HTMLResponse(frame("Parent Login", body))
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
    section: str = Query("goals"),
    child: Optional[str] = Query(None),
):
    if (redirect := require_admin(request)) is not None:
        return redirect
    run_weekly_allowance_if_needed()
    role = admin_role(request)
    selected_section = (section or "goals").strip().lower()
    selected_child = (child or "").strip()
    cd_rates_bps = {code: DEFAULT_CD_RATE_BPS for code, _, _ in CD_TERM_OPTIONS}
    with Session(engine) as session:
        kids = session.exec(select(Child).order_by(Child.name)).all()
        prizes = session.exec(select(Prize).order_by(desc(Prize.created_at))).all()
        events = session.exec(select(Event).order_by(desc(Event.timestamp)).limit(30)).all()
        pending = session.exec(
            select(ChoreInstance, Chore, Child)
            .where(ChoreInstance.status == "pending")
            .where(ChoreInstance.chore_id == Chore.id)
            .where(Chore.kid_id == Child.kid_id)
            .order_by(desc(ChoreInstance.completed_at))
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
        instruments = list_market_instruments(session)
        time_settings = get_time_settings(session)
        parent_admins = all_parent_admins(session)
        approved_global_claims = session.exec(
            select(GlobalChoreClaim)
            .where(GlobalChoreClaim.status == GLOBAL_CHORE_STATUS_APPROVED)
        ).all()
    approved_lookup: Dict[Tuple[int, str], List[GlobalChoreClaim]] = {}
    for approved in approved_global_claims:
        approved_lookup.setdefault((approved.chore_id, approved.period_key), []).append(approved)
    kids_by_id = {kid.kid_id: kid for kid in kids}
    kid_options_html = _kid_options(kids)
    parent_options_html = "".join(
        f"<option value='{admin['role']}'>{html_escape(admin['label'])}</option>"
        for admin in parent_admins
    )
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
            f"<form class='inline' method='post' action='/admin/chore_payout'><input type='hidden' name='instance_id' value='{inst.id}'>"
            "<input name='amount' type='text' data-money placeholder='override $ (optional)' style='max-width:150px'>"
            "<input name='reason' type='text' placeholder='reason (optional)' style='max-width:200px'>"
            "<button type='submit'>Payout</button></form> "
            f"<form class='inline' method='post' action='/admin/chore_deny' style='margin-left:6px;' onsubmit='return confirm(\"Deny and push back to Available?\");'><input type='hidden' name='instance_id' value='{inst.id}'><button type='submit' class='danger'>Deny</button></form>"
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
    pending_card = (
        "<div class='card'>"
        "<h3>Pending Payouts</h3>"
        "<table><tr><th>Kid</th><th>Chore</th><th>Award</th><th>Completed</th><th>Actions</th></tr>"
        f"{pending_rows}</table>"
        "<p class='muted' style='margin-top:6px;'>Audit trail: <a href='/admin/audit'>Pending vs Paid</a></p>"
        "</div>"
    )
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
            child_detail_card = (
                "<div class='card'>"
                f"<h3>{html_escape(child_obj.name)} — Details</h3>"
                f"<div class='muted'>{child_obj.kid_id} • Level {child_obj.level} • Streak {child_obj.streak_days} day{'s' if child_obj.streak_days != 1 else ''}</div>"
                f"<div style='margin-top:6px;'><b>Balance:</b> {usd(child_obj.balance_cents)}</div>"
                "<div class='actions' style='margin-top:12px; flex-wrap:wrap; gap:8px;'>"
                f"<a href='/admin/kiosk?kid_id={child_obj.kid_id}' class='button-link secondary'>Kiosk</a>"
                f"<a href='/admin/kiosk_full?kid_id={child_obj.kid_id}' class='button-link secondary'>Kiosk (auto)</a>"
                f"<a href='/admin/chores?kid_id={child_obj.kid_id}' class='button-link secondary'>Manage chores</a>"
                f"<a href='/admin/goals?kid_id={child_obj.kid_id}' class='button-link secondary'>Goals</a>"
                f"<a href='/admin/statement?kid_id={child_obj.kid_id}' class='button-link secondary'>Statement</a>"
                "</div>"
                "<div class='grid' style='grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin-top:12px;'>"
                f"<form method='post' action='/admin/set_allowance' class='stacked-form'><input type='hidden' name='kid_id' value='{child_obj.kid_id}'><label>Allowance (dollars / week)</label><input name='allowance' type='text' data-money value='{dollars_value(child_obj.allowance_cents)}'><button type='submit'>Save Allowance</button></form>"
                f"<form method='post' action='/admin/set_kid_pin' class='stacked-form'><input type='hidden' name='kid_id' value='{child_obj.kid_id}'><label>Set kid PIN</label><input name='new_pin' placeholder='e.g. 4321'><button type='submit'>Set PIN</button></form>"
                f"<form method='post' action='/delete_kid' class='stacked-form' onsubmit='return confirm(\"Delete kid and all events?\");'><input type='hidden' name='kid_id' value='{child_obj.kid_id}'><label>Parent PIN (confirm)</label><input name='pin' placeholder='parent PIN'><button type='submit' class='danger'>Delete Kid</button></form>"
                "</div>"
                "<p class='muted' style='margin-top:10px;'><a href='/admin?section=children'>← Back to overview</a></p>"
                "</div>"
            )
    children_content = children_overview_card + child_detail_card
    event_rows = "".join(
        (
            "<tr><td data-label='When'>"
            + event.timestamp.strftime("%Y-%m-%d %H:%M")
            + "</td><td data-label='Kid'>"
            + event.child_id
            + "</td><td data-label='Δ Amount' class='right'>"
            + ("+" if event.change_cents >= 0 else "")
            + usd(event.change_cents)
            + "</td><td data-label='Reason'>"
            + html_escape(event.reason)
            + "</td></tr>"
        )
        for event in events
    ) or "<tr><td colspan='4' class='muted'>(no events yet)</td></tr>"
    events_card = (
        "<div class='card'>"
        "<h3>Recent Events</h3>"
        "<p class='muted'>Need a CSV? <a href='/admin/ledger.csv'>Download ledger</a></p>"
        f"<table><tr><th>When</th><th>Kid</th><th>Δ Amount</th><th>Reason</th></tr>{event_rows}</table>"
        "</div>"
    )
    create_kid_card = (
        "<div class='card'>"
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
    accounts_content = create_kid_card + credit_card + transfer_card
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
    moment_admin = _time_provider()
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
    instrument_rows = "".join(
        (
            "<tr>"
            f"<td data-label='Symbol'><b>{inst.symbol}</b></td>"
            f"<td data-label='Name'>{html_escape(inst.name or '')}</td>"
            f"<td data-label='Type'>{('Crypto' if inst.kind == INSTRUMENT_KIND_CRYPTO else 'Stock')}</td>"
            + (
                f"<td data-label='Actions' class='right'><span class='pill'>Default</span></td>"
                if _normalize_symbol(inst.symbol) == _normalize_symbol(DEFAULT_MARKET_SYMBOL)
                else (
                    f"<td data-label='Actions' class='right'><form method='post' action='/admin/market_instruments/delete' class='inline' onsubmit=\"return confirm('Remove this market?');\"><input type='hidden' name='instrument_id' value='{inst.id}'><button type='submit' class='danger'>Delete</button></form></td>"
                )
            )
            + "</tr>"
        )
        for inst in instruments
    ) or "<tr><td colspan='4' class='muted'>No markets configured yet.</td></tr>"
    investing_card = (
        "<div class='card'>"
        "<h3>Investing Controls</h3>"
        "<div class='muted'>Manage available markets and CD settings.</div>"
        "<form method='post' action='/admin/market_instruments/add' style='margin-top:10px;' class='inline'>"
        "<input name='symbol' placeholder='Symbol (e.g. SP500)' required>"
        "<input name='name' placeholder='Display name'>"
        f"<select name='kind'><option value='{INSTRUMENT_KIND_STOCK}'>Stock / Fund</option><option value='{INSTRUMENT_KIND_CRYPTO}'>Crypto</option></select>"
        "<button type='submit'>Add / Update</button>"
        "</form>"
        f"<table style='margin-top:10px;'><tr><th>Symbol</th><th>Name</th><th>Type</th><th>Actions</th></tr>{instrument_rows}</table>"
        f"<div style='margin-top:6px;'><b>Current CD rates:</b> {rate_summary}</div>"
        f"<div>Active certificates: <b>{active_cd_count}</b> worth <b>{usd(active_cd_total)}</b></div>"
        f"<div>Early withdrawal penalties: <b>{penalty_summary}</b></div>"
        f"{ready_note}"
        "<form method='post' action='/admin/certificates/rate' style='margin-top:10px;'>"
        "  <p class='muted'>Adjust the APR for each available term.</p>"
        f"{rate_fields_html}"
        "  <button type='submit' style='margin-top:8px;'>Save Rates</button>"
        "</form>"
        "<form method='post' action='/admin/certificates/penalty' style='margin-top:10px;'>"
        "  <p class='muted'>Set how many days of interest are forfeited when cashing out early.</p>"
        f"{penalty_fields_html}"
        "  <button type='submit' style='margin-top:8px;'>Save Penalties</button>"
        "</form>"
        "</div>"
    )
    current_display = now_local()
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
    admin_list_items = []
    for admin in parent_admins:
        role_key = admin["role"]
        label = html_escape(admin["label"])
        if role_key in DEFAULT_PARENT_ROLES:
            admin_list_items.append(
                f"<li>{label} <span class='muted'>({role_key})</span> <span class='pill'>Default</span></li>"
            )
        else:
            admin_list_items.append(
                "<li>"
                + f"{label} <span class='muted'>({role_key})</span>"
                + f"<form method='post' action='/admin/delete_parent_admin' class='inline' style='margin-left:8px;'><input type='hidden' name='role' value='{role_key}'><button type='submit' class='danger secondary'>Delete</button></form>"
                + "</li>"
            )
    admin_list_html = "".join(admin_list_items) or "<li class='muted'>(none yet)</li>"
    parent_admins_card = (
        "<div class='card'>"
        "<h3>Parent Admins</h3>"
        f"<ul class='admin-list'>{admin_list_html}</ul>"
        "<form method='post' action='/admin/set_parent_pin' class='stacked-form' style='margin-top:12px;'>"
        "<h4>Update PIN</h4>"
        f"<select name='role'>{parent_options_html}</select>"
        "<label>New PIN</label><input name='new_pin' type='password' placeholder='****' autocomplete='new-password' required>"
        "<label>Confirm PIN</label><input name='confirm_pin' type='password' placeholder='****' autocomplete='new-password' required>"
        "<button type='submit'>Set PIN</button>"
        "</form>"
        "<form method='post' action='/admin/add_parent_admin' class='stacked-form' style='margin-top:12px;'>"
        "<h4>Add another admin</h4>"
        "<label>Name</label><input name='label' placeholder='Grandma' required>"
        "<label>PIN</label><input name='pin' type='password' placeholder='****' required>"
        "<label>Confirm PIN</label><input name='confirm_pin' type='password' placeholder='****' required>"
        "<button type='submit'>Add Admin</button>"
        "</form>"
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
        ("goals", "Goals needing action", goals_card, ""),
        ("payouts", "Pending payouts", pending_card, "".join(multi_modals)),
        ("children", "Children overview", children_content, ""),
        ("events", "Recent events", events_card, ""),
        ("accounts", "Account tools", accounts_content, ""),
        ("investing", "Investing controls", investing_card, ""),
        ("chores", "Chore publishing", chores_card, ""),
        ("prizes", "Prizes", prizes_card, ""),
        ("rules", "Allowance rules", rules_card, ""),
        ("time", "Time controls", time_card, ""),
        ("admins", "Parent admins", parent_admins_card, ""),
    ]
    sections_map = {key: {"label": label, "content": content, "extra": extra} for key, label, content, extra in sections}
    if selected_section not in sections_map:
        selected_section = "goals"
    sidebar_links = "".join(
        (
            f"<a href='/admin?section={key}' class='{ 'active' if key == selected_section else ''}'>{html_escape(cfg['label'])}</a>"
        )
        for key, cfg in sections_map.items()
    )
    selected_content = sections_map[selected_section]["content"]
    extra_html = sections_map[selected_section].get("extra", "")
    inner = (
        "<div class='topbar'><h3>Admin Portal</h3><div>"
        + _role_badge(role)
        + "<form method='post' action='/admin/logout' style='display:inline-block; margin-left:8px;'><button type='submit' class='pill'>Logout</button></form>"
        + "</div></div>"
        + "<div class='layout'><nav class='sidebar'>"
        + sidebar_links
        + "</nav><div class='content'>"
        + selected_content
        + "</div></div>"
    )
    return HTMLResponse(frame("Admin", inner + extra_html))
@app.get("/admin/kiosk", response_class=HTMLResponse)
def admin_kiosk(request: Request, kid_id: str = Query(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return HTMLResponse(frame("Kiosk", "<div class='card'>Kid not found.</div>"))
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
    return HTMLResponse(frame(f"Kiosk — {child.name}", inner))


@app.get("/admin/kiosk_full", response_class=HTMLResponse)
def admin_kiosk_full(request: Request, kid_id: str = Query(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return HTMLResponse(frame("Kiosk", "<div class='card'>Kid not found.</div>"))
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
    return HTMLResponse(frame(f"Kiosk — {child.name}", inner, head_extra=head))

@app.get("/admin/chores", response_class=HTMLResponse)
def admin_manage_chores(request: Request, kid_id: str = Query(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
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
                return HTMLResponse(frame("Chores", "<div class='card'>Kid not found.</div>"))
            chores = session.exec(
                select(Chore)
                .where(Chore.kid_id == kid_id)
                .order_by(desc(Chore.created_at))
            ).all()
    rows_parts: List[str] = []
    for chore in chores:
        selected_weekdays = chore_weekdays(chore)
        weekday_controls = "".join(
            f"<label style='margin-right:6px;'><input type='checkbox' name='weekdays' value='{day}'{' checked' if day in selected_weekdays else ''}> {label}</label>"
            for day, label in WEEKDAY_OPTIONS
        )
        specific_value = html_escape(chore.specific_dates or "")
        start_value = chore.start_date.isoformat() if chore.start_date else ""
        end_value = chore.end_date.isoformat() if chore.end_date else ""
        name_value = html_escape(chore.name)
        notes_value = html_escape(chore.notes or "")
        schedule_html = (
            f"<div><input name='start_date' type='date' value='{start_value}' style='max-width:140px;'>"
            f"<input name='end_date' type='date' value='{end_value}' style='max-width:140px; margin-left:6px;'></div>"
            f"<div style='margin-top:6px;'>{weekday_controls}</div>"
            f"<input name='specific_dates' placeholder='YYYY-MM-DD,YYYY-MM-DD' value='{specific_value}' style='margin-top:6px;'>"
        )
        is_global_chore = chore.kid_id == GLOBAL_CHORE_KID_ID
        chore_type = normalize_chore_type(chore.type, is_global=is_global_chore)
        action_html = "<button type='submit'>Save</button></form> "
        if not is_global_chore:
            action_html += (
                "<form class='inline' method='post' action='/admin/chore_make_available_now' style='margin-left:6px;'>"
                f"<input type='hidden' name='chore_id' value='{chore.id}'><button type='submit'>Make Available Now</button></form> "
            )
        if is_global_chore:
            type_options = ["daily", "weekly", "monthly"]
        else:
            type_options = ["daily", "weekly", "monthly", "special"]
        type_select = "".join(
            f"<option value='{opt}' {'selected' if chore_type == opt else ''}>{opt}</option>"
            for opt in type_options
        )
        action_html += (
            "<form class='inline' method='post' action='/admin/chores/deactivate' style='margin-left:6px;'>"
            f"<input type='hidden' name='chore_id' value='{chore.id}'><button type='submit' class='danger'>Deactivate</button></form>"
            if chore.active
            else
            "<form class='inline' method='post' action='/admin/chores/activate' style='margin-left:6px;'>"
            f"<input type='hidden' name='chore_id' value='{chore.id}'><button type='submit'>Activate</button></form>"
        )
        rows_parts.append(
            "<tr>"
            f"<td data-label='Name'><form class='inline' method='post' action='/admin/chores/update'>"
            f"<input type='hidden' name='chore_id' value='{chore.id}'>"
            f"<input name='name' value='{name_value}'></td>"
            f"<td data-label='Type'><select name='type'>{type_select}</select></td>"
            f"<td data-label='Award ($)' class='right'><input name='award' type='text' data-money value='{dollars_value(chore.award_cents)}' style='max-width:120px'></td>"
            f"<td data-label='Max Spots'><input name='max_claimants' type='number' min='1' value='{max(1, chore.max_claimants)}' style='max-width:120px'></td>"
            f"<td data-label='Schedule'>{schedule_html}</td>"
            f"<td data-label='Notes'><input name='notes' value='{notes_value}'></td>"
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
      <table>
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
    return HTMLResponse(frame("Manage Chores", inner))


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
    if (redirect := require_admin(request)) is not None:
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
    normalized_type = normalize_chore_type(type, is_global=kid_value == GLOBAL_CHORE_KID_ID)
    with Session(engine) as session:
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
    return RedirectResponse(f"/admin/chores?kid_id={kid_id}", status_code=302)


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
    if (redirect := require_admin(request)) is not None:
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
    if (redirect := require_admin(request)) is not None:
        return redirect
    form = await request.form()
    decision = (form.get("decision") or "approve").strip().lower()
    chore_id_raw = form.get("chore_id") or "0"
    period_key = (form.get("period_key") or "").strip()
    reason_text = (form.get("reason") or "").strip()
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
        return HTMLResponse(frame("Approve Global Chores", body), status_code=400)
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if (
            not chore
            or chore.kid_id != GLOBAL_CHORE_KID_ID
            or not period_key
        ):
            body = "<div class='card'><p style='color:#f87171;'>Could not find that Free-for-all chore.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
            return HTMLResponse(frame("Approve Global Chores", body), status_code=404)
        claims = session.exec(
            select(GlobalChoreClaim)
            .where(GlobalChoreClaim.id.in_(selected_ids))
        ).all()
        if len(claims) != len(selected_ids):
            body = "<div class='card'><p style='color:#f87171;'>Some submissions could not be loaded.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
            return HTMLResponse(frame("Approve Global Chores", body), status_code=400)
        claims.sort(key=lambda c: selected_ids.index(c.id))
        for claim in claims:
            if claim.chore_id != chore.id or claim.period_key != period_key:
                body = "<div class='card'><p style='color:#f87171;'>Selected claims do not match this chore/period.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
                return HTMLResponse(frame("Approve Global Chores", body), status_code=400)
            if claim.status != GLOBAL_CHORE_STATUS_PENDING:
                body = "<div class='card'><p style='color:#f87171;'>Only pending submissions can be processed.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
                return HTMLResponse(frame("Approve Global Chores", body), status_code=400)
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
            return RedirectResponse(f"/admin/chores?kid_id={GLOBAL_CHORE_KID_ID}", status_code=302)
        if len(claims) > remaining_slots:
            body = "<div class='card'><p style='color:#f87171;'>Not enough spots remain to approve that many kids.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
            return HTMLResponse(frame("Approve Global Chores", body), status_code=400)
        remaining_award = max(0, chore.award_cents - approved_total)
        overrides: Dict[int, int] = {}
        override_total = 0
        for claim in claims:
            raw_amount = (form.get(f"amount_{claim.id}") or "").strip()
            if not raw_amount:
                continue
            cents = to_cents_from_dollars_str(raw_amount, 0)
            cents = max(0, cents)
            overrides[claim.id] = cents
            override_total += cents
        if override_total > remaining_award:
            body = "<div class='card'><p style='color:#f87171;'>Override amounts exceed the remaining reward.</p><p><a href='/admin/chores?kid_id=" + GLOBAL_CHORE_KID_ID + "'>Back</a></p></div>"
            return HTMLResponse(frame("Approve Global Chores", body), status_code=400)
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
            return HTMLResponse(frame("Approve Global Chores", body), status_code=400)
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
    return RedirectResponse(f"/admin/chores?kid_id={GLOBAL_CHORE_KID_ID}", status_code=302)


@app.post("/admin/chores/activate")
def admin_chore_activate(request: Request, chore_id: int = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return RedirectResponse("/admin", status_code=302)
        target_kid = chore.kid_id
        chore.active = True
        session.add(chore)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={target_kid}", status_code=302)


@app.post("/admin/chores/deactivate")
def admin_chore_deactivate(request: Request, chore_id: int = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return RedirectResponse("/admin", status_code=302)
        target_kid = chore.kid_id
        chore.active = False
        session.add(chore)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={target_kid}", status_code=302)


@app.post("/admin/chore_make_available_now")
def chore_make_available_now(request: Request, chore_id: int = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    moment = now_local()
    today = moment.date()
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return HTMLResponse(frame("Admin", "<div class='card danger'>Chore not found.</div>"))
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
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return HTMLResponse(frame("Goals", "<div class='card'>Kid not found.</div>"))
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
    return HTMLResponse(frame("Goals", inner))


@app.get("/admin/statement", response_class=HTMLResponse)
def admin_statement(request: Request, kid_id: str = Query(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    cd_rates_bps = {code: DEFAULT_CD_RATE_BPS for code, _, _ in CD_TERM_OPTIONS}
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return HTMLResponse(frame("Statement", "<div class='card'>Kid not found.</div>"))
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
    metrics = compute_holdings_metrics(kid_id, DEFAULT_MARKET_SYMBOL)
    moment = datetime.utcnow()
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
    cert_rows = ""
    active_cd_total = 0
    active_cd_count = 0
    ready_cd = 0
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
        cert_rows += (
            f"<tr>"
            f"<td data-label='Principal'>{usd(certificate.principal_cents)}</td>"
            f"<td data-label='Rate'>{rate_display:.2f}%</td>"
            f"<td data-label='Term'>{certificate_term_label(certificate)}</td>"
            f"<td data-label='Value' class='right'>{usd(value_c)}</td>"
            f"<td data-label='Progress' class='right'>{format_percent(progress_pct)}</td>"
            f"<td data-label='Status'>{status}</td>"
            "</tr>"
        )
    if not cert_rows:
        cert_rows = "<tr><td colspan='6' class='muted'>(no certificates)</td></tr>"
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
    investing_card = f"""
    <div class='card'>
      <h3>Investing Overview</h3>
      <div><b>Stocks:</b> {usd(metrics['market_value_c'])} ({metrics['shares']:.4f} sh @ {usd(metrics['price_c'])})</div>
      <div>Certificates: <b>{usd(active_cd_total)}</b> across {active_cd_count} active • Rates {cd_rates_summary}</div>
      {ready_note}
      <table style='margin-top:10px;'><tr><th>Principal</th><th>Rate</th><th>Term</th><th>Value</th><th>Progress</th><th>Status</th></tr>{cert_rows}</table>
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
    return HTMLResponse(frame("Account Statement", inner))


@app.post("/admin/goal_create")
def admin_goal_create(request: Request, kid_id: str = Form(...), name: str = Form(...), target: str = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    target_c = to_cents_from_dollars_str(target, 0)
    with Session(engine) as session:
        session.add(Goal(kid_id=kid_id, name=name.strip(), target_cents=target_c))
        session.commit()
    return RedirectResponse(f"/admin/goals?kid_id={kid_id}", status_code=302)


@app.post("/admin/goal_update")
def admin_goal_update(request: Request, goal_id: int = Form(...), kid_id: str = Form(...), name: str = Form(...), target: str = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
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
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        if not goal:
            return RedirectResponse("/admin", status_code=302)
        kid = kid_id or goal.kid_id
        child = session.exec(select(Child).where(Child.kid_id == kid)).first()
        if not child:
            return RedirectResponse("/admin", status_code=302)
        if goal.saved_cents > 0:
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
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        if not goal:
            return RedirectResponse("/admin", status_code=302)
        child = session.exec(select(Child).where(Child.kid_id == goal.kid_id)).first()
        if child and goal.saved_cents > 0:
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
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        goal = session.get(Goal, goal_id)
        if not goal:
            return RedirectResponse("/admin", status_code=302)
        session.add(Event(child_id=goal.kid_id, change_cents=0, reason=f"goal_granted:{goal.name}"))
        session.delete(goal)
        session.commit()
    return RedirectResponse("/admin", status_code=302)

@app.post("/create_kid")
def create_kid(request: Request, kid_id: str = Form(...), name: str = Form(...), starting: str = Form("0.00"), allowance: str = Form("0.00"), kid_pin: str = Form("")):
    if (redirect := require_admin(request)) is not None:
        return redirect
    starting_c = to_cents_from_dollars_str(starting, 0)
    allowance_c = to_cents_from_dollars_str(allowance, 0)
    with Session(engine) as session:
        if session.exec(select(Child).where(Child.kid_id == kid_id)).first():
            body = "<div class='card'><p style='color:#ff6b6b;'>kid_id exists.</p><p><a href='/admin'>Back</a></p></div>"
            return HTMLResponse(frame("Admin", body))
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
    return RedirectResponse("/admin", status_code=302)


@app.post("/delete_kid")
def delete_kid(request: Request, kid_id: str = Form(...), pin: str = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        if resolve_admin_role(pin, session=session) is None:
            body = "<div class='card'><p style='color:#ff6b6b;'>Incorrect parent PIN.</p><p><a href='/admin'>Back</a></p></div>"
            return HTMLResponse(frame("Admin", body))
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return RedirectResponse("/admin", status_code=302)
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
        session.delete(child)
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/set_parent_pin")
def admin_set_parent_pin(
    request: Request,
    role: str = Form(...),
    new_pin: str = Form(...),
    confirm_pin: str = Form(...),
):
    if (redirect := require_admin(request)) is not None:
        return redirect
    normalized_role = (role or "").lower()
    available_roles = {admin["role"] for admin in all_parent_admins()}
    if normalized_role not in available_roles:
        body = "<div class='card'><p style='color:#ff6b6b;'>Select an existing admin before updating the PIN.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body))
    pin_value = (new_pin or "").strip()
    confirmation = (confirm_pin or "").strip()
    if not pin_value:
        body = "<div class='card'><p style='color:#ff6b6b;'>Enter a new PIN before saving.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body))
    if pin_value != confirmation:
        body = "<div class='card'><p style='color:#ff6b6b;'>Confirmation PIN does not match.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body))
    set_parent_pin(normalized_role, pin_value)
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/add_parent_admin")
def admin_add_parent_admin(
    request: Request,
    label: str = Form(...),
    pin: str = Form(...),
    confirm_pin: str = Form(...),
):
    if (redirect := require_admin(request)) is not None:
        return redirect
    display_name = (label or "").strip()
    pin_value = (pin or "").strip()
    confirmation = (confirm_pin or "").strip()
    if not display_name:
        body = "<div class='card'><p style='color:#ff6b6b;'>Enter a name for the new admin.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body))
    if not pin_value:
        body = "<div class='card'><p style='color:#ff6b6b;'>Enter a PIN for the new admin.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body))
    if pin_value != confirmation:
        body = "<div class='card'><p style='color:#ff6b6b;'>Confirmation PIN does not match.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body))
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
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/delete_parent_admin")
def admin_delete_parent_admin(request: Request, role: str = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    normalized_role = (role or "").strip().lower()
    if not normalized_role:
        body = "<div class='card'><p style='color:#ff6b6b;'>Select an admin to delete first.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body), status_code=400)
    if normalized_role in DEFAULT_PARENT_ROLES:
        body = "<div class='card'><p style='color:#ff6b6b;'>Default admins cannot be removed.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body), status_code=400)
    with Session(engine) as session:
        extras = _load_extra_parent_admins(session)
        filtered = [entry for entry in extras if entry["role"] != normalized_role]
        if len(filtered) == len(extras):
            body = "<div class='card'><p style='color:#ff6b6b;'>Could not find that admin account.</p><p><a href='/admin'>Back</a></p></div>"
            return HTMLResponse(frame("Admin", body), status_code=404)
        MetaDAO.set(session, EXTRA_PARENT_ADMINS_KEY, json.dumps(filtered))
        pin_key = _parent_pin_meta_key(normalized_role)
        existing_pin = session.get(MetaKV, pin_key)
        if existing_pin:
            session.delete(existing_pin)
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/set_allowance")
def admin_set_allowance(request: Request, kid_id: str = Form(...), allowance: str = Form("0.00")):
    if (redirect := require_admin(request)) is not None:
        return redirect
    allowance_c = to_cents_from_dollars_str(allowance, 0)
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return HTMLResponse(frame("Admin", "<div class='card'>Child not found.</div>"))
        child.allowance_cents = allowance_c
        child.updated_at = datetime.utcnow()
        session.add(child)
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/set_kid_pin")
def set_kid_pin(request: Request, kid_id: str = Form(...), new_pin: str = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return HTMLResponse(frame("Admin", "<div class='card'>Child not found.</div>"))
        child.kid_pin = (new_pin or "").strip()
        child.updated_at = datetime.utcnow()
        session.add(child)
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/adjust_balance")
def adjust_balance(request: Request, kid_id: str = Form(...), amount: str = Form("0.00"), kind: str = Form(...), reason: str = Form("")):
    if (redirect := require_admin(request)) is not None:
        return redirect
    amount_c = to_cents_from_dollars_str(amount, 0)
    kind = (kind or "").lower()
    if kind not in {"credit", "debit"}:
        return HTMLResponse(frame("Admin", "<div class='card'>Invalid type.</div>"))
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return HTMLResponse(frame("Admin", "<div class='card'>Child not found.</div>"))
        if kind == "credit":
            child.balance_cents += amount_c
            session.add(Event(child_id=kid_id, change_cents=amount_c, reason=reason or "credit"))
        else:
            if amount_c > child.balance_cents:
                return HTMLResponse(frame("Admin", "<div class='card'>Insufficient funds.</div>"))
            child.balance_cents -= amount_c
            session.add(Event(child_id=kid_id, change_cents=-amount_c, reason=reason or "debit"))
        child.updated_at = datetime.utcnow()
        session.add(child)
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/transfer")
def admin_transfer(
    request: Request,
    from_kid: str = Form(...),
    to_kid: str = Form(...),
    amount: str = Form("0.00"),
    note: str = Form(""),
):
    if (redirect := require_admin(request)) is not None:
        return redirect
    from_kid = (from_kid or "").strip()
    to_kid = (to_kid or "").strip()
    if not from_kid or not to_kid or from_kid == to_kid:
        body = "<div class='card'><p style='color:#ff6b6b;'>Choose two different kids for a transfer.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body), status_code=400)
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        return RedirectResponse("/admin", status_code=302)
    with Session(engine) as session:
        sender = session.exec(select(Child).where(Child.kid_id == from_kid)).first()
        recipient = session.exec(select(Child).where(Child.kid_id == to_kid)).first()
        if not sender or not recipient:
            body = "<div class='card'><p style='color:#ff6b6b;'>Child not found.</p><p><a href='/admin'>Back</a></p></div>"
            return HTMLResponse(frame("Admin", body), status_code=404)
        if amount_c > sender.balance_cents:
            body = "<div class='card'><p style='color:#ff6b6b;'>Insufficient funds for this transfer.</p><p><a href='/admin'>Back</a></p></div>"
            return HTMLResponse(frame("Admin", body), status_code=400)
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
    return RedirectResponse("/admin", status_code=302)


@app.post("/add_prize")
def add_prize(request: Request, name: str = Form(...), cost: str = Form("0.00"), notes: str = Form("")):
    if (redirect := require_admin(request)) is not None:
        return redirect
    cost_c = to_cents_from_dollars_str(cost, 0)
    with Session(engine) as session:
        prize = Prize(name=name.strip(), cost_cents=cost_c, notes=notes.strip() or None)
        session.add(prize)
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/redeem_prize")
def redeem_prize(request: Request, kid_id: str = Form(...), prize_id: int = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        prize = session.get(Prize, prize_id)
        if not child or not prize:
            return HTMLResponse(frame("Admin", "<div class='card'>Child or prize not found.</div>"))
        if prize.cost_cents > child.balance_cents:
            return HTMLResponse(frame("Admin", "<div class='card'>Insufficient funds for prize.</div>"))
        child.balance_cents -= prize.cost_cents
        child.updated_at = datetime.utcnow()
        session.add(Event(child_id=kid_id, change_cents=-prize.cost_cents, reason=f"prize:{prize.name}"))
        session.add(child)
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/delete_prize")
def delete_prize(request: Request, prize_id: int = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        prize = session.get(Prize, prize_id)
        if not prize:
            return HTMLResponse(frame("Admin", "<div class='card'>Prize not found.</div>"))
        session.delete(prize)
        session.commit()
    return RedirectResponse("/admin", status_code=302)

@app.post("/admin/chore_deny")
def admin_chore_deny(request: Request, instance_id: int = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        instance = session.get(ChoreInstance, instance_id)
        if not instance or instance.status != "pending":
            return RedirectResponse("/admin", status_code=302)
        instance.status = "available"
        instance.completed_at = None
        session.add(instance)
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/chore_payout")
def admin_chore_payout(request: Request, instance_id: int = Form(...), amount: str = Form(""), reason: str = Form("")):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        instance = session.get(ChoreInstance, instance_id)
        if not instance or instance.status != "pending":
            return RedirectResponse("/admin", status_code=302)
        chore = session.get(Chore, instance.chore_id)
        if not chore:
            return RedirectResponse("/admin", status_code=302)
        child = session.exec(select(Child).where(Child.kid_id == chore.kid_id)).first()
        if not child:
            return RedirectResponse("/admin", status_code=302)
        raw_amount = (amount or "").strip()
        if not raw_amount:
            payout_c = chore.award_cents
        else:
            payout_c = to_cents_from_dollars_str(raw_amount, chore.award_cents)
        payout_c = max(0, payout_c)
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
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/rules")
def admin_rules(request: Request, bonus_all: Optional[str] = Form(None), bonus: str = Form("0.00"), penalty_miss: Optional[str] = Form(None), penalty: str = Form("0.00")):
    if (redirect := require_admin(request)) is not None:
        return redirect
    bonus_c = to_cents_from_dollars_str(bonus, 0)
    penalty_c = to_cents_from_dollars_str(penalty, 0)
    with Session(engine) as session:
        MetaDAO.set(session, "rule_bonus_all_complete", "1" if bonus_all else "0")
        MetaDAO.set(session, "rule_bonus_cents", str(bonus_c))
        MetaDAO.set(session, "rule_penalty_on_miss", "1" if penalty_miss else "0")
        MetaDAO.set(session, "rule_penalty_cents", str(penalty_c))
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/certificates/rate")
async def admin_set_certificate_rate(request: Request):
    if (redirect := require_admin(request)) is not None:
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
        return HTMLResponse(frame("Admin", body), status_code=400)
    if not updates:
        return RedirectResponse("/admin", status_code=302)
    with Session(engine) as session:
        for code, rate_bps in updates.items():
            MetaDAO.set(session, _cd_rate_meta_key(code), str(rate_bps))
        default_rate = updates.get(DEFAULT_CD_TERM_CODE)
        if default_rate is not None:
            MetaDAO.set(session, CD_RATE_KEY, str(default_rate))
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/certificates/penalty")
async def admin_set_certificate_penalty(request: Request):
    if (redirect := require_admin(request)) is not None:
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
        return HTMLResponse(frame("Admin", body), status_code=400)
    with Session(engine) as session:
        for code, days in updates.items():
            MetaDAO.set(session, _cd_penalty_meta_key(code), str(days))
        default_days = updates.get(DEFAULT_CD_TERM_CODE)
        if default_days is not None:
            MetaDAO.set(session, CD_PENALTY_DAYS_KEY, str(default_days))
        session.commit()
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/market_instruments/add")
def admin_market_instrument_add(
    request: Request,
    symbol: str = Form(...),
    name: str = Form(""),
    kind: str = Form(INSTRUMENT_KIND_STOCK),
):
    if (redirect := require_admin(request)) is not None:
        return redirect
    add_market_instrument(symbol, name, kind)
    return RedirectResponse("/admin", status_code=302)


@app.post("/admin/market_instruments/delete")
def admin_market_instrument_delete(request: Request, instrument_id: int = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    delete_market_instrument(instrument_id)
    return RedirectResponse("/admin", status_code=302)


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
    return HTMLResponse(frame("Chore Audit", inner))


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
