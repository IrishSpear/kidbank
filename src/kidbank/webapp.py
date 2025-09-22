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
import os
import sqlite3
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, timedelta
from html import escape
from typing import Iterable, List, Literal, Optional, Tuple
from urllib.error import HTTPError, URLError
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


def now_local() -> datetime:
    """Return naive local time.  The UI is domestic in scope so this is fine."""

    return datetime.now()


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


class Prize(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    cost_cents: int
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


ChoreType = Literal["daily", "weekly", "special"]


class Chore(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kid_id: str
    name: str
    type: str  # daily|weekly|special
    award_cents: int
    notes: Optional[str] = None
    active: bool = True
    start_date: Optional[date] = None
    end_date: Optional[date] = None
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


class PeerRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    requester_kid: str
    target_kid: str
    amount_cents: int
    reason: str = ""
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class Certificate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kid_id: str
    principal_cents: int
    rate_bps: int
    term_months: int
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    matured_at: Optional[datetime] = None


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
                opened_at TEXT,
                matured_at TEXT
            );
            """
        )
        if not _column_exists(raw, "certificate", "matured_at"):
            raw.execute("ALTER TABLE certificate ADD COLUMN matured_at TEXT;")
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
            CREATE TABLE IF NOT EXISTS peerrequest (
                id INTEGER PRIMARY KEY,
                requester_kid TEXT,
                target_kid TEXT,
                amount_cents INTEGER,
                reason TEXT,
                status TEXT,
                created_at TEXT,
                resolved_at TEXT,
                resolved_by TEXT
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
DEFAULT_CD_RATE_BPS = 250


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
      form.inline{ display:grid; grid-template-columns: 1fr auto; gap:8px; align-items:end; }
      @media (min-width: 900px){ .card form.inline{ grid-template-columns: 1fr 1fr auto; } }
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
      @media (max-width: 640px){
        .card{padding:12px}
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
def period_key_for(chore_type: str, moment: datetime) -> str:
    if chore_type == "daily":
        return moment.strftime("%Y-%m-%d")
    if chore_type == "weekly":
        days_since_sunday = (moment.weekday() + 1) % 7
        sunday = (moment - timedelta(days=days_since_sunday)).date()
        return f"{sunday.isoformat()}-WEEK"
    return "SPECIAL"


def is_chore_in_window(chore: Chore, today: date) -> bool:
    if chore.start_date and today < chore.start_date:
        return False
    if chore.end_date and today > chore.end_date:
        return False
    return chore.active


def ensure_instances_for_kid(kid_id: str) -> None:
    moment = now_local()
    today = moment.date()
    with Session(engine) as session:
        chores = session.exec(select(Chore).where(Chore.kid_id == kid_id, Chore.active == True)).all()  # noqa: E712
        for chore in chores:
            if not is_chore_in_window(chore, today):
                continue
            if chore.type == "special":
                continue
            pk = period_key_for(chore.type, moment)
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
            if chore.type == "daily":
                current = next((i for i in insts if i.period_key == pk_daily), None)
            elif chore.type == "weekly":
                current = next((i for i in insts if i.period_key == pk_weekly), None)
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


def get_cd_rate_bps(session: Session) -> int:
    raw = MetaDAO.get(session, CD_RATE_KEY)
    if raw is None:
        return DEFAULT_CD_RATE_BPS
    try:
        rate = int(raw)
    except ValueError:
        return DEFAULT_CD_RATE_BPS
    return max(0, rate)


def certificate_term_days(certificate: Certificate) -> int:
    return max(0, certificate.term_months) * 30


def certificate_maturity_date(certificate: Certificate) -> datetime:
    return certificate.opened_at + timedelta(days=certificate_term_days(certificate))


def certificate_maturity_value_cents(certificate: Certificate) -> int:
    principal = Decimal(certificate.principal_cents)
    rate = Decimal(certificate.rate_bps) / Decimal(10000)
    interest = (principal * rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(principal + interest)


def certificate_value_cents(certificate: Certificate, *, at: Optional[datetime] = None) -> int:
    principal = Decimal(certificate.principal_cents)
    total_days = certificate_term_days(certificate)
    if total_days <= 0:
        return int(principal)
    moment = at or datetime.utcnow()
    if certificate.matured_at:
        moment = max(moment, certificate.matured_at)
    elapsed = max(0, min((moment - certificate.opened_at).days, total_days))
    progress = Decimal(elapsed) / Decimal(total_days) if total_days else Decimal(0)
    rate = Decimal(certificate.rate_bps) / Decimal(10000)
    interest = (principal * rate * progress).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(principal + interest)


def certificate_progress_percent(certificate: Certificate, *, at: Optional[datetime] = None) -> float:
    if certificate.matured_at:
        return 100.0
    total_days = certificate_term_days(certificate)
    if total_days <= 0:
        return 100.0
    moment = at or datetime.utcnow()
    elapsed = max(0, min((moment - certificate.opened_at).days, total_days))
    pct = (elapsed / total_days) * 100
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
# Investing helpers (live S&P 500 with cached fallback)
# ---------------------------------------------------------------------------
def _cache_set_price(session: Session, cents: int) -> None:
    MetaDAO.set(session, "real_sp500_price_cents", str(int(cents)))
    MetaDAO.set(session, "real_sp500_last_ts", datetime.utcnow().isoformat())


def _cache_get_price(session: Session) -> tuple[int, Optional[datetime]]:
    price_raw = MetaDAO.get(session, "real_sp500_price_cents")
    ts_raw = MetaDAO.get(session, "real_sp500_last_ts")
    price_c = int(price_raw) if price_raw and price_raw.isdigit() else 0
    try:
        last = datetime.fromisoformat(ts_raw) if ts_raw else None
    except Exception:
        last = None
    return price_c, last


def _should_refresh(last: Optional[datetime]) -> bool:
    if not last:
        return True
    return (datetime.utcnow() - last) > timedelta(minutes=5)


def _fetch_sp500_from_yahoo() -> Optional[int]:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=1d&interval=5m"
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


def _append_price_history(session: Session, price_c: int, max_len: int = 2016) -> None:
    try:
        raw = MetaDAO.get(session, "real_sp500_hist") or "[]"
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
    MetaDAO.set(session, "real_sp500_hist", json.dumps(history))


def get_price_history() -> list[dict]:
    with Session(engine) as session:
        try:
            raw = MetaDAO.get(session, "real_sp500_hist") or "[]"
            return json.loads(raw)
        except Exception:
            return []


def sp500_update_to_today() -> int:
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


def real_sp500_price_cents() -> Optional[int]:
    with Session(engine) as session:
        cached, last = _cache_get_price(session)
        if cached and not _should_refresh(last):
            return cached
        live = _fetch_sp500_from_yahoo()
        if live and live > 0:
            _cache_set_price(session, live)
            _append_price_history(session, live)
            session.commit()
            return live
        if cached > 0:
            return cached
        return None


def sp500_price_cents() -> int:
    live = real_sp500_price_cents()
    if isinstance(live, int) and live > 0:
        return live
    return sp500_update_to_today()


def compute_holdings_metrics(kid_id: str) -> dict:
    price_c = sp500_price_cents()
    with Session(engine) as session:
        holding = session.exec(
            select(Investment).where(Investment.kid_id == kid_id, Investment.fund == "SP500")
        ).first()
        txs = session.exec(
            select(InvestmentTx)
            .where(InvestmentTx.kid_id == kid_id, InvestmentTx.fund == "SP500")
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
def kid_home(request: Request) -> HTMLResponse:
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    try:
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
            siblings = session.exec(
                select(Child)
                .where(Child.kid_id != kid_id)
                .order_by(Child.name)
            ).all()
            incoming_requests = session.exec(
                select(PeerRequest)
                .where(PeerRequest.target_kid == kid_id)
                .where(PeerRequest.status == "pending")
                .order_by(desc(PeerRequest.created_at))
            ).all()
            outgoing_requests = session.exec(
                select(PeerRequest)
                .where(PeerRequest.requester_kid == kid_id)
                .order_by(desc(PeerRequest.created_at))
                .limit(20)
            ).all()
        event_rows = "".join(
            f"<tr><td data-label='When'>{event.timestamp.strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td data-label='Δ Amount' class='right'>{'+' if event.change_cents>=0 else ''}{usd(event.change_cents)}</td>"
            f"<td data-label='Reason'>{event.reason}</td></tr>"
            for event in events
        ) or "<tr><td>(no events)</td></tr>"
        kid_names = {child.kid_id: child.name}
        for sibling in siblings:
            kid_names[sibling.kid_id] = sibling.name

        def kid_label(kid: str) -> str:
            name = kid_names.get(kid, kid)
            return f"{escape(name)} <span class='muted'>({escape(kid)})</span>"

        def reason_display(text: Optional[str]) -> str:
            val = (text or "").strip()
            if not val:
                return "<span class='muted'>(no reason)</span>"
            return escape(val)

        sibling_options = "".join(
            f"<option value='{escape(s.kid_id)}'>{escape(s.name)} ({escape(s.kid_id)})</option>"
            for s in siblings
        )
        select_prompt = (
            "<option value='' disabled selected>Pick a sibling</option>" + sibling_options
            if sibling_options
            else ""
        )
        if sibling_options:
            send_form = (
                "<form method='post' action='/kid/send'>"
                "<label>Send to</label>"
                f"<select name='to_kid' required>{select_prompt}</select>"
                "<input name='amount' type='text' data-money placeholder='amount $' required>"
                "<textarea name='reason' placeholder='Why are you sending money?' rows='2'></textarea>"
                "<button type='submit' style='margin-top:6px;'>Send Money</button>"
                "</form>"
            )
            request_form = (
                "<form method='post' action='/kid/request_money' style='margin-top:12px;'>"
                "<label>Request from</label>"
                f"<select name='target_kid' required>{select_prompt}</select>"
                "<input name='amount' type='text' data-money placeholder='amount $' required>"
                "<textarea name='reason' placeholder='Why do you need it?' rows='2'></textarea>"
                "<button type='submit' style='margin-top:6px;'>Request Money</button>"
                "</form>"
            )
        else:
            send_form = "<p class='muted'>No other kids are linked yet for transfers.</p>"
            request_form = ""

        incoming_rows = "".join(
            (
                "<tr>"
                f"<td data-label='From'>{kid_label(req.requester_kid)}</td>"
                f"<td data-label='Amount' class='right'>{usd(req.amount_cents)}</td>"
                f"<td data-label='Reason'>{reason_display(req.reason)}</td>"
                f"<td data-label='Requested' class='muted'>{req.created_at.strftime('%Y-%m-%d %H:%M')}</td>"
                "<td data-label='Actions' class='right'>"
                f"<form method='post' action='/kid/request/respond' class='inline'>"
                f"<input type='hidden' name='request_id' value='{req.id}'>"
                "<input type='hidden' name='action' value='fulfill'>"
                "<button type='submit'>Send</button></form> "
                f"<form method='post' action='/kid/request/respond' class='inline' style='margin-left:4px;'>"
                f"<input type='hidden' name='request_id' value='{req.id}'>"
                "<input type='hidden' name='action' value='decline'>"
                "<button type='submit' class='danger'>Decline</button></form>"
                "</td>"
                "</tr>"
            )
            for req in incoming_requests
        ) or "<tr><td colspan='5' class='muted'>(no pending requests)</td></tr>"

        def status_badge(req: PeerRequest) -> str:
            if req.status == "fulfilled":
                return "<span class='pill' style='background:var(--good);'>Fulfilled</span>"
            if req.status == "declined":
                return "<span class='pill' style='background:var(--bad);'>Declined</span>"
            if req.status == "cancelled":
                return "<span class='pill'>Cancelled</span>"
            return "<span class='pill'>Pending</span>"

        outgoing_rows = "".join(
            (
                "<tr>"
                f"<td data-label='To'>{kid_label(req.target_kid)}</td>"
                f"<td data-label='Amount' class='right'>{usd(req.amount_cents)}</td>"
                f"<td data-label='Reason'>{reason_display(req.reason)}</td>"
                f"<td data-label='Status'>{status_badge(req)}</td>"
                f"<td data-label='Requested' class='muted'>{req.created_at.strftime('%Y-%m-%d %H:%M')}</td>"
                "<td data-label='Actions' class='right'>"
                + (
                    f"<form method='post' action='/kid/request/cancel' class='inline'>"
                    f"<input type='hidden' name='request_id' value='{req.id}'>"
                    "<button type='submit'>Cancel</button></form>"
                    if req.status == "pending"
                    else ""
                )
                + "</td>"
                "</tr>"
            )
            for req in outgoing_requests
        ) or "<tr><td colspan='6' class='muted'>(no requests yet)</td></tr>"

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
                window = f"<div class='muted' style='margin-top:4px;'>Active: {chore.start_date or '…'} → {chore.end_date or '…'}</div>"
            chore_cards += (
                f"<div class='card'><div><b>{chore.name}</b> <span class='muted'>({chore.type})</span></div>"
                f"{window}<div style='margin-top:6px;'>{action}</div></div>"
            )
        if not chore_cards:
            chore_cards = "<div class='muted'>(no chores yet)</div>"
        goal_rows = "".join(
            f"<tr><td data-label='Goal'><b>{goal.name}</b>"
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
        investing_card = _safe_investing_card(kid_id)
        peer_card = f"""
          <div class='card'>
            <h3>Send &amp; Request Money</h3>
            <p class='muted'>Share funds with siblings and explain why.</p>
            {send_form}
            {request_form}
            <h4 style='margin-top:12px;'>Requests for Me</h4>
            <table>
              <tr><th>From</th><th>Amount</th><th>Reason</th><th>Requested</th><th>Actions</th></tr>
              {incoming_rows}
            </table>
            <h4 style='margin-top:12px;'>My Requests</h4>
            <table>
              <tr><th>To</th><th>Amount</th><th>Reason</th><th>Status</th><th>Requested</th><th>Actions</th></tr>
              {outgoing_rows}
            </table>
          </div>
        """
        inner = f"""
        <div class='card kiosk'>
          <div>
            <div class='name'>{child.name} <span class='muted'>({child.kid_id})</span></div>
            <div class='muted'>Level {child.level} • Streak {child.streak_days} days • Badges: {_badges_html(child.badges)}</div>
          </div>
          <div class='balance'>{usd(child.balance_cents)}</div>
        </div>
        <div class='grid'>
          <div class='card'>
            <h3>My Chores</h3>
            {chore_cards}
          </div>
          {investing_card}
          {peer_card}
          <div class='card'>
            <h3>My Goals</h3>
            <form method='post' action='/kid/goal_create' class='inline'>
              <input name='name' placeholder='e.g. Lego set' required>
              <input name='target' type='text' data-money placeholder='target $' required>
              <button type='submit'>Create Goal</button>
            </form>
            <table style='margin-top:8px;'><tr><th>Goal</th><th>Saved</th><th>Actions</th></tr>{goal_rows}</table>
            <form method='post' action='/kid/logout' style='margin-top:10px;'><button type='submit'>Logout</button></form>
          </div>
          <div class='card'>
            <h3>Recent Activity</h3>
            <table><tr><th>When</th><th>Δ Amount</th><th>Reason</th></tr>{event_rows}</table>
          </div>
        </div>
        """
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


def _safe_investing_card(kid_id: str) -> str:
    try:
        price_c = sp500_price_cents()
        shares = 0.0
        cd_total_c = 0
        cd_count = 0
        ready_count = 0
        rate_bps = DEFAULT_CD_RATE_BPS
        moment = datetime.utcnow()
        with Session(engine) as session:
            holding = session.exec(
                select(Investment).where(Investment.kid_id == kid_id, Investment.fund == "SP500")
            ).first()
            shares = holding.shares if holding else 0.0
            certificates = session.exec(
                select(Certificate)
                .where(Certificate.kid_id == kid_id)
                .where(Certificate.matured_at == None)  # noqa: E711
                .order_by(desc(Certificate.opened_at))
            ).all()
            rate_bps = get_cd_rate_bps(session)
        if certificates:
            for certificate in certificates:
                cd_total_c += certificate_value_cents(certificate, at=moment)
                if moment >= certificate_maturity_date(certificate):
                    ready_count += 1
            cd_count = len(certificates)
        value_c = int(round(shares * price_c))
        total_c = value_c + cd_total_c
        rate_pct = rate_bps / 100
        if cd_count:
            ready_text = f" • {ready_count} ready" if ready_count else ""
            cd_line = (
                f"Certificates: <b>{usd(cd_total_c)}</b> across {cd_count} active{ready_text}"
            )
        else:
            cd_line = "Certificates: <span class='muted'>none yet</span>"
        return f"""
          <div class='card'>
            <h3>Investing</h3>
            <div class='muted'>Stocks &amp; certificates of deposit</div>
            <div style='margin-top:6px;'>Stocks: <b>{usd(value_c)}</b> ({shares:.4f} sh @ {usd(price_c)})</div>
            <div style='margin-top:4px;'>{cd_line}</div>
            <div class='muted' style='margin-top:4px;'>Total invested: <b>{usd(total_c)}</b> • CD rate {rate_pct:.2f}% APR</div>
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
        pk = "SPECIAL" if chore.type == "special" else period_key_for(chore.type, moment)
        query = select(ChoreInstance).where(ChoreInstance.chore_id == chore.id)
        if chore.type != "special":
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


@app.post("/kid/send")
def kid_send_money(
    request: Request,
    to_kid: str = Form(...),
    amount: str = Form(...),
    reason: str = Form(""),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    target = (to_kid or "").strip()
    if not target or target == kid_id:
        return RedirectResponse("/kid", status_code=302)
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        return RedirectResponse("/kid", status_code=302)
    note = (reason or "").strip()
    with Session(engine) as session:
        sender = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        recipient = session.exec(select(Child).where(Child.kid_id == target)).first()
        if not sender or not recipient:
            return RedirectResponse("/kid", status_code=302)
        if amount_c > sender.balance_cents:
            return RedirectResponse("/kid", status_code=302)
        sender.balance_cents -= amount_c
        recipient.balance_cents += amount_c
        now = datetime.utcnow()
        sender.updated_at = now
        recipient.updated_at = now
        suffix = f" ({note})" if note else ""
        session.add(sender)
        session.add(recipient)
        session.add(
            Event(
                child_id=kid_id,
                change_cents=-amount_c,
                reason=f"peer_transfer_to:{target}{suffix}",
            )
        )
        session.add(
            Event(
                child_id=target,
                change_cents=amount_c,
                reason=f"peer_transfer_from:{kid_id}{suffix}",
            )
        )
        session.commit()
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
    if not target or target == kid_id:
        return RedirectResponse("/kid", status_code=302)
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        return RedirectResponse("/kid", status_code=302)
    note = (reason or "").strip()
    with Session(engine) as session:
        sibling = session.exec(select(Child).where(Child.kid_id == target)).first()
        if not sibling:
            return RedirectResponse("/kid", status_code=302)
        session.add(
            PeerRequest(
                requester_kid=kid_id,
                target_kid=target,
                amount_cents=amount_c,
                reason=note,
            )
        )
        session.commit()
    return RedirectResponse("/kid", status_code=302)


@app.post("/kid/request/respond")
def kid_request_respond(
    request: Request,
    request_id: int = Form(...),
    action: str = Form(...),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    action_name = (action or "").strip().lower()
    with Session(engine) as session:
        peer_request = session.get(PeerRequest, request_id)
        if not peer_request or peer_request.status != "pending" or peer_request.target_kid != kid_id:
            return RedirectResponse("/kid", status_code=302)
        if action_name == "fulfill":
            payer = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
            receiver = session.exec(select(Child).where(Child.kid_id == peer_request.requester_kid)).first()
            if not payer or not receiver or peer_request.amount_cents <= 0:
                return RedirectResponse("/kid", status_code=302)
            if peer_request.amount_cents > payer.balance_cents:
                return RedirectResponse("/kid", status_code=302)
            payer.balance_cents -= peer_request.amount_cents
            receiver.balance_cents += peer_request.amount_cents
            now = datetime.utcnow()
            payer.updated_at = now
            receiver.updated_at = now
            suffix = ""
            note = (peer_request.reason or "").strip()
            if note:
                suffix = f" ({note})"
            session.add(payer)
            session.add(receiver)
            session.add(
                Event(
                    child_id=kid_id,
                    change_cents=-peer_request.amount_cents,
                    reason=f"peer_request_pay:{peer_request.requester_kid}{suffix}",
                )
            )
            session.add(
                Event(
                    child_id=peer_request.requester_kid,
                    change_cents=peer_request.amount_cents,
                    reason=f"peer_request_receive:{kid_id}{suffix}",
                )
            )
            peer_request.status = "fulfilled"
            peer_request.resolved_at = datetime.utcnow()
            peer_request.resolved_by = kid_id
            session.add(peer_request)
            session.commit()
        elif action_name == "decline":
            peer_request.status = "declined"
            peer_request.resolved_at = datetime.utcnow()
            peer_request.resolved_by = kid_id
            session.add(peer_request)
            session.commit()
    return RedirectResponse("/kid", status_code=302)


@app.post("/kid/request/cancel")
def kid_request_cancel(request: Request, request_id: int = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    with Session(engine) as session:
        peer_request = session.get(PeerRequest, request_id)
        if not peer_request or peer_request.status != "pending" or peer_request.requester_kid != kid_id:
            return RedirectResponse("/kid", status_code=302)
        peer_request.status = "cancelled"
        peer_request.resolved_at = datetime.utcnow()
        peer_request.resolved_by = kid_id
        session.add(peer_request)
        session.commit()
    return RedirectResponse("/kid", status_code=302)


@app.get("/kid/invest", response_class=HTMLResponse)
def kid_invest_home(request: Request) -> HTMLResponse:
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    try:
        metrics = compute_holdings_metrics(kid_id)
        history = get_price_history()
        svg = sparkline_svg_from_history(history)
        with Session(engine) as session:
            child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
            balance_c = child.balance_cents if child else 0
            certificates = session.exec(
                select(Certificate)
                .where(Certificate.kid_id == kid_id)
                .order_by(desc(Certificate.opened_at))
            ).all()
            rate_bps = get_cd_rate_bps(session)

        def fmt(value: int) -> str:
            return f"{'+' if value >= 0 else ''}{usd(value)}"

        moment = datetime.utcnow()
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
            progress_text = format_percent(progress_pct)
            cert_rows += (
                f"<tr>"
                f"<td data-label='Principal'>{usd(certificate.principal_cents)}</td>"
                f"<td data-label='Rate'>{rate_display:.2f}%</td>"
                f"<td data-label='Term'>{certificate.term_months} mo</td>"
                f"<td data-label='Value Today' class='right'>{usd(value_c)}</td>"
                f"<td data-label='Progress' class='right'>{progress_text}</td>"
                f"<td data-label='Status'>{status}</td>"
                "</tr>"
            )
        if not cert_rows:
            cert_rows = "<tr><td colspan='6' class='muted'>(no certificates yet)</td></tr>"
        rate_pct_display = (rate_bps / 100) if 'rate_bps' in locals() else (DEFAULT_CD_RATE_BPS / 100)
        summary_bits = [
            f"<div><b>Current rate:</b> {rate_pct_display:.2f}% APR</div>",
            f"<div>Total active value: <b>{usd(cd_total_c)}</b></div>",
        ]
        if next_maturity:
            summary_bits.append(f"<div>Next maturity: <b>{next_maturity:%Y-%m-%d}</b></div>")
        if matured_ready:
            summary_bits.append(
                f"<div class='muted'>{matured_ready} certificate{'s' if matured_ready != 1 else ''} ready to cash out.</div>"
            )
        cd_summary_html = "".join(summary_bits)
        cash_out_form = ""
        if matured_ready:
            cash_out_form = (
                "<form method='post' action='/kid/invest/cd/mature' style='margin-top:10px;'>"
                "<button type='submit'>Cash out matured</button>"
                "</form>"
            )

        inner = f"""
        <div class='card'>
          <h3>Stock Simulator — S&amp;P 500 Fund</h3>
          <p class='muted'>Live S&amp;P 500 price (cached every 5 min) — learning tool only.</p>
          <div style='margin-bottom:12px;'><b>Available Balance:</b> {usd(balance_c)}</div>
          <div class='grid' style='grid-template-columns:1fr 1fr; gap:12px;'>
            <div class='card'>
              <div><b>Current Price</b></div>
              <div style='font-size:28px; font-weight:800; margin-top:6px;'>{usd(metrics['price_c'])}</div>
              <div class='muted'>per share</div>
              <div style='margin-top:8px;'>{svg}</div>
            </div>
            <div class='card'>
              <div><b>Your Holdings</b></div>
              <div style='margin-top:6px;'>Shares: <b>{metrics['shares']:.4f}</b></div>
              <div>Value: <b>{usd(metrics['market_value_c'])}</b></div>
              <div>Avg Cost: <b>{usd(metrics['avg_cost_c'])}</b></div>
              <div>Invested: <b>{usd(metrics['invested_cost_c'])}</b></div>
              <div style='color:#{'16a34a' if metrics['unrealized_pl_c']>=0 else 'dc2626'};'>Unrealized P/L: <b>{fmt(metrics['unrealized_pl_c'])}</b></div>
              <div style='color:#{'16a34a' if metrics['realized_pl_c']>=0 else 'dc2626'};'>Realized P/L: <b>{fmt(metrics['realized_pl_c'])}</b></div>
            </div>
          </div>
          <h4 style='margin-top:12px;'>Buy (deposit from balance)</h4>
          <form method='post' action='/kid/invest/buy' class='inline'>
            <input name='amount' type='text' data-money placeholder='amount $' required>
            <button type='submit'>Buy Shares</button>
          </form>
          <h4 style='margin-top:12px;'>Sell (withdraw to balance)</h4>
          <form method='post' action='/kid/invest/sell' class='inline'>
            <input name='amount' type='text' data-money placeholder='amount $' required>
            <button type='submit' class='danger'>Sell Shares</button>
          </form>
        </div>
        <div class='card'>
          <h3>Certificates of Deposit</h3>
          <p class='muted'>Lock part of your balance to earn interest.</p>
          {cd_summary_html}
          {cash_out_form}
          <h4 style='margin-top:12px;'>Open a certificate</h4>
          <form method='post' action='/kid/invest/cd/open' class='inline'>
            <input name='amount' type='text' data-money placeholder='amount $' required>
            <label style='margin-left:6px;'>Term</label>
            <select name='term_months'>
              <option value='3'>3 months</option>
              <option value='6'>6 months</option>
              <option value='12' selected>12 months</option>
            </select>
            <button type='submit' style='margin-left:6px;'>Lock Savings</button>
          </form>
          <p class='muted' style='margin-top:6px;'>Funds move from your balance into the certificate.</p>
          <table style='margin-top:10px;'><tr><th>Principal</th><th>Rate</th><th>Term</th><th>Value Today</th><th>Progress</th><th>Status</th></tr>{cert_rows}</table>
        </div>
        <p class='muted' style='margin-top:10px;'><a href='/kid'>← Back to My Account</a></p>
        """
        return HTMLResponse(frame("Investing — S&P 500 Simulator", inner))
    except Exception:
        body = """
        <div class='card'>
          <h3>Investing</h3>
          <p class='muted'>The investing dashboard hit an error. Check server logs.</p>
          <a href='/kid'><button>Back</button></a>
        </div>
        """
        return HTMLResponse(frame("Investing — Error", body))

@app.post("/kid/invest/buy")
def kid_invest_buy(request: Request, amount: str = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        return RedirectResponse("/kid/invest", status_code=302)
    price_c = sp500_price_cents()
    price = price_c / 100.0
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child or amount_c > child.balance_cents:
            return RedirectResponse("/kid/invest", status_code=302)
        shares = (amount_c / 100.0) / price
        holding = session.exec(
            select(Investment).where(Investment.kid_id == kid_id, Investment.fund == "SP500")
        ).first()
        if not holding:
            holding = Investment(kid_id=kid_id, fund="SP500", shares=0.0)
        holding.shares += shares
        child.balance_cents -= amount_c
        child.updated_at = datetime.utcnow()
        tx = InvestmentTx(
            kid_id=kid_id,
            fund="SP500",
            tx_type="buy",
            shares=shares,
            price_cents=price_c,
            amount_cents=-amount_c,
            realized_pl_cents=0,
        )
        session.add(holding)
        session.add(child)
        session.add(tx)
        session.add(Event(child_id=kid_id, change_cents=-amount_c, reason=f"invest_buy_sp500:{shares:.4f}sh @ {usd(price_c)}"))
        session.commit()
    return RedirectResponse("/kid/invest", status_code=302)


@app.post("/kid/invest/sell")
def kid_invest_sell(request: Request, amount: str = Form(...)):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    amount_c = to_cents_from_dollars_str(amount, 0)
    if amount_c <= 0:
        return RedirectResponse("/kid/invest", status_code=302)
    price_c = sp500_price_cents()
    price = price_c / 100.0
    with Session(engine) as session:
        holding = session.exec(
            select(Investment).where(Investment.kid_id == kid_id, Investment.fund == "SP500")
        ).first()
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not holding or not child or holding.shares <= 0:
            return RedirectResponse("/kid/invest", status_code=302)
        need_shares = (amount_c / 100.0) / price
        sell_shares = min(holding.shares, need_shares)
        proceeds_c = int(round(sell_shares * price * 100))
        if sell_shares <= 0 or proceeds_c <= 0:
            return RedirectResponse("/kid/invest", status_code=302)
        txs = session.exec(
            select(InvestmentTx)
            .where(InvestmentTx.kid_id == kid_id, InvestmentTx.fund == "SP500")
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
        child.updated_at = datetime.utcnow()
        tx = InvestmentTx(
            kid_id=kid_id,
            fund="SP500",
            tx_type="sell",
            shares=sell_shares,
            price_cents=price_c,
            amount_cents=proceeds_c,
            realized_pl_cents=realized_pl,
        )
        session.add(holding)
        session.add(child)
        session.add(tx)
        session.add(Event(child_id=kid_id, change_cents=proceeds_c, reason=f"invest_sell_sp500:{sell_shares:.4f}sh @ {usd(price_c)}"))
        session.commit()
    return RedirectResponse("/kid/invest", status_code=302)


@app.post("/kid/invest/cd/open")
def kid_invest_cd_open(
    request: Request,
    amount: str = Form(...),
    term_months: int = Form(...),
):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    amount_c = to_cents_from_dollars_str(amount, 0)
    term = max(1, int(term_months))
    if amount_c <= 0:
        return RedirectResponse("/kid/invest", status_code=302)
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child or amount_c > child.balance_cents:
            return RedirectResponse("/kid/invest", status_code=302)
        rate_bps = get_cd_rate_bps(session)
        certificate = Certificate(
            kid_id=kid_id,
            principal_cents=amount_c,
            rate_bps=rate_bps,
            term_months=term,
            opened_at=datetime.utcnow(),
        )
        child.balance_cents -= amount_c
        child.updated_at = datetime.utcnow()
        rate_pct = rate_bps / 100
        session.add(child)
        session.add(certificate)
        session.add(
            Event(
                child_id=kid_id,
                change_cents=-amount_c,
                reason=f"invest_cd_open:{term}m @ {rate_pct:.2f}%",
            )
        )
        session.commit()
    return RedirectResponse("/kid/invest", status_code=302)


@app.post("/kid/invest/cd/mature")
def kid_invest_cd_mature(request: Request):
    if (redirect := require_kid(request)) is not None:
        return redirect
    kid_id = kid_authed(request)
    assert kid_id
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return RedirectResponse("/kid/invest", status_code=302)
        certificates = session.exec(
            select(Certificate)
            .where(Certificate.kid_id == kid_id)
            .where(Certificate.matured_at == None)  # noqa: E711
        ).all()
        moment = datetime.utcnow()
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
            child.updated_at = datetime.utcnow()
            session.add(child)
            session.add(
                Event(
                    child_id=kid_id,
                    change_cents=payout_total,
                    reason=f"invest_cd_mature:{matured_count}x",
                )
            )
            session.commit()
        else:
            session.rollback()
    return RedirectResponse("/kid/invest", status_code=302)


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
    role = None
    if pin == MOM_PIN:
        role = "mom"
    elif pin == DAD_PIN:
        role = "dad"
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
    return f"<span class='pill'>Signed in as {role.title()}</span>"


def _kid_options(kids: Iterable[Child]) -> str:
    return "".join(f"<option value='{kid.kid_id}'>{kid.name} ({kid.kid_id})</option>" for kid in kids)


@app.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request):
    if (redirect := require_admin(request)) is not None:
        return redirect
    run_weekly_allowance_if_needed()
    role = admin_role(request)
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
        cd_rate_bps = get_cd_rate_bps(session)
        active_certs = session.exec(
            select(Certificate).where(Certificate.matured_at == None)  # noqa: E711
        ).all()
    kids_rows = "".join(
        f"<tr>"
        f"<td data-label='Child'><b>{child.name}</b><div class='muted'>{child.kid_id} • Level {child.level} • Streak {child.streak_days} • {_badges_html(child.badges)}</div></td>"
        f"<td data-label='Balance' class='right'><b>{usd(child.balance_cents)}</b></td>"
        f"<td data-label='Actions' class='right'>"
        f"<a href='/admin/kiosk?kid_id={child.kid_id}'><button type='button'>Kiosk</button></a> "
        f"<a href='/admin/kiosk_full?kid_id={child.kid_id}' style='margin-left:6px;'><button type='button'>Kiosk (auto)</button></a> "
        f"<a href='/admin/chores?kid_id={child.kid_id}' style='margin-left:6px;'><button type='button'>Manage Chores</button></a> "
        f"<a href='/admin/goals?kid_id={child.kid_id}' style='margin-left:6px;'><button type='button'>Goals</button></a> "
        f"<a href='/admin/statement?kid_id={child.kid_id}' style='margin-left:6px;'><button type='button'>Statement</button></a> "
        f"<form class='inline' method='post' action='/admin/set_allowance' style='margin-left:6px;'>"
        f"<input type='hidden' name='kid_id' value='{child.kid_id}'>"
        f"<input name='allowance' type='text' data-money value='{dollars_value(child.allowance_cents)}' style='max-width:130px' placeholder='allowance $'>"
        f"<button type='submit'>Save</button></form> "
        f"<form class='inline' method='post' action='/admin/set_kid_pin' style='margin-left:6px;'>"
        f"<input type='hidden' name='kid_id' value='{child.kid_id}'>"
        f"<input name='new_pin' placeholder='kid PIN' style='max-width:130px;'>"
        f"<button type='submit'>Set PIN</button></form> "
        f"<form class='inline' method='post' action='/delete_kid' onsubmit=\"return confirm('Delete kid and all events?');\" style='margin-left:6px;'>"
        f"<input type='hidden' name='kid_id' value='{child.kid_id}'>"
        f"<input name='pin' placeholder='parent PIN' style='max-width:110px;'>"
        f"<button type='submit' class='danger'>Delete</button></form>"
        f"</td></tr>"
        for child in kids
    ) or "<tr><td>(no kids yet)</td></tr>"
    prize_rows = "".join(
        f"<tr><td data-label='Prize'><b>{prize.name}</b><div class='muted'>{prize.notes or ''}</div></td>"
        f"<td data-label='Cost' class='right'>{usd(prize.cost_cents)}</td>"
        f"<td data-label='Actions' class='right'>"
        f"<form class='inline' method='post' action='/delete_prize' onsubmit=\"return confirm('Delete this prize?');\">"
        f"<input type='hidden' name='prize_id' value='{prize.id}'><button type='submit' class='danger'>Delete</button></form>"
        f"</td></tr>"
        for prize in prizes
    ) or "<tr><td>(no prizes yet)</td></tr>"
    event_rows = "".join(
        f"<tr><td data-label='When'>{event.timestamp.strftime('%Y-%m-%d %H:%M')}</td>"
        f"<td data-label='Kid'>{event.child_id}</td>"
        f"<td data-label='Δ Amount' class='right'>{'+' if event.change_cents>=0 else ''}{usd(event.change_cents)}</td>"
        f"<td data-label='Reason'>{event.reason}</td></tr>"
        for event in events
    ) or "<tr><td>(no events yet)</td></tr>"
    pending_rows = "".join(
        f"<tr>"
        f"<td data-label='Kid'><b>{child.name}</b><div class='muted'>{child.kid_id}</div></td>"
        f"<td data-label='Chore'><b>{chore.name}</b><div class='muted'>{chore.type}</div></td>"
        f"<td data-label='Award' class='right'><b>{usd(chore.award_cents)}</b></td>"
        f"<td data-label='Completed'>{inst.completed_at.strftime('%Y-%m-%d %H:%M') if inst.completed_at else ''}</td>"
        f"<td data-label='Actions' class='right'>"
        f"<form class='inline' method='post' action='/admin/chore_payout'>"
        f"<input type='hidden' name='instance_id' value='{inst.id}'>"
        f"<input name='amount' type='text' data-money placeholder='override $ (optional)' style='max-width:150px'>"
        f"<input name='reason' type='text' placeholder='reason (optional)' style='max-width:200px'>"
        f"<button type='submit'>Payout</button></form> "
        f"<form class='inline' method='post' action='/admin/chore_deny' style='margin-left:6px;' onsubmit=\"return confirm('Deny and push back to Available?');\">"
        f"<input type='hidden' name='instance_id' value='{inst.id}'><button type='submit' class='danger'>Deny</button></form>"
        f"</td></tr>"
        for inst, chore, child in pending
    ) or "<tr><td>(no pending)</td></tr>"
    goals_rows = "".join(
        f"<tr>"
        f"<td data-label='Kid'><b>{child.name}</b><div class='muted'>{child.kid_id}</div></td>"
        f"<td data-label='Goal'><b>{goal.name}</b></td>"
        f"<td data-label='Saved' class='right'>{usd(goal.saved_cents)} / {usd(goal.target_cents)}"
        f"<div class='muted'>{format_percent(percent_complete(goal.saved_cents, goal.target_cents))} complete</div></td>"
        f"<td data-label='Actions' class='right'>"
        f"<form class='inline' method='post' action='/admin/goal_grant'>"
        f"<input type='hidden' name='goal_id' value='{goal.id}'><button type='submit'>Grant Goal</button></form> "
        f"<form class='inline' method='post' action='/admin/goal_return_funds' style='margin-left:6px;'>"
        f"<input type='hidden' name='goal_id' value='{goal.id}'><button type='submit' class='danger'>Return Funds</button></form>"
        f"</td></tr>"
        for goal, child in needs
    ) or "<tr><td>(none)</td></tr>"
    moment_admin = datetime.utcnow()
    active_cd_total = sum(certificate_value_cents(cert, at=moment_admin) for cert in active_certs)
    active_cd_count = len(active_certs)
    ready_cd = sum(1 for cert in active_certs if moment_admin >= certificate_maturity_date(cert))
    cd_rate_pct = cd_rate_bps / 100
    ready_note = (
        f"<div class='muted' style='margin-top:4px;'>{ready_cd} certificate{'s' if ready_cd != 1 else ''} ready to cash out.</div>"
        if ready_cd
        else "<div class='muted' style='margin-top:4px;'>Kids manage certificates from their investing page.</div>"
    )
    goals_card = f"""
    <div class='card'>
      <h3>Goals Needing Action</h3>
      <table><tr><th>Kid</th><th>Goal</th><th>Saved</th><th>Actions</th></tr>{goals_rows}</table>
      <p class='muted' style='margin-top:6px;'>When a goal is fully funded, approve the purchase (Grant) or return funds.</p>
    </div>
    """
    investing_card = f"""
    <div class='card'>
      <h3>Investing Controls</h3>
      <div><b>Current CD rate:</b> {cd_rate_pct:.2f}% APR</div>
      <div>Active certificates: <b>{active_cd_count}</b> worth <b>{usd(active_cd_total)}</b></div>
      {ready_note}
      <form method='post' action='/admin/certificates/rate' style='margin-top:10px;'>
        <label>Set CD rate (% APR)</label>
        <input name='rate' type='number' step='0.01' min='0' value='{cd_rate_pct:.2f}' required>
        <button type='submit' style='margin-top:8px;'>Save Rate</button>
      </form>
    </div>
    """
    rules_card = f"""
    <div class='card'>
      <h3>Allowance Rules</h3>
      <form method='post' action='/admin/rules'>
        <div class='grid' style='grid-template-columns:1fr 1fr; gap:8px;'>
          <div>
            <label><input type='checkbox' name='bonus_all' {'checked' if bonus_on_all else ''}> Bonus if all chores complete</label>
            <input name='bonus' type='text' data-money value='{dollars_value(bonus_cents)}' placeholder='bonus $' style='margin-top:6px;'>
          </div>
          <div>
            <label><input type='checkbox' name='penalty_miss' {'checked' if penalty_on_miss else ''}> Penalty if chores missed</label>
            <input name='penalty' type='text' data-money value='{dollars_value(penalty_cents)}' placeholder='penalty $' style='margin-top:6px;'>
          </div>
        </div>
        <button type='submit' style='margin-top:10px;'>Save Rules</button>
      </form>
      <p class='muted' style='margin-top:6px;'>Rules apply when weekly allowance runs (first admin view each Sunday).</p>
    </div>
    """
    inner = f"""
    <div class='topbar'><h3>Admin Portal</h3>
      <div>
        {_role_badge(role)}
        <form method='post' action='/admin/logout' style='display:inline-block; margin-left:8px;'><button type='submit' class='pill'>Logout</button></form>
      </div>
    </div>
    <div class='grid admin-top'>
      <div class='card'>
        <h3>Create Kid</h3>
        <form method='post' action='/create_kid'>
          <label>kid_id</label><input name='kid_id' placeholder='alex01' required>
          <label style='margin-top:6px;'>Name</label><input name='name' placeholder='Alex' required>
          <div class='grid' style='grid-template-columns:1fr 1fr; gap:8px;'>
            <div><label>Starting (dollars)</label><input name='starting' type='text' data-money value='0.00'></div>
            <div><label>Allowance (dollars / week)</label><input name='allowance' type='text' data-money value='0.00'></div>
          </div>
          <label style='margin-top:6px;'>Set kid PIN (optional)</label><input name='kid_pin' placeholder='e.g. 4321'>
          <button type='submit' style='margin-top:10px;'>Create</button>
        </form>
      </div>
      <div class='card'>
        <h3>Credit / Debit</h3>
        <form method='post' action='/adjust_balance'>
          <label>kid_id</label>
          <select name='kid_id' required>{_kid_options(kids)}</select>
          <div class='grid' style='grid-template-columns:1fr 1fr; gap:8px; margin-top:6px;'>
            <div><label>Amount (dollars)</label><input name='amount' type='text' data-money value='1.00'></div>
            <div><label>Type</label><select name='kind'><option value='credit'>Credit (chore)</option><option value='debit'>Debit (redeem)</option></select></div>
          </div>
          <label style='margin-top:6px;'>Reason</label><input name='reason' placeholder='chore / redeem'>
          <button type='submit' style='margin-top:10px;'>Apply</button>
        </form>
      </div>
      <div class='card'>
        <h3>Family Transfer</h3>
        <form method='post' action='/admin/transfer'>
          <label>From</label>
          <select name='from_kid' required>{_kid_options(kids)}</select>
          <label style='margin-top:6px;'>To</label>
          <select name='to_kid' required>{_kid_options(kids)}</select>
          <label style='margin-top:6px;'>Amount (dollars)</label><input name='amount' type='text' data-money value='1.00'>
          <label style='margin-top:6px;'>Note</label><input name='note' placeholder='optional note'>
          <button type='submit' style='margin-top:10px;'>Transfer</button>
        </form>
      </div>
      <div class='card'>
        <h3>Prize Catalog</h3>
        <form method='post' action='/add_prize'>
          <label>Name</label><input name='name' placeholder='Ice cream' required>
          <label style='margin-top:6px;'>Cost (dollars)</label><input name='cost' type='text' data-money value='1.00'>
          <label style='margin-top:6px;'>Notes</label><input name='notes' placeholder='One serving'>
          <button type='submit' style='margin-top:10px;'>Add Prize</button>
        </form>
      </div>
      {rules_card}
    </div>
    <div class='grid'>
      {goals_card}
      {investing_card}
      <div class='card'>
        <h3>Chores</h3>
        <form method='post' action='/admin/chores/create'>
          <label>kid_id</label>
          <select name='kid_id' required>{_kid_options(kids)}</select>
          <label style='margin-top:6px;'>Name</label><input name='name' placeholder='Take out trash' required>
          <div class='grid' style='grid-template-columns:1fr 1fr; gap:8px; margin-top:6px;'>
            <div><label>Type</label><select name='type'><option value='daily'>Daily</option><option value='weekly'>Weekly</option><option value='special'>Special</option></select></div>
            <div><label>Award (dollars)</label><input name='award' type='text' data-money value='0.50'></div>
          </div>
          <div class='grid' style='grid-template-columns:1fr 1fr; gap:8px; margin-top:6px;'>
            <div><label>Start Date (optional)</label><input name='start_date' type='date'></div>
            <div><label>End Date (optional)</label><input name='end_date' type='date'></div>
          </div>
          <label style='margin-top:6px;'>Notes</label><input name='notes' placeholder='Any details'>
          <button type='submit' style='margin-top:10px;'>Add Chore</button>
        </form>
      </div>
    </div>
    <div class='card' id='pending'>
      <h3>Pending Payouts</h3>
      <table><tr><th>Kid</th><th>Chore</th><th>Award</th><th>Completed</th><th>Actions</th></tr>{pending_rows}</table>
      <p class='muted' style='margin-top:6px;'>Audit: <a href='/admin/audit'>Pending vs Paid</a></p>
    </div>
    <div class='card'>
      <h3>Children</h3>
      <table><tr><th>Child</th><th>Balance</th><th>Actions</th></tr>{kids_rows}</table>
    </div>
    <div class='card'>
      <h3>Prizes</h3>
      <table><tr><th>Prize</th><th>Cost</th><th>Actions</th></tr>{prize_rows}</table>
    </div>
    <div class='card'>
      <h3>Recent Events</h3>
      <p class='muted'>Need a CSV? <a href='/admin/ledger.csv'>Download ledger</a></p>
      <table><tr><th>When</th><th>Kid</th><th>Δ Amount</th><th>Reason</th></tr>{event_rows}</table>
    </div>
    """
    return HTMLResponse(frame("Admin", inner))

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
    with Session(engine) as session:
        child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        if not child:
            return HTMLResponse(frame("Chores", "<div class='card'>Kid not found.</div>"))
        chores = session.exec(select(Chore).where(Chore.kid_id == kid_id).order_by(desc(Chore.created_at))).all()
    rows = "".join(
        f"<tr>"
        f"<td data-label='Name'><form class='inline' method='post' action='/admin/chores/update'>"
        f"<input type='hidden' name='chore_id' value='{chore.id}'>"
        f"<input name='name' value='{chore.name}'></td>"
        f"<td data-label='Type'><select name='type'>"
        f"<option value='daily' {'selected' if chore.type=='daily' else ''}>daily</option>"
        f"<option value='weekly' {'selected' if chore.type=='weekly' else ''}>weekly</option>"
        f"<option value='special' {'selected' if chore.type=='special' else ''}>special</option>"
        f"</select></td>"
        f"<td data-label='Award ($)' class='right'><input name='award' type='text' data-money value='{dollars_value(chore.award_cents)}' style='max-width:120px'></td>"
        f"<td data-label='Window'>"
        f"<input name='start_date' type='date' value='{chore.start_date or ''}' style='max-width:160px'>"
        f"<input name='end_date' type='date' value='{chore.end_date or ''}' style='max-width:160px; margin-left:6px;'>"
        f"</td>"
        f"<td data-label='Notes'><input name='notes' value='{chore.notes or ''}'></td>"
        f"<td data-label='Status'><span class='pill'>{'Active' if chore.active else 'Inactive'}</span></td>"
        f"<td data-label='Actions' class='right'>"
        f"<button type='submit'>Save</button></form> "
        f"<form class='inline' method='post' action='/admin/chore_make_available_now' style='margin-left:6px;'>"
        f"<input type='hidden' name='chore_id' value='{chore.id}'><button type='submit'>Make Available Now</button></form> "
        + (
            f"<form class='inline' method='post' action='/admin/chores/deactivate' style='margin-left:6px;'>"
            f"<input type='hidden' name='chore_id' value='{chore.id}'><button type='submit' class='danger'>Deactivate</button></form>"
            if chore.active
            else
            f"<form class='inline' method='post' action='/admin/chores/activate' style='margin-left:6px;'>"
            f"<input type='hidden' name='chore_id' value='{chore.id}'><button type='submit'>Activate</button></form>"
        )
        + "</td></tr>"
        for chore in chores
    ) or "<tr><td>(no chores yet)</td></tr>"
    inner = f"""
    <div class='topbar'><h3>Manage Chores — {child.name} <span class='pill' style='margin-left:8px;'>{child.kid_id}</span></h3>
      <a href='/admin'><button>Back</button></a>
    </div>
    <div class='card'>
      <table>
        <tr><th>Name</th><th>Type</th><th>Award ($)</th><th>Window</th><th>Notes</th><th>Status</th><th>Actions</th></tr>
        {rows}
      </table>
      <p class='muted' style='margin-top:6px;'>“Make Available Now” republishes the chore for the current period (within its active window).</p>
    </div>
    """
    return HTMLResponse(frame("Manage Chores", inner))


@app.post("/admin/chores/create")
def admin_chore_create(request: Request, kid_id: str = Form(...), name: str = Form(...), type: str = Form(...), award: str = Form(...), start_date: Optional[str] = Form(None), end_date: Optional[str] = Form(None), notes: str = Form("")):
    if (redirect := require_admin(request)) is not None:
        return redirect
    award_c = to_cents_from_dollars_str(award, 0)
    with Session(engine) as session:
        chore = Chore(
            kid_id=kid_id,
            name=name.strip(),
            type=type,
            award_cents=award_c,
            notes=notes.strip() or None,
            start_date=date.fromisoformat(start_date) if start_date else None,
            end_date=date.fromisoformat(end_date) if end_date else None,
        )
        session.add(chore)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={kid_id}", status_code=302)


@app.post("/admin/chores/update")
def admin_chore_update(request: Request, chore_id: int = Form(...), name: str = Form(...), type: str = Form(...), award: str = Form(...), start_date: Optional[str] = Form(None), end_date: Optional[str] = Form(None), notes: str = Form("")):
    if (redirect := require_admin(request)) is not None:
        return redirect
    award_c = to_cents_from_dollars_str(award, 0)
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return RedirectResponse("/admin", status_code=302)
        chore.name = name.strip()
        chore.type = type
        chore.award_cents = award_c
        chore.notes = notes.strip() or None
        chore.start_date = date.fromisoformat(start_date) if start_date else None
        chore.end_date = date.fromisoformat(end_date) if end_date else None
        session.add(chore)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={chore.kid_id}", status_code=302)


@app.post("/admin/chores/activate")
def admin_chore_activate(request: Request, chore_id: int = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return RedirectResponse("/admin", status_code=302)
        chore.active = True
        session.add(chore)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={chore.kid_id}", status_code=302)


@app.post("/admin/chores/deactivate")
def admin_chore_deactivate(request: Request, chore_id: int = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    with Session(engine) as session:
        chore = session.get(Chore, chore_id)
        if not chore:
            return RedirectResponse("/admin", status_code=302)
        chore.active = False
        session.add(chore)
        session.commit()
    return RedirectResponse(f"/admin/chores?kid_id={chore.kid_id}", status_code=302)


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
        pk = "SPECIAL" if chore.type == "special" else period_key_for(chore.type, moment)
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
        cd_rate_bps = get_cd_rate_bps(session)
    metrics = compute_holdings_metrics(kid_id)
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
            f"<td data-label='Term'>{certificate.term_months} mo</td>"
            f"<td data-label='Value' class='right'>{usd(value_c)}</td>"
            f"<td data-label='Progress' class='right'>{format_percent(progress_pct)}</td>"
            f"<td data-label='Status'>{status}</td>"
            "</tr>"
        )
    if not cert_rows:
        cert_rows = "<tr><td colspan='6' class='muted'>(no certificates)</td></tr>"
    cd_rate_pct = cd_rate_bps / 100
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
      <div>Certificates: <b>{usd(active_cd_total)}</b> across {active_cd_count} active • Rate {cd_rate_pct:.2f}% APR</div>
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
    if pin not in {MOM_PIN, DAD_PIN}:
        body = "<div class='card'><p style='color:#ff6b6b;'>Incorrect parent PIN.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body))
    with Session(engine) as session:
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
            body = "<div class='card'><p style='color:#ff6b6b;'>Enter an override amount before approving the payout.</p><p><a href='/admin'>Back to admin</a></p></div>"
            return HTMLResponse(frame("Admin", body), status_code=400)
        payout_c = to_cents_from_dollars_str(raw_amount, chore.award_cents)
        payout_c = max(0, payout_c)
        child.balance_cents += payout_c
        child.updated_at = datetime.utcnow()
        _update_gamification(child, payout_c)
        reason_text = f"chore:{chore.name}" + (f" ({reason.strip()})" if reason.strip() else "")
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
def admin_set_certificate_rate(request: Request, rate: str = Form(...)):
    if (redirect := require_admin(request)) is not None:
        return redirect
    raw = (rate or "").strip()
    try:
        rate_value = float(raw)
    except ValueError:
        body = "<div class='card'><p style='color:#ff6b6b;'>Enter a numeric rate in percent.</p><p><a href='/admin'>Back</a></p></div>"
        return HTMLResponse(frame("Admin", body), status_code=400)
    rate_bps = max(0, int(round(rate_value * 100)))
    with Session(engine) as session:
        MetaDAO.set(session, CD_RATE_KEY, str(rate_bps))
        session.commit()
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
