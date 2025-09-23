import asyncio
import importlib
import sys
import types
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

import pytest

pytest.importorskip("sqlmodel")
from sqlmodel import SQLModel, Session, select


def ensure_fastapi_stub() -> None:
    if "fastapi" in sys.modules:
        return
    try:
        import fastapi  # type: ignore  # noqa: F401
        import starlette  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        fake_fastapi = types.ModuleType("fastapi")

        class FakeResponse:
            def __init__(self, content=None, status_code: int = 200, headers=None, media_type=None):
                self.content = content
                self.status_code = status_code
                self.headers = headers or {}
                self.media_type = media_type

        class FastAPI:
            def __init__(self, *_, **__):
                pass

            def add_middleware(self, *_args, **_kwargs) -> None:
                return None

            def get(self, *_args, **_kwargs):
                def decorator(func):
                    return func

                return decorator

            def post(self, *_args, **_kwargs):
                def decorator(func):
                    return func

                return decorator

        def Form(default=None, **_):
            return default

        def Query(default=None, **_):
            return default

        class Request:  # minimal placeholder for typing
            pass

        fake_fastapi.FastAPI = FastAPI
        fake_fastapi.Form = Form
        fake_fastapi.Query = Query
        fake_fastapi.Request = Request

        responses_module = types.ModuleType("fastapi.responses")
        responses_module.HTMLResponse = FakeResponse
        responses_module.RedirectResponse = FakeResponse
        responses_module.StreamingResponse = FakeResponse

        sys.modules["fastapi"] = fake_fastapi
        sys.modules["fastapi.responses"] = responses_module

        starlette_module = types.ModuleType("starlette")
        middleware_module = types.ModuleType("starlette.middleware")
        sessions_module = types.ModuleType("starlette.middleware.sessions")

        class SessionMiddleware:
            def __init__(self, app, **_):
                self.app = app

        sessions_module.SessionMiddleware = SessionMiddleware
        middleware_module.sessions = sessions_module
        starlette_module.middleware = middleware_module

        sys.modules["starlette"] = starlette_module
        sys.modules["starlette.middleware"] = middleware_module
        sys.modules["starlette.middleware.sessions"] = sessions_module


class DummyRequest:
    def __init__(self, session=None, form_data=None):
        self.session = session or {}
        self._form_data = form_data

    async def form(self):
        return self._form_data


class SimpleForm:
    def __init__(self, pairs):
        self._values = {}
        for key, value in pairs:
            self._values.setdefault(key, []).append(value)

    def get(self, key, default=None):
        values = self._values.get(key)
        if not values:
            return default
        return values[-1]

    def getlist(self, key):
        return list(self._values.get(key, []))


@pytest.fixture()
def webapp_env(tmp_path, monkeypatch):
    ensure_fastapi_stub()
    db_path = tmp_path / "webapp.db"
    monkeypatch.setenv("KIDBANK_SQLITE", str(db_path))
    import kidbank.webapp as webapp

    importlib.reload(webapp)
    SQLModel.metadata.drop_all(webapp.engine)
    SQLModel.metadata.create_all(webapp.engine)
    yield webapp


def test_kid_global_chore_claim_cancel(webapp_env) -> None:
    webapp = webapp_env
    with Session(webapp.engine) as session:
        session.add(webapp.Child(kid_id="alex", name="Alex"))
        chore = webapp.GlobalChore(name="Car Wash", reward_cents=1000, max_claims=2)
        session.add(chore)
        session.commit()
        session.refresh(chore)
        chore_id = chore.id
    request = DummyRequest(session={"kid_authed": "alex"})
    webapp.kid_global_claim(request, chore_id=chore_id, comment="I helped")
    with Session(webapp.engine) as session:
        claim = session.exec(
            select(webapp.GlobalChoreClaim).where(webapp.GlobalChoreClaim.kid_id == "alex")
        ).first()
        assert claim is not None and claim.status == "pending"
        claim_id = claim.id
    cancel_request = DummyRequest(session={"kid_authed": "alex"})
    webapp.kid_global_claim_cancel(cancel_request, claim_id=claim_id)
    with Session(webapp.engine) as session:
        updated = session.get(webapp.GlobalChoreClaim, claim_id)
        assert updated is not None and updated.status == "cancelled"


def test_admin_global_chore_approval_and_denial(webapp_env) -> None:
    webapp = webapp_env
    with Session(webapp.engine) as session:
        session.add(webapp.Child(kid_id="alex", name="Alex"))
        session.add(webapp.Child(kid_id="ben", name="Ben"))
        chore = webapp.GlobalChore(name="Car Wash", reward_cents=1000, max_claims=2)
        session.add(chore)
        session.commit()
        session.refresh(chore)
        claim1 = webapp.GlobalChoreClaim(chore_id=chore.id, kid_id="alex")
        claim2 = webapp.GlobalChoreClaim(chore_id=chore.id, kid_id="ben")
        session.add(claim1)
        session.add(claim2)
        session.commit()
        session.refresh(claim1)
        session.refresh(claim2)
        chore_id = chore.id
        claim_ids = [claim1.id, claim2.id]
    form = SimpleForm([("chore_id", str(chore_id))] + [("claim_id", str(cid)) for cid in claim_ids])
    approve_request = DummyRequest(session={"admin_role": "mom"}, form_data=form)
    asyncio.run(webapp.admin_global_chores_approve(approve_request))
    with Session(webapp.engine) as session:
        approved = session.exec(
            select(webapp.GlobalChoreClaim)
            .where(webapp.GlobalChoreClaim.status == "approved")
            .order_by(webapp.GlobalChoreClaim.id)
        ).all()
        assert [claim.awarded_amount_cents for claim in approved] == [500, 500]
        stored_chore = session.get(webapp.GlobalChore, chore_id)
        assert stored_chore is not None and stored_chore.active is False
        alex = session.exec(select(webapp.Child).where(webapp.Child.kid_id == "alex")).first()
        ben = session.exec(select(webapp.Child).where(webapp.Child.kid_id == "ben")).first()
        assert alex.balance_cents == 500
        assert ben.balance_cents == 500
    with Session(webapp.engine) as session:
        chore2 = webapp.GlobalChore(name="Garage", reward_cents=500, max_claims=1)
        session.add(chore2)
        session.commit()
        session.refresh(chore2)
        claim3 = webapp.GlobalChoreClaim(chore_id=chore2.id, kid_id="alex")
        session.add(claim3)
        session.commit()
        session.refresh(claim3)
        claim3_id = claim3.id
    deny_request = DummyRequest(session={"admin_role": "mom"})
    webapp.admin_global_chores_deny(deny_request, claim_id=claim3_id)
    with Session(webapp.engine) as session:
        denied = session.get(webapp.GlobalChoreClaim, claim3_id)
        assert denied is not None and denied.status == "denied"
        assert denied.resolved_at is not None


def test_kid_transfers_and_money_requests(webapp_env) -> None:
    webapp = webapp_env
    with Session(webapp.engine) as session:
        session.add(webapp.Child(kid_id="alex", name="Alex", balance_cents=2000))
        session.add(webapp.Child(kid_id="ben", name="Ben", balance_cents=3000))
        session.commit()
    kid_request = DummyRequest(session={"kid_authed": "alex"})
    webapp.kid_transfer_send(kid_request, to_kid="ben", amount="5.00", comment="snacks")
    with Session(webapp.engine) as session:
        alex = session.exec(select(webapp.Child).where(webapp.Child.kid_id == "alex")).first()
        ben = session.exec(select(webapp.Child).where(webapp.Child.kid_id == "ben")).first()
        assert alex.balance_cents == 1500
        assert ben.balance_cents == 3500
    webapp.kid_transfer_request(kid_request, from_kid="ben", amount="4.00", comment="lunch")
    with Session(webapp.engine) as session:
        pending = session.exec(select(webapp.MoneyRequest).where(webapp.MoneyRequest.requester_kid_id == "alex"))
        req = pending.first()
        assert req is not None and req.status == "pending"
        request_id = req.id
    webapp.kid_transfer_request_cancel(DummyRequest(session={"kid_authed": "alex"}), request_id=request_id)
    with Session(webapp.engine) as session:
        cancelled = session.get(webapp.MoneyRequest, request_id)
        assert cancelled is not None and cancelled.status == "cancelled"
    webapp.kid_transfer_request(kid_request, from_kid="ben", amount="3.50", comment="chips")
    with Session(webapp.engine) as session:
        pending = session.exec(select(webapp.MoneyRequest).where(webapp.MoneyRequest.status == "pending"))
        req = pending.first()
        assert req is not None
        approve_id = req.id
    admin_request = DummyRequest(session={"admin_role": "mom"})
    webapp.admin_money_request_approve(admin_request, request_id=approve_id)
    with Session(webapp.engine) as session:
        approved = session.get(webapp.MoneyRequest, approve_id)
        assert approved is not None and approved.status == "approved"
        alex = session.exec(select(webapp.Child).where(webapp.Child.kid_id == "alex")).first()
        ben = session.exec(select(webapp.Child).where(webapp.Child.kid_id == "ben")).first()
        assert alex.balance_cents == 1850
        assert ben.balance_cents == 3150
    webapp.kid_transfer_request(kid_request, from_kid="ben", amount="2.00", comment="")
    with Session(webapp.engine) as session:
        pending = session.exec(select(webapp.MoneyRequest).where(webapp.MoneyRequest.status == "pending"))
        req = pending.first()
        assert req is not None
        deny_id = req.id
    webapp.admin_money_request_deny(admin_request, request_id=deny_id)
    with Session(webapp.engine) as session:
        denied = session.get(webapp.MoneyRequest, deny_id)
        assert denied is not None and denied.status == "denied"


def test_certificate_rates_penalties_and_withdrawal(webapp_env) -> None:
    webapp = webapp_env
    with Session(webapp.engine) as session:
        session.add(webapp.Child(kid_id="alex", name="Alex", balance_cents=20000))
        session.commit()
    admin_request = DummyRequest(session={"admin_role": "mom"})
    webapp.admin_set_certificate_rate(admin_request, rate="5.00", term=6)
    webapp.admin_set_certificate_penalty(admin_request, term="6", days="30")
    with Session(webapp.engine) as session:
        assert webapp.get_cd_rate_bps(session, 6) == 500
        assert webapp.get_cd_penalty_days(session, 6) == 30
    kid_request = DummyRequest(session={"kid_authed": "alex"})
    webapp.kid_invest_cd_open(kid_request, amount="100.00", term_months=6)
    with Session(webapp.engine) as session:
        cert = session.exec(select(webapp.Certificate).where(webapp.Certificate.kid_id == "alex")).first()
        assert cert is not None
        cert.opened_at = datetime.utcnow() - timedelta(days=60)
        session.add(cert)
        session.commit()
    with Session(webapp.engine) as session:
        cert = session.exec(select(webapp.Certificate).where(webapp.Certificate.kid_id == "alex")).first()
        assert cert is not None
        moment = datetime.utcnow()
        expected_value = webapp.certificate_value_cents(cert, at=moment)
        total_days = webapp.certificate_term_days(cert)
        penalty_days = min(cert.penalty_days, total_days)
        principal = Decimal(cert.principal_cents)
        rate = Decimal(cert.rate_bps) / Decimal(10000)
        daily_interest = (principal * rate) / Decimal(total_days)
        penalty = (daily_interest * Decimal(penalty_days)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        accrued = Decimal(max(0, expected_value - cert.principal_cents))
        expected_penalty = int(min(penalty, accrued))
    webapp.kid_invest_cd_withdraw(kid_request, certificate_id=cert.id)
    with Session(webapp.engine) as session:
        child = session.exec(select(webapp.Child).where(webapp.Child.kid_id == "alex")).first()
        assert child is not None
        assert child.balance_cents == 10000 + expected_value - expected_penalty
        stored_cert = session.get(webapp.Certificate, cert.id)
        assert stored_cert is not None and stored_cert.status == "early"
        events = session.exec(select(webapp.Event).where(webapp.Event.child_id == "alex")).all()
        assert any("invest_cd_withdraw" in event.reason and event.change_cents == expected_value for event in events)
        assert any("invest_cd_penalty" in event.reason and event.change_cents == -expected_penalty for event in events)
