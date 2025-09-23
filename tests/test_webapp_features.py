import sys
import types
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, delete, select, desc

# Ensure the web app can import without the optional dotenv package during tests.
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda: None  # type: ignore[attr-defined]
sys.modules.setdefault("dotenv", dotenv_stub)

from kidbank.webapp import (  # noqa: E402
    GLOBAL_CHORE_KID_ID,
    GLOBAL_CHORE_STATUS_APPROVED,
    GLOBAL_CHORE_STATUS_PENDING,
    KidMarketInstrument,
    MoneyRequest,
    app,
    engine,
    filter_events,
    list_kid_market_symbols,
    run_migrations,
)
from kidbank.webapp import Child, Goal, Event, Chore, ChoreInstance, GlobalChoreClaim, Certificate, Investment, InvestmentTx


run_migrations()


@pytest.fixture(autouse=True)
def clean_database() -> None:
    with Session(engine) as session:
        for model in (
            InvestmentTx,
            Investment,
            KidMarketInstrument,
            GlobalChoreClaim,
            ChoreInstance,
            Chore,
            Certificate,
            MoneyRequest,
            Goal,
            Event,
            Child,
        ):
            session.exec(delete(model))
        session.commit()


def test_goal_deposit_defaults_to_goal_name_without_error() -> None:
    client = TestClient(app)
    with Session(engine) as session:
        child = Child(kid_id="alex", name="Alex", balance_cents=5_000, kid_pin="1234")
        session.add(child)
        session.commit()
        session.refresh(child)
        goal = Goal(kid_id=child.kid_id, name="Bike", target_cents=10_000)
        session.add(goal)
        session.commit()
        session.refresh(goal)
        goal_id = goal.id
    response = client.post(
        "/kid/login", data={"kid_id": "alex", "kid_pin": "1234"}, follow_redirects=False
    )
    assert response.status_code == 302
    deposit = client.post(
        "/kid/goal_deposit", data={"goal_id": goal_id, "amount": "20"}, follow_redirects=False
    )
    assert deposit.status_code == 302
    with Session(engine) as session:
        updated_goal = session.get(Goal, goal_id)
        updated_child = session.exec(select(Child).where(Child.kid_id == "alex")).first()
        events = session.exec(select(Event).where(Event.child_id == "alex").order_by(desc(Event.timestamp))).all()
    assert updated_goal is not None and updated_goal.saved_cents == 2_000
    assert updated_child is not None and updated_child.balance_cents == 3_000
    assert any(evt.reason == "goal_deposit:Bike" for evt in events)


def test_kid_market_symbols_are_scoped_per_kid() -> None:
    client_one = TestClient(app)
    client_two = TestClient(app)
    with Session(engine) as session:
        session.add_all(
            [
                Child(kid_id="kid1", name="One", balance_cents=100_00, kid_pin="1111"),
                Child(kid_id="kid2", name="Two", balance_cents=100_00, kid_pin="2222"),
            ]
        )
        session.commit()
    resp = client_one.post(
        "/kid/login", data={"kid_id": "kid1", "kid_pin": "1111"}, follow_redirects=False
    )
    assert resp.status_code == 302
    track = client_one.post(
        "/kid/invest/track", data={"symbol": "AAPL", "name": "Apple"}, follow_redirects=False
    )
    assert track.status_code == 302
    resp2 = client_two.post(
        "/kid/login", data={"kid_id": "kid2", "kid_pin": "2222"}, follow_redirects=False
    )
    assert resp2.status_code == 302
    symbols_kid1 = list_kid_market_symbols("kid1")
    symbols_kid2 = list_kid_market_symbols("kid2")
    assert any(sym == "AAPL" for sym in symbols_kid1)
    assert all(sym != "AAPL" for sym in symbols_kid2)


def test_filter_events_supports_search_direction_and_kid() -> None:
    now = datetime.utcnow()
    events = [
        Event(child_id="kid1", change_cents=500, reason="chore:dishes", timestamp=now - timedelta(days=1)),
        Event(child_id="kid1", change_cents=-200, reason="prize:toy", timestamp=now - timedelta(days=2)),
        Event(child_id="kid2", change_cents=0, reason="adjustment", timestamp=now - timedelta(days=3)),
    ]
    kid_lookup = {"kid1": Child(kid_id="kid1", name="Alex"), "kid2": Child(kid_id="kid2", name="Blair")}
    credits = filter_events(events, direction="credit")
    assert len(credits) == 1 and credits[0].reason.startswith("chore")
    zero_events = filter_events(events, direction="zero")
    assert len(zero_events) == 1 and zero_events[0].reason == "adjustment"
    search = filter_events(events, search="toy")
    assert len(search) == 1 and search[0].reason == "prize:toy"
    kid_filtered = filter_events(events, kid_filter="kid2", kid_lookup=kid_lookup)
    assert len(kid_filtered) == 1 and kid_filtered[0].child_id == "kid2"


def test_admin_chore_payout_zero_override_uses_default_award() -> None:
    client = TestClient(app)
    with Session(engine) as session:
        child = Child(kid_id="piper", name="Piper", balance_cents=0)
        chore = Chore(kid_id="piper", name="Laundry", type="daily", award_cents=150)
        session.add(child)
        session.add(chore)
        session.commit()
        session.refresh(chore)
        instance = ChoreInstance(
            chore_id=chore.id,
            period_key="2024-01-01",
            status="pending",
            completed_at=datetime.utcnow(),
        )
        session.add(instance)
        session.commit()
        session.refresh(instance)
        instance_id = instance.id
    login = client.post("/admin/login", data={"pin": "1022"}, follow_redirects=False)
    assert login.status_code == 302
    payout = client.post(
        "/admin/chore_payout",
        data={"instance_id": instance_id, "amount": "0", "reason": "Great work!", "redirect": "/admin?section=payouts"},
        follow_redirects=False,
    )
    assert payout.status_code == 302
    assert payout.headers["location"] == "/admin?section=payouts"
    with Session(engine) as session:
        updated_child = session.exec(select(Child).where(Child.kid_id == "piper")).first()
        updated_instance = session.exec(select(ChoreInstance).where(ChoreInstance.id == instance_id)).first()
        payout_event = session.exec(select(Event).where(Event.child_id == "piper")).first()
    assert updated_child is not None and updated_child.balance_cents == 150
    assert updated_instance is not None and updated_instance.status == "paid"
    assert payout_event is not None and payout_event.change_cents == 150


def test_global_chore_claims_respect_redirect_and_pay_award() -> None:
    client = TestClient(app)
    with Session(engine) as session:
        child = Child(kid_id="casey", name="Casey", balance_cents=0)
        session.add(child)
        session.commit()
        session.refresh(child)
        chore = Chore(
            kid_id=GLOBAL_CHORE_KID_ID,
            name="Garage Clean",
            type="global",
            award_cents=600,
            max_claimants=2,
        )
        session.add(chore)
        session.commit()
        session.refresh(chore)
        claim = GlobalChoreClaim(
            chore_id=chore.id,
            kid_id=child.kid_id,
            period_key="2024-W01",
            status=GLOBAL_CHORE_STATUS_PENDING,
            submitted_at=datetime.utcnow(),
        )
        session.add(claim)
        session.commit()
        session.refresh(claim)
        claim_id = claim.id
        chore_id = chore.id
    login = client.post("/admin/login", data={"pin": "1022"}, follow_redirects=False)
    assert login.status_code == 302
    approve = client.post(
        "/admin/global_chore/claims",
        data={
            "decision": "approve",
            "chore_id": chore_id,
            "period_key": "2024-W01",
            "claim_ids": str(claim_id),
            "redirect": "/admin?section=payouts",
        },
        follow_redirects=False,
    )
    assert approve.status_code == 302
    assert approve.headers["location"] == "/admin?section=payouts"
    with Session(engine) as session:
        updated_claim = session.get(GlobalChoreClaim, claim_id)
        updated_child = session.exec(select(Child).where(Child.kid_id == "casey")).first()
        payout_event = session.exec(select(Event).where(Event.child_id == "casey")).first()
    assert updated_claim is not None and updated_claim.status == GLOBAL_CHORE_STATUS_APPROVED
    assert updated_child is not None and updated_child.balance_cents == 600
    assert payout_event is not None and payout_event.change_cents == 600
