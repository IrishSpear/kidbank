import json
import sys
import types
from datetime import date, datetime, timedelta

import pytest
try:  # pragma: no cover - allow running without FastAPI installed
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    from starlette.testclient import TestClient
from sqlmodel import Session, delete, select, desc

# Ensure the web app can import without the optional dotenv package during tests.
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda: None  # type: ignore[attr-defined]
sys.modules.setdefault("dotenv", dotenv_stub)

from kidbank.webapp import (  # noqa: E402
    apply_chore_penalties,
    CHORE_STATUS_PENDING_MARKETPLACE,
    DAD_PIN,
    GLOBAL_CHORE_KID_ID,
    GLOBAL_CHORE_STATUS_APPROVED,
    GLOBAL_CHORE_STATUS_PENDING,
    KidMarketInstrument,
    MarketplaceListing,
    MoneyRequest,
    REMEMBER_NAME_COOKIE,
    app,
    detailed_history_chart_svg,
    engine,
    ensure_default_learning_content,
    filter_events,
    list_kid_market_symbols,
    load_admin_privileges,
    run_migrations,
)
from kidbank.webapp import (
    Certificate,
    Child,
    Chore,
    ChoreInstance,
    Event,
    Goal,
    GlobalChoreClaim,
    Investment,
    InvestmentTx,
    Lesson,
    MetaKV,
    Quiz,
    QuizAttempt,
)


run_migrations()


@pytest.fixture(autouse=True)
def clean_database() -> None:
    with Session(engine) as session:
        for model in (
            MetaKV,
            InvestmentTx,
            Investment,
            KidMarketInstrument,
            GlobalChoreClaim,
            MarketplaceListing,
            ChoreInstance,
            Chore,
            Certificate,
            MoneyRequest,
            QuizAttempt,
            Quiz,
            Lesson,
            Goal,
            Event,
            Child,
        ):
            session.exec(delete(model))
        session.commit()
    ensure_default_learning_content()


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


def test_marketplace_completion_skips_standard_pending_entry() -> None:
    owner_client = TestClient(app)
    worker_client = TestClient(app)
    with Session(engine) as session:
        owner = Child(kid_id="owner", name="Owner", balance_cents=5_000, kid_pin="1111")
        worker = Child(kid_id="worker", name="Worker", balance_cents=0, kid_pin="2222")
        chore = Chore(kid_id=owner.kid_id, name="Special", type="special", award_cents=1_000)
        session.add_all([owner, worker, chore])
        session.commit()
        session.refresh(chore)
        chore_id = chore.id
    login_owner = owner_client.post(
        "/kid/login", data={"kid_id": "owner", "kid_pin": "1111"}, follow_redirects=False
    )
    assert login_owner.status_code == 302
    listed = owner_client.post(
        "/kid/marketplace/list",
        data={"chore_id": chore_id, "offer": "10.50"},
        follow_redirects=False,
    )
    assert listed.status_code == 302
    with Session(engine) as session:
        listing = session.exec(
            select(MarketplaceListing)
            .where(MarketplaceListing.chore_id == chore_id)
            .where(MarketplaceListing.owner_kid_id == "owner")
        ).first()
        assert listing is not None and listing.id is not None
        listing_id = listing.id
    login_worker = worker_client.post(
        "/kid/login", data={"kid_id": "worker", "kid_pin": "2222"}, follow_redirects=False
    )
    assert login_worker.status_code == 302
    claim = worker_client.post(
        "/kid/marketplace/claim", data={"listing_id": listing_id}, follow_redirects=False
    )
    assert claim.status_code == 302
    complete = worker_client.post(
        "/kid/marketplace/complete", data={"listing_id": listing_id}, follow_redirects=False
    )
    assert complete.status_code == 302
    with Session(engine) as session:
        instances = session.exec(
            select(ChoreInstance).where(ChoreInstance.chore_id == chore_id)
        ).all()
        statuses = {inst.status for inst in instances}
        assert CHORE_STATUS_PENDING_MARKETPLACE in statuses
        assert "pending" not in statuses
        standard_pending = session.exec(
            select(ChoreInstance, Chore, Child)
            .where(ChoreInstance.status == "pending")
            .where(ChoreInstance.chore_id == Chore.id)
            .where(Chore.kid_id == Child.kid_id)
        ).all()
    assert not standard_pending


def test_marketplace_payout_includes_award_and_offer() -> None:
    owner_client = TestClient(app)
    worker_client = TestClient(app)
    admin_client = TestClient(app)
    with Session(engine) as session:
        owner = Child(kid_id="owner", name="Owner", balance_cents=5_000, kid_pin="1111")
        worker = Child(kid_id="worker", name="Worker", balance_cents=0, kid_pin="2222")
        chore = Chore(kid_id=owner.kid_id, name="Yardwork", type="special", award_cents=1_500)
        session.add_all([owner, worker, chore])
        session.commit()
        session.refresh(chore)
        chore_id = chore.id
    login_owner = owner_client.post(
        "/kid/login", data={"kid_id": "owner", "kid_pin": "1111"}, follow_redirects=False
    )
    assert login_owner.status_code == 302
    listed = owner_client.post(
        "/kid/marketplace/list",
        data={"chore_id": chore_id, "offer": "5.50"},
        follow_redirects=False,
    )
    assert listed.status_code == 302
    with Session(engine) as session:
        listing = session.exec(
            select(MarketplaceListing)
            .where(MarketplaceListing.chore_id == chore_id)
            .where(MarketplaceListing.owner_kid_id == "owner")
        ).first()
        assert listing is not None and listing.id is not None
        listing_id = listing.id
    login_worker = worker_client.post(
        "/kid/login", data={"kid_id": "worker", "kid_pin": "2222"}, follow_redirects=False
    )
    assert login_worker.status_code == 302
    claim = worker_client.post(
        "/kid/marketplace/claim", data={"listing_id": listing_id}, follow_redirects=False
    )
    assert claim.status_code == 302
    complete = worker_client.post(
        "/kid/marketplace/complete", data={"listing_id": listing_id}, follow_redirects=False
    )
    assert complete.status_code == 302
    admin_login = admin_client.post(
        "/admin/login", data={"pin": DAD_PIN}, follow_redirects=False
    )
    assert admin_login.status_code == 302
    payout = admin_client.post(
        "/admin/marketplace/payout",
        data={"listing_id": listing_id, "redirect": "/admin?section=payouts"},
        follow_redirects=False,
    )
    assert payout.status_code == 302
    with Session(engine) as session:
        owner = session.exec(select(Child).where(Child.kid_id == "owner")).first()
        worker = session.exec(select(Child).where(Child.kid_id == "worker")).first()
        listing = session.get(MarketplaceListing, listing_id)
        events = session.exec(select(Event).where(Event.child_id == "worker")).all()
    assert owner is not None and owner.balance_cents == 5_000 - 550
    assert worker is not None and worker.balance_cents == 1_500 + 550
    assert listing is not None and listing.final_payout_cents == 1_500 + 550
    assert any(evt.change_cents == 1_500 + 550 for evt in events)


def test_marketplace_payout_zero_override_defaults_to_total() -> None:
    owner_client = TestClient(app)
    worker_client = TestClient(app)
    admin_client = TestClient(app)
    with Session(engine) as session:
        owner = Child(kid_id="owner", name="Owner", balance_cents=10_000, kid_pin="1111")
        worker = Child(kid_id="helper", name="Helper", balance_cents=0, kid_pin="3333")
        chore = Chore(kid_id=owner.kid_id, name="Garage", type="special", award_cents=2_000)
        session.add_all([owner, worker, chore])
        session.commit()
        session.refresh(chore)
        chore_id = chore.id
    login_owner = owner_client.post(
        "/kid/login", data={"kid_id": "owner", "kid_pin": "1111"}, follow_redirects=False
    )
    assert login_owner.status_code == 302
    listed = owner_client.post(
        "/kid/marketplace/list",
        data={"chore_id": chore_id, "offer": "3.25"},
        follow_redirects=False,
    )
    assert listed.status_code == 302
    with Session(engine) as session:
        listing = session.exec(
            select(MarketplaceListing)
            .where(MarketplaceListing.chore_id == chore_id)
            .where(MarketplaceListing.owner_kid_id == "owner")
        ).first()
        assert listing is not None and listing.id is not None
        listing_id = listing.id
    login_worker = worker_client.post(
        "/kid/login", data={"kid_id": "helper", "kid_pin": "3333"}, follow_redirects=False
    )
    assert login_worker.status_code == 302
    claim = worker_client.post(
        "/kid/marketplace/claim", data={"listing_id": listing_id}, follow_redirects=False
    )
    assert claim.status_code == 302
    complete = worker_client.post(
        "/kid/marketplace/complete", data={"listing_id": listing_id}, follow_redirects=False
    )
    assert complete.status_code == 302
    admin_login = admin_client.post(
        "/admin/login", data={"pin": DAD_PIN}, follow_redirects=False
    )
    assert admin_login.status_code == 302
    payout = admin_client.post(
        "/admin/marketplace/payout",
        data={
            "listing_id": listing_id,
            "amount": "0",
            "redirect": "/admin?section=payouts",
        },
        follow_redirects=False,
    )
    assert payout.status_code == 302
    with Session(engine) as session:
        owner = session.exec(select(Child).where(Child.kid_id == "owner")).first()
        worker = session.exec(select(Child).where(Child.kid_id == "helper")).first()
        listing = session.get(MarketplaceListing, listing_id)
        events = session.exec(select(Event).where(Event.child_id == "helper")).all()
    assert owner is not None and owner.balance_cents == 10_000 - 325
    assert worker is not None and worker.balance_cents == 2_000 + 325
    assert listing is not None and listing.final_payout_cents == 2_000 + 325
    assert any(evt.change_cents == 2_000 + 325 for evt in events)


def test_remember_me_cookie_prefills_username() -> None:
    client = TestClient(app)
    with Session(engine) as session:
        child = Child(kid_id="alex", name="Alex", balance_cents=0, kid_pin="1234")
        session.add(child)
        session.commit()
    login = client.post(
        "/kid/login",
        data={"kid_id": "alex", "kid_pin": "1234", "remember_me": "1"},
        follow_redirects=False,
    )
    assert login.status_code == 302
    assert client.cookies.get(REMEMBER_NAME_COOKIE) == "alex"
    logout = client.post("/kid/logout", follow_redirects=False)
    assert logout.status_code == 302
    landing = client.get("/")
    assert "value='alex'" in landing.text
    login_without_remember = client.post(
        "/kid/login",
        data={"kid_id": "alex", "kid_pin": "1234"},
        follow_redirects=False,
    )
    assert login_without_remember.status_code == 302
    assert not client.cookies.get(REMEMBER_NAME_COOKIE)
    client.post("/kid/logout", follow_redirects=False)
    landing_after_clear = client.get("/")
    assert "value='alex'" not in landing_after_clear.text


def test_kid_investing_section_alias() -> None:
    client = TestClient(app)
    with Session(engine) as session:
        child = Child(kid_id="ivy", name="Ivy", balance_cents=0, kid_pin="2468")
        session.add(child)
        session.commit()
    login = client.post(
        "/kid/login", data={"kid_id": "ivy", "kid_pin": "2468"}, follow_redirects=False
    )
    assert login.status_code == 302
    alias_page = client.get("/kid?section=invest")
    canonical_page = client.get("/kid?section=investing")
    assert alias_page.status_code == 200
    assert canonical_page.status_code == 200
    assert "Investing" in alias_page.text
    assert "section=investing" in alias_page.text
    assert "Investing" in canonical_page.text


def test_detailed_history_chart_has_no_marker_circles() -> None:
    now = datetime.utcnow()
    history = [
        {"p": 1000, "t": (now - timedelta(days=1)).isoformat()},
        {"p": 1500, "t": now.isoformat()},
        {"p": 1400, "t": (now + timedelta(days=1)).isoformat()},
    ]
    svg = detailed_history_chart_svg(history, width=300, height=180)
    assert "<circle" not in svg


def test_dad_updates_privileges_and_restricted_admin_blocked() -> None:
    client = TestClient(app)
    login = client.post(
        "/admin/login", data={"pin": DAD_PIN}, follow_redirects=False
    )
    assert login.status_code == 302
    add_admin = client.post(
        "/admin/add_parent_admin",
        data={"label": "Grandma", "pin": "4444", "confirm_pin": "4444"},
        follow_redirects=False,
    )
    assert add_admin.status_code == 302
    update = client.post(
        "/admin/update_privileges",
        data={"role": "grandma", "kid_scope": "all", "perm_payouts": "on"},
        follow_redirects=False,
    )
    assert update.status_code == 302
    assert update.headers["location"] == "/admin?section=admins"
    with Session(engine) as session:
        saved_privs = load_admin_privileges(session, "grandma")
    assert not saved_privs.can_create_accounts
    logout = client.post("/admin/logout", follow_redirects=False)
    assert logout.status_code == 302
    limited_login = client.post(
        "/admin/login", data={"pin": "4444"}, follow_redirects=False
    )
    assert limited_login.status_code == 302
    attempt_create = client.post(
        "/create_kid",
        data={
            "kid_id": "newkid",
            "name": "New Kid",
            "starting": "5.00",
            "allowance": "1.00",
            "kid_pin": "0000",
        },
        follow_redirects=False,
    )
    assert attempt_create.status_code == 302
    assert attempt_create.headers["location"].startswith("/admin?section=accounts")
    with Session(engine) as session:
        assert session.exec(select(Child)).first() is None


def test_only_dad_can_update_privileges() -> None:
    client = TestClient(app)
    login = client.post(
        "/admin/login", data={"pin": DAD_PIN}, follow_redirects=False
    )
    assert login.status_code == 302
    add_admin = client.post(
        "/admin/add_parent_admin",
        data={"label": "Helper", "pin": "5555", "confirm_pin": "5555"},
        follow_redirects=False,
    )
    assert add_admin.status_code == 302
    client.post("/admin/logout", follow_redirects=False)
    helper_login = client.post(
        "/admin/login", data={"pin": "5555"}, follow_redirects=False
    )
    assert helper_login.status_code == 302
    attempt = client.post(
        "/admin/update_privileges",
        data={"role": "helper", "kid_scope": "all"},
        follow_redirects=False,
    )
    assert attempt.status_code == 302
    assert attempt.headers["location"] == "/admin?section=admins"
    with Session(engine) as session:
        privs = load_admin_privileges(session, "helper")
    assert privs.can_create_accounts
    assert privs.can_manage_payouts


def test_global_chore_multiple_claims_split_evenly() -> None:
    client = TestClient(app)
    period_key = "2024-W02"
    with Session(engine) as session:
        kids = [
            Child(kid_id="avery", name="Avery", balance_cents=0),
            Child(kid_id="blake", name="Blake", balance_cents=0),
            Child(kid_id="cameron", name="Cameron", balance_cents=0),
        ]
        session.add_all(kids)
        session.commit()
        for kid in kids:
            session.refresh(kid)
        chore = Chore(
            kid_id=GLOBAL_CHORE_KID_ID,
            name="Community Cleanup",
            type="global",
            award_cents=500,
            max_claimants=3,
        )
        session.add(chore)
        session.commit()
        session.refresh(chore)
        claims: list[GlobalChoreClaim] = []
        for kid in kids:
            claim = GlobalChoreClaim(
                chore_id=chore.id,
                kid_id=kid.kid_id,
                period_key=period_key,
                status=GLOBAL_CHORE_STATUS_PENDING,
                submitted_at=datetime.utcnow(),
            )
            session.add(claim)
            claims.append(claim)
        session.commit()
        for claim in claims:
            session.refresh(claim)
        claim_ids = [claim.id for claim in claims]
        chore_id = chore.id
    login = client.post(
        "/admin/login", data={"pin": DAD_PIN}, follow_redirects=False
    )
    assert login.status_code == 302
    form_data = {
        "decision": "approve",
        "chore_id": str(chore_id),
        "period_key": period_key,
        "redirect": "/admin?section=payouts",
        "claim_ids": [str(claim_id) for claim_id in claim_ids],
    }
    approve = client.post(
        "/admin/global_chore/claims", data=form_data, follow_redirects=False
    )
    assert approve.status_code == 302
    assert approve.headers["location"] == "/admin?section=payouts"
    with Session(engine) as session:
        awards = []
        balances = []
        for kid_id in ["avery", "blake", "cameron"]:
            claim = session.exec(
                select(GlobalChoreClaim).where(
                    GlobalChoreClaim.kid_id == kid_id,
                    GlobalChoreClaim.period_key == period_key,
                )
            ).first()
            child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
            assert claim is not None and claim.status == GLOBAL_CHORE_STATUS_APPROVED
            assert child is not None
            awards.append(claim.award_cents)
            balances.append(child.balance_cents)
    assert sorted(awards) == [166, 167, 167]
    assert sorted(balances) == [166, 167, 167]


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


def test_ui_preferences_toggle_applies_to_body() -> None:
    client = TestClient(app)
    response = client.post(
        "/ui/preferences",
        data={"font": "dyslexic", "contrast": "high", "redirect_to": "/"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    home = client.get("/")
    assert "data-font='dyslexic'" in home.text
    assert "data-contrast='high'" in home.text


def test_kid_lesson_quiz_awards_bonus_once() -> None:
    client = TestClient(app)
    with Session(engine) as session:
        child = Child(kid_id="learner", name="Learner", balance_cents=0, kid_pin="1234")
        session.add(child)
        session.commit()
        session.refresh(child)
        lesson = Lesson(title="Saving basics", content_md="Be kind to your future self.")
        session.add(lesson)
        session.commit()
        session.refresh(lesson)
        quiz_payload = {
            "questions": [
                {"prompt": "What grows when you save?", "options": ["Choices", "Dust"], "answer": 0},
            ],
            "passing_score": 1,
            "reward_cents": 150,
        }
        quiz = Quiz(lesson_id=lesson.id, payload=json.dumps(quiz_payload), reward_cents=150)
        session.add(quiz)
        session.commit()
    login = client.post(
        "/kid/login", data={"kid_id": "learner", "kid_pin": "1234"}, follow_redirects=False
    )
    assert login.status_code == 302
    lesson_page = client.get(f"/kid/lesson/{lesson.id}")
    assert lesson_page.status_code == 200
    assert "Submit answers" in lesson_page.text
    submit = client.post(
        f"/kid/lesson/{lesson.id}", data={"q0": "0"}, follow_redirects=False
    )
    assert submit.status_code == 302
    with Session(engine) as session:
        updated_child = session.exec(select(Child).where(Child.kid_id == "learner")).first()
        attempts = session.exec(select(QuizAttempt).where(QuizAttempt.child_id == "learner")).all()
        events = session.exec(select(Event).where(Event.child_id == "learner").order_by(desc(Event.timestamp))).all()
    assert updated_child is not None and updated_child.balance_cents == 150
    assert len(attempts) == 1 and attempts[0].score == 1
    assert any(evt.reason.startswith("lesson_reward:") for evt in events)
    submit_again = client.post(
        f"/kid/lesson/{lesson.id}", data={"q0": "0"}, follow_redirects=False
    )
    assert submit_again.status_code == 302
    with Session(engine) as session:
        child_after = session.exec(select(Child).where(Child.kid_id == "learner")).first()
        attempts_after = session.exec(
            select(QuizAttempt).where(QuizAttempt.child_id == "learner").order_by(desc(QuizAttempt.created_at))
        ).all()
    assert child_after is not None and child_after.balance_cents == 150
    assert len(attempts_after) == 2


def test_apply_chore_penalties_charges_for_missed_daily_chore() -> None:
    with Session(engine) as session:
        child = Child(kid_id="casey", name="Casey", balance_cents=500, kid_pin="1111")
        session.add(child)
        session.commit()
        session.refresh(child)
        chore = Chore(
            kid_id=child.kid_id,
            name="Dishes",
            type="daily",
            award_cents=100,
            penalty_cents=200,
            created_at=datetime(2024, 1, 1, 8, 0),
        )
        session.add(chore)
        session.commit()
        session.refresh(chore)
        kid_id = child.kid_id
        chore_id = chore.id

    apply_chore_penalties(moment=datetime(2024, 1, 2, 1, 0))

    with Session(engine) as session:
        updated_child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        assert updated_child is not None
        assert updated_child.balance_cents == 300
        reason = f"chore_penalty_missed:{chore_id}:2024-01-01"
        events = session.exec(select(Event).where(Event.child_id == kid_id)).all()
        assert any(evt.reason == reason for evt in events)
        refreshed = session.get(Chore, chore_id)
        assert refreshed is not None
        assert refreshed.penalty_last_date == date(2024, 1, 1)


def test_apply_chore_penalties_ignores_submitted_chore() -> None:
    with Session(engine) as session:
        child = Child(kid_id="blake", name="Blake", balance_cents=400, kid_pin="2222")
        session.add(child)
        session.commit()
        session.refresh(child)
        chore = Chore(
            kid_id=child.kid_id,
            name="Laundry",
            type="daily",
            award_cents=100,
            penalty_cents=200,
            created_at=datetime(2024, 1, 1, 8, 0),
        )
        session.add(chore)
        session.commit()
        session.refresh(chore)
        instance = ChoreInstance(
            chore_id=chore.id,
            period_key="2024-01-01",
            status="pending",
            completed_at=datetime(2024, 1, 1, 21, 0),
        )
        session.add(instance)
        session.commit()
        kid_id = child.kid_id
        chore_id = chore.id

    apply_chore_penalties(moment=datetime(2024, 1, 2, 9, 0))

    with Session(engine) as session:
        updated_child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        assert updated_child is not None
        assert updated_child.balance_cents == 400
        reason = f"chore_penalty_missed:{chore_id}:2024-01-01"
        penalty_events = session.exec(
            select(Event).where(Event.child_id == kid_id, Event.reason == reason)
        ).all()
        assert penalty_events == []
        refreshed = session.get(Chore, chore_id)
        assert refreshed is not None
        assert refreshed.penalty_last_date == date(2024, 1, 1)


def test_apply_chore_penalties_catches_multiple_days_and_is_idempotent() -> None:
    with Session(engine) as session:
        child = Child(kid_id="jules", name="Jules", balance_cents=1_000, kid_pin="3333")
        session.add(child)
        session.commit()
        session.refresh(child)
        chore = Chore(
            kid_id=child.kid_id,
            name="Room",
            type="daily",
            award_cents=150,
            penalty_cents=150,
            created_at=datetime(2024, 1, 1, 8, 0),
        )
        session.add(chore)
        session.commit()
        session.refresh(chore)
        kid_id = child.kid_id
        chore_id = chore.id

    apply_chore_penalties(moment=datetime(2024, 1, 4, 8, 0))
    apply_chore_penalties(moment=datetime(2024, 1, 4, 23, 0))

    with Session(engine) as session:
        updated_child = session.exec(select(Child).where(Child.kid_id == kid_id)).first()
        assert updated_child is not None
        # Missed January 1st, 2nd, and 3rd
        assert updated_child.balance_cents == 550
        reasons = {
            f"chore_penalty_missed:{chore_id}:2024-01-01",
            f"chore_penalty_missed:{chore_id}:2024-01-02",
            f"chore_penalty_missed:{chore_id}:2024-01-03",
        }
        logged = session.exec(select(Event).where(Event.child_id == kid_id)).all()
        assert reasons.issubset({evt.reason for evt in logged})
        assert len([evt for evt in logged if evt.reason in reasons]) == 3
        refreshed = session.get(Chore, chore_id)
        assert refreshed is not None
        assert refreshed.penalty_last_date == date(2024, 1, 3)
