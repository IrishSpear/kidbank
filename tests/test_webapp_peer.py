import importlib
import sys
from typing import Iterator

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from sqlmodel import Session, select


@pytest.fixture
def webapp_module(tmp_path, monkeypatch):
    db_path = tmp_path / "kidbank.db"
    monkeypatch.setenv("KIDBANK_SQLITE", str(db_path))
    module = importlib.import_module("kidbank.webapp") if "kidbank.webapp" not in sys.modules else sys.modules["kidbank.webapp"]
    module = importlib.reload(module)
    return module


@pytest.fixture
def client(webapp_module) -> Iterator[TestClient]:
    with TestClient(webapp_module.app) as test_client:
        yield test_client


def add_child(module, kid_id: str, name: str, balance_cents: int, pin: str) -> None:
    with Session(module.engine) as session:
        child = module.Child(kid_id=kid_id, name=name, balance_cents=balance_cents, kid_pin=pin)
        session.add(child)
        session.commit()


def test_kid_can_send_money_to_sibling(webapp_module, client: TestClient) -> None:
    add_child(webapp_module, "ava01", "Ava", 1_000, "1111")
    add_child(webapp_module, "ben01", "Ben", 200, "2222")

    response = client.post("/kid/login", data={"kid_id": "ava01", "kid_pin": "1111"})
    assert response.status_code in {200, 302}

    response = client.post(
        "/kid/send",
        data={"to_kid": "ben01", "amount": "3.50", "reason": "Sharing snacks"},
    )
    assert response.status_code in {200, 302}

    with Session(webapp_module.engine) as session:
        ava = session.exec(select(webapp_module.Child).where(webapp_module.Child.kid_id == "ava01")).first()
        ben = session.exec(select(webapp_module.Child).where(webapp_module.Child.kid_id == "ben01")).first()
        ava_event = (
            session.exec(
                select(webapp_module.Event)
                .where(webapp_module.Event.child_id == "ava01")
                .order_by(webapp_module.Event.timestamp)
            ).first()
        )
        ben_event = (
            session.exec(
                select(webapp_module.Event)
                .where(webapp_module.Event.child_id == "ben01")
                .order_by(webapp_module.Event.timestamp)
            ).first()
        )

    assert ava is not None and ben is not None
    assert ava.balance_cents == 1_000 - 350
    assert ben.balance_cents == 200 + 350
    assert ava_event is not None and "peer_transfer_to:ben01" in ava_event.reason
    assert "Sharing snacks" in ava_event.reason
    assert ben_event is not None and "peer_transfer_from:ava01" in ben_event.reason
    assert "Sharing snacks" in ben_event.reason


def test_request_flow_allows_sibling_to_fulfill(webapp_module, client: TestClient) -> None:
    add_child(webapp_module, "ava01", "Ava", 100, "1111")
    add_child(webapp_module, "ben01", "Ben", 1_500, "2222")

    response = client.post("/kid/login", data={"kid_id": "ava01", "kid_pin": "1111"})
    assert response.status_code in {200, 302}

    response = client.post(
        "/kid/request_money",
        data={"target_kid": "ben01", "amount": "4.25", "reason": "Field trip"},
    )
    assert response.status_code in {200, 302}

    with Session(webapp_module.engine) as session:
        peer_request = session.exec(select(webapp_module.PeerRequest)).first()
        assert peer_request is not None
        request_id = peer_request.id
        assert peer_request.status == "pending"

    client.post("/kid/logout")

    response = client.post("/kid/login", data={"kid_id": "ben01", "kid_pin": "2222"})
    assert response.status_code in {200, 302}

    response = client.post(
        "/kid/request/respond",
        data={"request_id": request_id, "action": "fulfill"},
    )
    assert response.status_code in {200, 302}

    with Session(webapp_module.engine) as session:
        updated_request = session.get(webapp_module.PeerRequest, request_id)
        ava = session.exec(select(webapp_module.Child).where(webapp_module.Child.kid_id == "ava01")).first()
        ben = session.exec(select(webapp_module.Child).where(webapp_module.Child.kid_id == "ben01")).first()
        ava_event = (
            session.exec(
                select(webapp_module.Event)
                .where(webapp_module.Event.child_id == "ava01")
                .order_by(webapp_module.Event.timestamp.desc())
            ).first()
        )
        ben_event = (
            session.exec(
                select(webapp_module.Event)
                .where(webapp_module.Event.child_id == "ben01")
                .order_by(webapp_module.Event.timestamp.desc())
            ).first()
        )

    assert updated_request is not None
    assert updated_request.status == "fulfilled"
    assert updated_request.resolved_by == "ben01"
    assert ava is not None and ben is not None
    assert ava.balance_cents == 100 + 425
    assert ben.balance_cents == 1_500 - 425
    assert ava_event is not None and "Field trip" in ava_event.reason
    assert "peer_request_receive:ben01" in ava_event.reason
    assert ben_event is not None and "peer_request_pay:ava01" in ben_event.reason
