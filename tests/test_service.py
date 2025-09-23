from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from kidbank.chores import Weekday
from kidbank.exceptions import AccountNotFoundError, DuplicateAccountError, InsufficientFundsError
from kidbank.models import EventCategory, TransactionType, TransferRequestStatus
from kidbank.service import KidBank


def test_create_and_lookup_accounts() -> None:
    bank = KidBank()

    ava = bank.create_account("Ava", starting_balance=5)
    assert bank.has_account("Ava")
    assert ava.balance == Decimal("5.00")

    with pytest.raises(DuplicateAccountError):
        bank.create_account("Ava")

    with pytest.raises(AccountNotFoundError):
        bank.get_account("Nonexistent")


def test_deposit_and_withdraw_workflow() -> None:
    bank = KidBank()
    bank.create_account("Ava")

    deposit_tx = bank.deposit("Ava", 12.5, description="Chores")
    assert deposit_tx.amount == Decimal("12.50")

    withdrawal_tx = bank.withdraw("Ava", 2, description="Snack")
    assert withdrawal_tx.amount == Decimal("2.00")
    assert withdrawal_tx.type is TransactionType.WITHDRAWAL
    assert bank.get_account("Ava").balance == Decimal("10.50")


def test_transfer_between_accounts() -> None:
    bank = KidBank()
    bank.create_account("Ava")
    bank.create_account("Ben")
    bank.deposit("Ava", 15)

    outgoing, incoming = bank.transfer("Ava", "Ben", 4.5, description="Gift", comment="Birthday share")

    assert outgoing.type is TransactionType.TRANSFER_OUT
    assert incoming.type is TransactionType.TRANSFER_IN
    assert bank.get_account("Ava").balance == Decimal("10.50")
    assert bank.get_account("Ben").balance == Decimal("4.50")
    assert outgoing.metadata.get("comment") == "Birthday share"

    with pytest.raises(InsufficientFundsError):
        bank.transfer("Ava", "Ben", 100)


def test_goals_and_rewards_via_service() -> None:
    bank = KidBank()
    bank.create_account("Ava")
    bank.deposit("Ava", 30)

    goal = bank.add_goal("Ava", name="Bicycle", target_amount=60)
    assert goal.remaining == Decimal("60.00")

    bank.contribute_to_goal("Ava", name="Bicycle", amount=20)
    assert goal.saved_amount == Decimal("20.00")
    assert goal.remaining == Decimal("40.00")

    reward = bank.redeem_reward("Ava", name="Movie night", cost=5)
    assert reward.name == "Movie night"
    assert bank.get_account("Ava").balance == Decimal("5.00")


def test_summary_lists_accounts() -> None:
    bank = KidBank()
    bank.create_account("Ava", starting_balance=2)
    bank.create_account("Ben")
    bank.deposit("Ben", 3)

    summary = bank.summary()

    assert "KidBank summary:" in summary
    assert "Ava" in summary and "Ben" in summary
    assert "Total balance" in summary


def test_certificate_of_deposit_flow() -> None:
    bank = KidBank()
    bank.create_account("Ava")
    bank.deposit("Ava", 100, description="Gift")

    new_rate = bank.set_certificate_rate("Ava", 0.05)
    assert new_rate == Decimal("0.0500")

    certificate = bank.open_certificate("Ava", 40, term_months=6)
    assert certificate.principal == Decimal("40.00")
    assert certificate.rate == Decimal("0.0500")
    assert bank.get_account("Ava").balance == Decimal("60.00")
    assert bank.portfolio("Ava").total_certificate_value() >= Decimal("40.00")

    payout = bank.mature_certificates("Ava", at=certificate.matures_on + timedelta(days=1))
    assert payout == certificate.payout_amount()
    assert bank.get_account("Ava").balance == Decimal("60.00") + payout
    assert bank.portfolio("Ava").total_certificate_value() == Decimal("0.00")
    assert bank.certificates("Ava") == ()


def test_certificate_term_rates_and_penalties() -> None:
    bank = KidBank()
    bank.create_account("Ava")
    bank.deposit("Ava", 200, description="Birthday money")

    bank.set_certificate_rate("Ava", 0.03, term_months=6)
    bank.set_certificate_penalty("Ava", term_months=6, days=30)

    opened_on = datetime.utcnow() - timedelta(days=60)
    certificate = bank.open_certificate("Ava", 100, term_months=6, opened_on=opened_on)
    assert certificate.rate == Decimal("0.0300")

    # Update rates for future certificates but existing one should remain unchanged
    bank.set_certificate_rate("Ava", 0.07, term_months=6)
    assert certificate.rate == Decimal("0.0300")

    gross, penalty, net = bank.withdraw_certificate(
        "Ava", certificate, at=opened_on + timedelta(days=90)
    )

    assert gross == Decimal("101.49")
    assert penalty == Decimal("0.50")
    assert net == Decimal("100.99")
    account = bank.get_account("Ava")
    deposit_tx, penalty_tx = account.transactions[-2:]
    assert deposit_tx.category is EventCategory.INVEST
    assert penalty_tx.category is EventCategory.PENALTY
    assert penalty_tx.amount == penalty
    assert account.balance == Decimal("200.99")
    assert certificate not in bank.certificates("Ava")


def test_schedule_chore_on_specific_dates() -> None:
    bank = KidBank()
    bank.create_account("Ava")
    target = date.today()

    chore = bank.schedule_chore("Ava", name="Library drop", value=5, dates=[target])
    board = bank._chores["Ava"]
    scheduled = next(entry for entry in board.chores() if entry.name == "Library drop")

    assert chore.schedule.specific_dates == frozenset({target})
    moment = datetime.combine(target, datetime.min.time())
    assert scheduled.schedule.is_due(moment)

    weekday_chore = bank.schedule_chore("Ava", name="Trash", value=3, weekdays=Weekday.MONDAY)
    assert weekday_chore.schedule.weekdays == frozenset({Weekday.MONDAY})


def test_global_chore_flow() -> None:
    bank = KidBank()
    bank.create_account("Ava")
    bank.create_account("Ben")

    chore = bank.create_global_chore(name="Car Wash", reward=Decimal("10.00"), max_claims=2)
    assert chore.reward == Decimal("10.00")

    bank.submit_global_chore("Ava", "Car Wash", comment="I can scrub")
    bank.submit_global_chore("Ben", "Car Wash")

    payouts = bank.approve_global_chore("Car Wash", ["Ava", "Ben"])
    assert payouts == {"Ava": Decimal("5.00"), "Ben": Decimal("5.00")}

    assert bank.get_account("Ava").balance == Decimal("5.00")
    assert bank.get_account("Ben").balance == Decimal("5.00")
    assert bank.global_chores() == ()

    bank.create_global_chore(name="Garage", reward=Decimal("9.00"), max_claims=2)
    bank.submit_global_chore("Ava", "Garage")
    bank.submit_global_chore("Ben", "Garage")

    override = {"Ava": Decimal("6.00"), "Ben": Decimal("3.00")}
    payouts_override = bank.approve_global_chore("Garage", ["Ava", "Ben"], amount_override=override)
    assert payouts_override == override
    assert bank.get_account("Ava").balance == Decimal("11.00")
    assert bank.get_account("Ben").balance == Decimal("8.00")


def test_money_request_flow() -> None:
    bank = KidBank()
    bank.create_account("Ava")
    bank.create_account("Ben")
    bank.deposit("Ben", 20)

    request = bank.request_money("Ava", "Ben", 5, comment="Lunch")
    assert request.status is TransferRequestStatus.PENDING
    pending = bank.transfer_requests(for_child="Ben", status=TransferRequestStatus.PENDING)
    assert request in pending

    bank.respond_money_request(request.request_id, responder="Ben", approve=True)
    assert request.status is TransferRequestStatus.APPROVED
    assert bank.get_account("Ava").balance == Decimal("5.00")
    assert bank.get_account("Ben").balance == Decimal("15.00")
    assert bank.get_account("Ben").transactions[-1].metadata.get("comment") == "Lunch"

    second = bank.request_money("Ava", "Ben", 2, comment="Snacks")
    bank.respond_money_request(second.request_id, responder="Ben", approve=False)
    assert second.status is TransferRequestStatus.DECLINED
    assert bank.get_account("Ava").balance == Decimal("5.00")
