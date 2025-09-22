from datetime import timedelta
from decimal import Decimal

import pytest

from kidbank.exceptions import AccountNotFoundError, DuplicateAccountError, InsufficientFundsError
from kidbank.models import TransactionType
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

    outgoing, incoming = bank.transfer("Ava", "Ben", 4.5, description="Gift")

    assert outgoing.type is TransactionType.TRANSFER_OUT
    assert incoming.type is TransactionType.TRANSFER_IN
    assert bank.get_account("Ava").balance == Decimal("10.50")
    assert bank.get_account("Ben").balance == Decimal("4.50")

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
