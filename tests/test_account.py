from decimal import Decimal

import pytest

from kidbank.account import Account
from kidbank.exceptions import GoalNotFoundError, InsufficientFundsError
from kidbank.models import TransactionType


def test_deposit_records_transaction() -> None:
    account = Account("Ava")

    transaction = account.deposit(10, "Weekly allowance")

    assert account.balance == Decimal("10.00")
    assert transaction.description == "Weekly allowance"
    assert transaction.type is TransactionType.DEPOSIT
    assert account.transactions[-1] is transaction


def test_withdraw_reduces_balance() -> None:
    account = Account("Ava")
    account.deposit(Decimal("20"))

    transaction = account.withdraw("5.50", "Book purchase")

    assert account.balance == Decimal("14.50")
    assert transaction.amount == Decimal("5.50")
    assert transaction.type is TransactionType.WITHDRAWAL


def test_withdraw_raises_when_insufficient_funds() -> None:
    account = Account("Ava")

    with pytest.raises(InsufficientFundsError):
        account.withdraw(1)


def test_reward_redemption_tracks_balance_and_history() -> None:
    account = Account("Ava")
    account.deposit(20)

    reward = account.redeem_reward("Ice cream", 3, description="Dessert treat")

    assert reward.cost == Decimal("3.00")
    assert account.balance == Decimal("17.00")
    last_transaction = account.transactions[-1]
    assert last_transaction.type is TransactionType.REWARD
    assert "Dessert" in last_transaction.description


def test_goal_contribution_updates_goal_and_balance() -> None:
    account = Account("Ava")
    account.deposit(30)
    goal = account.add_goal("Lego set", 50, description="Birthday gift")

    updated_goal = account.contribute_to_goal("Lego set", 12.75)

    assert updated_goal is goal
    assert goal.saved_amount == Decimal("12.75")
    assert account.balance == Decimal("17.25")

    with pytest.raises(GoalNotFoundError):
        account.contribute_to_goal("Non-existent", 5)

    with pytest.raises(ValueError):
        account.add_goal("Lego set", 10)


def test_transfer_moves_funds_between_accounts() -> None:
    giver = Account("Ava", starting_balance=25)
    receiver = Account("Ben")

    giver.deposit(5)  # starting balance logged as deposit; add an explicit deposit for clarity

    outgoing, incoming = giver.transfer_to(receiver, 10, description="Sibling share")

    assert giver.balance == Decimal("20.00")
    assert receiver.balance == Decimal("10.00")
    assert outgoing.type is TransactionType.TRANSFER_OUT
    assert incoming.type is TransactionType.TRANSFER_IN
    assert receiver.transactions[-1] is incoming

    with pytest.raises(ValueError):
        giver.transfer_to(giver, 1)

    with pytest.raises(InsufficientFundsError):
        giver.transfer_to(receiver, 100)
