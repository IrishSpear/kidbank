"""Account object encapsulating business logic for the KidBank application."""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal
from io import StringIO
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .exceptions import GoalNotFoundError, InsufficientFundsError
from .models import EventCategory, Goal, Reward, Transaction, TransactionType
from .money import AmountLike, format_currency, require_positive, to_decimal


class Account:
    """Represents a child's account in the KidBank system."""

    __slots__ = (
        "child_name",
        "_balance",
        "_transactions",
        "_goals",
        "_rewards",
    )

    def __init__(self, child_name: str, *, starting_balance: AmountLike = 0) -> None:
        self.child_name = child_name
        starting_value = to_decimal(starting_balance)
        require_positive(starting_value, allow_zero=True)
        self._balance: Decimal = starting_value
        self._transactions: list[Transaction] = []
        self._goals: dict[str, Goal] = {}
        self._rewards: list[Reward] = []

        if self._balance > Decimal("0"):
            self._log_transaction(
                self._balance,
                TransactionType.DEPOSIT,
                "Starting balance",
                EventCategory.MANUAL,
                {"source": "initial"},
            )

    @property
    def balance(self) -> Decimal:
        """Return the current account balance."""

        return self._balance

    @property
    def transactions(self) -> Tuple[Transaction, ...]:
        """Return an immutable view of the transaction history."""

        return tuple(self._transactions)

    @property
    def goals(self) -> Tuple[Goal, ...]:
        """Return an immutable view of the savings goals."""

        return tuple(self._goals.values())

    @property
    def rewards(self) -> Tuple[Reward, ...]:
        """Return an immutable view of redeemed rewards."""

        return tuple(self._rewards)

    @property
    def escrow_balance(self) -> Decimal:
        """Return the total funds reserved across all goals."""

        return sum((goal.saved_amount for goal in self._goals.values()), Decimal("0.00"))

    def deposit(
        self,
        amount: AmountLike,
        description: str = "Deposit",
        *,
        category: EventCategory | None = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Transaction:
        """Add money to the account."""

        value = to_decimal(amount)
        require_positive(value)
        self._balance += value
        return self._log_transaction(
            value,
            TransactionType.DEPOSIT,
            description,
            category or EventCategory.MANUAL,
            metadata,
        )

    def withdraw(
        self,
        amount: AmountLike,
        description: str = "Withdrawal",
        *,
        category: EventCategory | None = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Transaction:
        """Remove money from the account if sufficient funds are available."""

        value = to_decimal(amount)
        require_positive(value)
        self._ensure_sufficient_funds(value)
        self._balance -= value
        return self._log_transaction(
            value,
            TransactionType.WITHDRAWAL,
            description,
            category or EventCategory.MANUAL,
            metadata,
        )

    def redeem_reward(
        self,
        name: str,
        cost: AmountLike,
        description: str = "",
        *,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Reward:
        """Redeem a reward by deducting its cost from the account balance."""

        value = to_decimal(cost)
        require_positive(value)
        self._ensure_sufficient_funds(value)
        self._balance -= value
        reward = Reward(name=name, cost=value, description=description)
        self._rewards.append(reward)
        summary = description or f"Reward redeemed: {name}"
        self._log_transaction(
            value,
            TransactionType.REWARD,
            summary,
            EventCategory.REWARD,
            metadata,
        )
        return reward

    def add_goal(
        self,
        name: str,
        target_amount: AmountLike,
        description: str = "",
        *,
        image_url: str = "",
    ) -> Goal:
        """Create a new savings goal for the account."""

        if name in self._goals:
            raise ValueError(f"A goal named '{name}' already exists.")
        goal = Goal(
            name=name,
            target_amount=target_amount,
            description=description,
            image_url=image_url,
        )
        self._goals[name] = goal
        return goal

    def contribute_to_goal(
        self,
        name: str,
        amount: AmountLike,
        *,
        description: str | None = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Goal:
        """Contribute funds from the balance towards a savings goal."""

        goal = self._goals.get(name)
        if goal is None:
            raise GoalNotFoundError(f"Goal '{name}' does not exist.")
        value = to_decimal(amount)
        require_positive(value)
        self._ensure_sufficient_funds(value)
        self._balance -= value
        goal.contribute(value)
        summary = description or f"Contribution to goal: {name}"
        self._log_transaction(
            value,
            TransactionType.GOAL_CONTRIBUTION,
            summary,
            EventCategory.GOAL,
            metadata,
        )
        return goal

    def transfer_to(
        self,
        other: "Account",
        amount: AmountLike,
        description: str | None = None,
        *,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> tuple[Transaction, Transaction]:
        """Transfer money from this account to another."""

        if self is other:
            raise ValueError("Cannot transfer to the same account.")

        value = to_decimal(amount)
        require_positive(value)
        self._ensure_sufficient_funds(value)

        self._balance -= value
        outgoing_summary = description or f"Transfer to {other.child_name}"
        outgoing = self._log_transaction(
            value,
            TransactionType.TRANSFER_OUT,
            outgoing_summary,
            EventCategory.TRANSFER,
            metadata,
        )

        other._balance += value
        incoming_summary = description or f"Transfer from {self.child_name}"
        incoming = other._log_transaction(
            value,
            TransactionType.TRANSFER_IN,
            incoming_summary,
            EventCategory.TRANSFER,
            metadata,
        )

        return outgoing, incoming

    def recent_transactions(self, count: int = 5) -> Tuple[Transaction, ...]:
        """Return the most recent ``count`` transactions."""

        if count < 0:
            raise ValueError("count must not be negative")
        if count == 0:
            return tuple()
        return tuple(self._transactions[-count:])

    def generate_statement(self, *, max_transactions: int = 10) -> str:
        """Create a human-readable summary of the account state."""

        lines = [
            f"Account holder: {self.child_name}",
            f"Current balance: {format_currency(self._balance)}",
            "",
            "Recent transactions:",
        ]
        transactions = self.recent_transactions(min(max_transactions, len(self._transactions)))
        if not transactions:
            lines.append("  (no transactions yet)")
        else:
            for transaction in transactions:
                lines.append(
                    "  "
                    f"[{transaction.timestamp:%Y-%m-%d}] "
                    f"{transaction.type.value.replace('_', ' ').title()}: "
                    f"{format_currency(transaction.amount)} "
                    f"(balance {format_currency(transaction.balance_after)})"
                )
        if self._goals:
            lines.append("")
            lines.append("Savings goals:")
            for goal in self._goals.values():
                status = "complete" if goal.is_complete else f"{goal.progress()*Decimal(100):.1f}%"
                lines.append(
                    "  "
                    f"{goal.name}: saved {format_currency(goal.saved_amount)} "
                    f"of {format_currency(goal.target_amount)} ({status})"
                )
        if self._rewards:
            lines.append("")
            lines.append("Redeemed rewards:")
            for reward in self._rewards[-max_transactions:]:
                lines.append(
                    "  "
                    f"[{reward.redeemed_at:%Y-%m-%d}] {reward.name} "
                    f"for {format_currency(reward.cost)}"
                )
        return "\n".join(lines)

    def filter_transactions(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        categories: Optional[Sequence[EventCategory]] = None,
        types: Optional[Sequence[TransactionType]] = None,
    ) -> Tuple[Transaction, ...]:
        """Return transactions filtered by the provided criteria."""

        result: list[Transaction] = []
        for transaction in self._transactions:
            if start and transaction.timestamp < start:
                continue
            if end and transaction.timestamp > end:
                continue
            if categories and transaction.category not in categories:
                continue
            if types and transaction.type not in types:
                continue
            result.append(transaction)
        return tuple(result)

    def transactions_by_category(
        self, category: EventCategory, *, start: datetime | None = None, end: datetime | None = None
    ) -> Tuple[Transaction, ...]:
        """Return transactions that match a specific category."""

        return self.filter_transactions(start=start, end=end, categories=(category,))

    def last_transaction(
        self,
        *,
        category: EventCategory | None = None,
        transaction_type: TransactionType | None = None,
    ) -> Optional[Transaction]:
        """Return the most recent transaction optionally matching filters."""

        for transaction in reversed(self._transactions):
            if category and transaction.category is not category:
                continue
            if transaction_type and transaction.type is not transaction_type:
                continue
            return transaction
        return None

    def export_transactions_csv(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        category: EventCategory | None = None,
    ) -> str:
        """Return a CSV export of the transaction ledger."""

        filtered = self.filter_transactions(start=start, end=end, categories=(category,) if category else None)
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["timestamp", "type", "category", "description", "amount", "balance"])
        for transaction in filtered:
            writer.writerow(
                [
                    transaction.timestamp.isoformat(),
                    transaction.type.value,
                    transaction.category.value if transaction.category else "",
                    transaction.description,
                    f"{transaction.amount:.2f}",
                    f"{transaction.balance_after:.2f}",
                ]
            )
        return buffer.getvalue()

    def _log_transaction(
        self,
        amount: Decimal,
        transaction_type: TransactionType,
        description: str,
        category: EventCategory | None = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Transaction:
        transaction = Transaction(
            amount=amount,
            type=transaction_type,
            description=description,
            balance_after=self._balance,
            category=category,
            metadata=dict(metadata or {}),
        )
        self._transactions.append(transaction)
        return transaction

    def _ensure_sufficient_funds(self, amount: Decimal) -> None:
        if self._balance < amount:
            raise InsufficientFundsError(
                f"Account '{self.child_name}' has insufficient funds for {format_currency(amount)}."
            )
