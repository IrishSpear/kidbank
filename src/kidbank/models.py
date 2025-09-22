"""Domain models used by the KidBank package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from .money import require_positive, to_decimal


class TransactionType(str, Enum):
    """Enumerates the supported types of account transactions."""

    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    REWARD = "reward"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    GOAL_CONTRIBUTION = "goal_contribution"


@dataclass(slots=True)
class Transaction:
    """Represents a single ledger entry for an :class:`~kidbank.account.Account`."""

    amount: Decimal
    type: TransactionType
    description: str
    balance_after: Decimal
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", to_decimal(self.amount))
        object.__setattr__(self, "balance_after", to_decimal(self.balance_after))


@dataclass(slots=True)
class Goal:
    """Represents a savings goal that children can contribute towards."""

    name: str
    target_amount: Decimal
    description: str = ""
    saved_amount: Decimal = Decimal("0.00")
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        target = to_decimal(self.target_amount)
        saved = to_decimal(self.saved_amount)
        require_positive(target)
        if saved < Decimal("0"):
            raise ValueError("saved_amount cannot be negative.")
        object.__setattr__(self, "target_amount", target)
        object.__setattr__(self, "saved_amount", saved)

    def contribute(self, amount: Decimal) -> Decimal:
        """Increase the amount saved towards the goal."""

        increment = to_decimal(amount)
        require_positive(increment)
        new_total = self.saved_amount + increment
        object.__setattr__(self, "saved_amount", new_total)
        return increment

    @property
    def remaining(self) -> Decimal:
        """Return the amount still required to achieve the goal."""

        remainder = self.target_amount - self.saved_amount
        return remainder if remainder > Decimal("0") else Decimal("0.00")

    @property
    def is_complete(self) -> bool:
        """True when the goal has been fully funded."""

        return self.saved_amount >= self.target_amount

    def progress(self) -> Decimal:
        """Return the progress towards the goal as a decimal ratio (0-1)."""

        ratio = self.saved_amount / self.target_amount
        return ratio.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


@dataclass(slots=True)
class Reward:
    """Represents a redeemed reward item."""

    name: str
    cost: Decimal
    description: str = ""
    redeemed_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        cost = to_decimal(self.cost)
        require_positive(cost)
        object.__setattr__(self, "cost", cost)
