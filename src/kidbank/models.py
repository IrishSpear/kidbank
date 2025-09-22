"""Domain models used by the KidBank package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from .money import require_positive, to_decimal


class TransactionType(str, Enum):
    """Enumerates the supported types of account transactions."""

    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    REWARD = "reward"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    GOAL_CONTRIBUTION = "goal_contribution"


class EventCategory(str, Enum):
    """High level categories used for reporting and filtering events."""

    CHORE = "chore"
    REWARD = "reward"
    GOAL = "goal"
    INVEST = "invest"
    MANUAL = "manual"
    TRANSFER = "transfer"
    BONUS = "bonus"
    PENALTY = "penalty"


@dataclass(slots=True)
class Transaction:
    """Represents a single ledger entry for an :class:`~kidbank.account.Account`."""

    amount: Decimal
    type: TransactionType
    description: str
    balance_after: Decimal
    category: EventCategory | None = None
    metadata: Dict[str, str] = field(default_factory=dict)
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
    image_url: str = ""
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

    def milestone_reached(self, percentage: int) -> bool:
        """Return ``True`` once a progress milestone is achieved."""

        if percentage not in {25, 50, 75, 100}:
            raise ValueError("Milestones supported: 25, 50, 75 and 100 percent.")
        achieved = (self.progress() * Decimal(100)).quantize(Decimal("1"))
        return achieved >= Decimal(percentage)


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


@dataclass(slots=True)
class AuditEvent:
    """Represents an auditable admin action."""

    actor: str
    action: str
    target: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


class PayoutStatus(str, Enum):
    """Lifecycle for payouts that require soft-limit approvals."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class PayoutRequest:
    """Represents a payout awaiting parent approval."""

    request_id: str
    child_name: str
    amount: Decimal
    description: str
    created_by: str
    status: PayoutStatus = PayoutStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", to_decimal(self.amount))
        require_positive(self.amount)

    def approve(self, actor: str, *, when: datetime | None = None) -> None:
        self.status = PayoutStatus.APPROVED
        self.resolved_by = actor
        self.resolved_at = when or datetime.utcnow()

    def reject(self, actor: str, *, when: datetime | None = None) -> None:
        self.status = PayoutStatus.REJECTED
        self.resolved_by = actor
        self.resolved_at = when or datetime.utcnow()

    def cancel(self, *, when: datetime | None = None) -> None:
        self.status = PayoutStatus.CANCELLED
        self.resolved_at = when or datetime.utcnow()


@dataclass(slots=True)
class LeaderboardEntry:
    """Simple value object for leaderboard standings."""

    name: str
    score: int


@dataclass(slots=True)
class KidSummary:
    """Snapshot used by the admin dashboard quick cards."""

    completion_percentage: Decimal
    last_payout: Optional[datetime]
    next_allowance_eta: Optional[datetime]
    pending_chores: int


@dataclass(slots=True)
class FeatureFlag:
    """Feature toggle stored in the pseudo MetaKV store."""

    key: str
    enabled: bool
    description: str = ""


@dataclass(slots=True)
class ScheduledDigest:
    """Metadata describing an upcoming weekly digest notification."""

    recipient: str
    send_on: datetime
    summary: Mapping[str, Any]


@dataclass(slots=True)
class NFCEvent:
    """Represents an NFC tap recorded for a kid."""

    child_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    bonus_awarded: Decimal = Decimal("0.00")


@dataclass(slots=True)
class BackupMetadata:
    """Metadata describing a generated backup artefact."""

    backup_id: str
    created_at: datetime
    label: str
    size_bytes: int
