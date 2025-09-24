"""Chore scheduling, streak logic and pack handling for KidBank."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Dict, List, Mapping, Optional, Sequence
from uuid import uuid4


class Weekday(IntEnum):
    """Enum representing days of the week for scheduling."""

    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

    @classmethod
    def from_datetime(cls, moment: datetime) -> "Weekday":
        return cls(moment.weekday())


@dataclass(slots=True)
class TimeWindow:
    """Simple inclusive time window constraint for a chore."""

    start: time | None = None
    end: time | None = None

    def includes(self, moment: datetime) -> bool:
        if not self.start and not self.end:
            return True
        current = moment.time()
        if self.start and current < self.start:
            return False
        if self.end and current > self.end:
            return False
        return True


@dataclass(slots=True)
class ChoreSchedule:
    """Schedule describing when a chore should be available."""

    weekdays: frozenset[Weekday] = field(default_factory=frozenset)
    window: TimeWindow | None = None
    specific_dates: frozenset[date] | None = None

    def __post_init__(self) -> None:
        if self.specific_dates is not None:
            object.__setattr__(self, "specific_dates", frozenset(self.specific_dates))
        if self.weekdays is None:
            object.__setattr__(self, "weekdays", frozenset())

    def is_due(self, moment: datetime) -> bool:
        if self.specific_dates is not None:
            due_today = moment.date() in self.specific_dates
        elif self.weekdays:
            due_today = Weekday.from_datetime(moment) in self.weekdays
        else:
            due_today = True
        return due_today and (self.window is None or self.window.includes(moment))

    @classmethod
    def daily(cls) -> "ChoreSchedule":
        return cls(weekdays=frozenset(Weekday), window=None)

    @classmethod
    def on_dates(cls, dates: Sequence[date], *, window: TimeWindow | None = None) -> "ChoreSchedule":
        return cls(weekdays=frozenset(), window=window, specific_dates=frozenset(dates))


@dataclass(slots=True)
class ChoreCompletion:
    """Represents a single completion entry for a chore."""

    chore_name: str
    timestamp: datetime
    base_value: Decimal
    multiplier: Decimal
    proof: Optional[str]

    @property
    def awarded_value(self) -> Decimal:
        return (self.base_value * self.multiplier).quantize(Decimal("0.01"))


@dataclass(slots=True)
class Chore:
    """A chore with scheduling information and streak logic."""

    name: str
    value: Decimal
    schedule: ChoreSchedule
    requires_proof: bool = False
    proof_type: str | None = None  # "photo" or "note"
    pending_since: Optional[datetime] = None
    last_completed: Optional[datetime] = None
    streak: int = 0
    history: List[ChoreCompletion] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.requires_proof and not self.proof_type:
            raise ValueError("Proof type must be provided for special chores.")
        self.value = Decimal(self.value).quantize(Decimal("0.01"))

    def multiplier(self) -> Decimal:
        """Return the multiplier associated with the current streak."""

        if self.streak >= 7:
            bonus_steps = self.streak - 6
            return (Decimal("1.0") + (Decimal("0.1") * bonus_steps)).quantize(Decimal("0.01"))
        return Decimal("1.0")

    def current_value(self) -> Decimal:
        return (self.value * self.multiplier()).quantize(Decimal("0.01"))

    def mark_completed(self, *, at: Optional[datetime] = None, proof: Optional[str] = None) -> ChoreCompletion:
        moment = at or datetime.utcnow()
        if self.requires_proof and not proof:
            raise ValueError("Proof is required to complete this chore.")
        if proof and not self.requires_proof and not proof.strip():
            proof = None
        if not self.schedule.is_due(moment):
            # Allow completion but do not update streak when outside scheduled window
            self.pending_since = None
            completion = ChoreCompletion(
                chore_name=self.name,
                timestamp=moment,
                base_value=self.value,
                multiplier=Decimal("1.0"),
                proof=proof,
            )
            self.history.append(completion)
            self.last_completed = moment
            self.streak = 1
            return completion

        last = self.last_completed.date() if self.last_completed else None
        today = moment.date()
        if last and (today - last) == timedelta(days=1):
            self.streak += 1
        elif last == today:
            # repeat completion in same day keeps streak as-is
            pass
        else:
            self.streak = 1
        completion = ChoreCompletion(
            chore_name=self.name,
            timestamp=moment,
            base_value=self.value,
            multiplier=self.multiplier(),
            proof=proof,
        )
        self.history.append(completion)
        self.last_completed = moment
        self.pending_since = None
        return completion

    def advance_day(self, *, at: Optional[datetime] = None) -> bool:
        """Mark the chore as pending if due but not completed."""

        moment = at or datetime.utcnow()
        if self.schedule.is_due(moment):
            if self.last_completed and self.last_completed.date() == moment.date():
                self.pending_since = None
                return False
            if not self.pending_since or self.pending_since.date() != moment.date():
                self.pending_since = datetime.combine(moment.date(), time())
                return True
        else:
            # reset pending flag if outside schedule
            if self.pending_since and self.pending_since.date() < moment.date():
                self.pending_since = None
        return False

    def pending_for_hours(self, hours: int, *, at: Optional[datetime] = None) -> bool:
        if not self.pending_since:
            return False
        moment = at or datetime.utcnow()
        return moment - self.pending_since >= timedelta(hours=hours)


@dataclass(slots=True)
class ChorePack:
    """Group of chores that can be completed together."""

    name: str
    chore_names: Sequence[str]
    bonus_value: Decimal = Decimal("0.00")
    description: str = ""
    last_completed: Optional[datetime] = None

    def includes(self, chore_name: str) -> bool:
        return chore_name in self.chore_names


@dataclass(slots=True)
class GlobalChoreSubmission:
    """A pending submission for a shared chore."""

    child_name: str
    comment: str = ""
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    proof: Optional[str] = None


@dataclass(slots=True)
class GlobalChore:
    """Represents a chore that can be shared across multiple children."""

    name: str
    reward: Decimal
    max_claims: int
    description: str = ""
    active: bool = True
    submissions: Dict[str, GlobalChoreSubmission] = field(default_factory=dict)
    approved: Dict[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reward = Decimal(self.reward).quantize(Decimal("0.01"))
        if reward <= Decimal("0.00"):
            raise ValueError("Reward must be positive for a global chore.")
        if self.max_claims <= 0:
            raise ValueError("max_claims must be positive.")
        object.__setattr__(self, "reward", reward)

    def remaining_claims(self) -> int:
        return max(0, self.max_claims - len(self.approved))

    def submit(self, child_name: str, *, comment: str = "", proof: Optional[str] = None) -> GlobalChoreSubmission:
        if not self.active:
            raise ValueError("Global chore is no longer active.")
        submission = GlobalChoreSubmission(
            child_name=child_name,
            comment=comment.strip(),
            proof=proof,
        )
        self.submissions[child_name] = submission
        return submission

    def _split_evenly(self, count: int) -> List[Decimal]:
        if count <= 0:
            raise ValueError("Cannot split reward among zero participants.")
        total_cents = int((self.reward * 100).quantize(Decimal("1")))
        base = total_cents // count
        remainder = total_cents % count
        amounts: List[Decimal] = []
        for index in range(count):
            cents = base + (1 if index < remainder else 0)
            amounts.append(Decimal(cents) / Decimal(100))
        return amounts

    def approve(
        self,
        participants: Sequence[str],
        *,
        amount_override: Mapping[str, Decimal] | None = None,
    ) -> Dict[str, Decimal]:
        if not participants:
            raise ValueError("At least one participant must be selected for approval.")
        if len(participants) > self.max_claims:
            raise ValueError("Cannot approve more participants than allowed.")
        for child in participants:
            if child not in self.submissions:
                raise KeyError(f"No submission on file for '{child}'.")
        if amount_override:
            payout: Dict[str, Decimal] = {}
            total = Decimal("0.00")
            for child in participants:
                if child not in amount_override:
                    raise KeyError(f"Missing override amount for '{child}'.")
                value = Decimal(amount_override[child]).quantize(Decimal("0.01"))
                if value < Decimal("0.00"):
                    raise ValueError("Override amounts cannot be negative.")
                payout[child] = value
                total += value
            if total != self.reward:
                raise ValueError("Override amounts must total the reward value.")
        else:
            split = self._split_evenly(len(participants))
            payout = {child: amount for child, amount in zip(participants, split)}

        for child, amount in payout.items():
            self.approved[child] = amount
            self.submissions.pop(child, None)
        self.active = False
        return payout


class ChoreListingStatus(str, Enum):
    """Lifecycle states for marketplace listings."""

    OPEN = "open"
    CLAIMED = "claimed"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"



@dataclass(slots=True)
class ChoreListing:
    """A listing advertising a chore available for pickup by another child."""

    listing_id: str
    owner: str
    chore_name: str
    offer: Decimal
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ChoreListingStatus = field(default=ChoreListingStatus.OPEN)
    claimed_by: str | None = None
    claimed_at: datetime | None = None
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    final_payout: Decimal | None = None
    payout_note: str | None = None
    resolved_by: str | None = None
    pending_completion: ChoreCompletion | None = None

    def __post_init__(self) -> None:
        offer = Decimal(self.offer).quantize(Decimal("0.01"))
        if offer <= Decimal("0.00"):
            raise ValueError("Offer must be positive for a marketplace listing.")
        object.__setattr__(self, "offer", offer)

    def claim(self, child_name: str, *, when: datetime | None = None) -> None:
        if self.status is not ChoreListingStatus.OPEN:
            raise ValueError("Listing is not available to claim.")
        if child_name == self.owner:
            raise ValueError("Owner cannot claim their own chore listing.")
        self.status = ChoreListingStatus.CLAIMED
        self.claimed_by = child_name
        self.claimed_at = when or datetime.utcnow()

    def submit(self, child_name: str, completion: ChoreCompletion) -> None:
        if self.status is not ChoreListingStatus.CLAIMED:
            raise ValueError("Listing must be claimed before submission.")
        if child_name != self.claimed_by:
            raise ValueError("Only the child who claimed the listing can submit it.")
        self.status = ChoreListingStatus.SUBMITTED
        self.submitted_at = completion.timestamp
        self.pending_completion = completion

    def cancel(self, child_name: str, *, when: datetime | None = None) -> None:
        if child_name != self.owner:
            raise ValueError("Only the owner can cancel the listing.")
        if self.status is not ChoreListingStatus.OPEN:
            raise ValueError("Only open listings can be cancelled.")
        self.status = ChoreListingStatus.CANCELLED
        self.cancelled_at = when or datetime.utcnow()

    def approve(self, *, payout: Decimal, resolved_by: str, note: str | None = None, when: datetime | None = None) -> None:
        if self.status is not ChoreListingStatus.SUBMITTED:
            raise ValueError("Listing must be submitted before approval.")
        resolution_time = when or datetime.utcnow()
        self.status = ChoreListingStatus.COMPLETED
        self.completed_at = resolution_time
        self.final_payout = Decimal(payout).quantize(Decimal("0.01"))
        self.payout_note = note
        self.resolved_by = resolved_by
        self.pending_completion = None

    def reject(self, *, resolved_by: str, note: str | None = None, when: datetime | None = None) -> None:
        if self.status is not ChoreListingStatus.SUBMITTED:
            raise ValueError("Listing must be submitted before it can be rejected.")
        resolution_time = when or datetime.utcnow()
        self.status = ChoreListingStatus.REJECTED
        self.completed_at = resolution_time
        self.final_payout = Decimal("0.00")
        self.payout_note = note
        self.resolved_by = resolved_by
        self.pending_completion = None

    @property
    def is_active(self) -> bool:
        return self.status in (
            ChoreListingStatus.OPEN,
            ChoreListingStatus.CLAIMED,
            ChoreListingStatus.SUBMITTED,
        )


class ChoreMarketplace:
    """Manage marketplace listings for chores."""

    def __init__(self) -> None:
        self._listings: Dict[str, ChoreListing] = {}

    def create_listing(self, owner: str, chore_name: str, offer: Decimal) -> ChoreListing:
        if self.active_listing_for(owner, chore_name):
            raise ValueError(f"Chore '{chore_name}' already has an active listing.")
        listing = ChoreListing(
            listing_id=str(uuid4()),
            owner=owner,
            chore_name=chore_name,
            offer=offer,
        )
        self._listings[listing.listing_id] = listing
        return listing

    def listing(self, listing_id: str) -> ChoreListing:
        try:
            return self._listings[listing_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Marketplace listing '{listing_id}' does not exist.") from exc

    def listings(self, *, include_closed: bool = False) -> Sequence[ChoreListing]:
        if include_closed:
            return tuple(self._listings.values())
        return tuple(listing for listing in self._listings.values() if listing.is_active)

    def claim(self, listing_id: str, child_name: str, *, when: datetime | None = None) -> ChoreListing:
        listing = self.listing(listing_id)
        listing.claim(child_name, when=when)
        return listing

    def submit(self, listing_id: str, child_name: str, completion: ChoreCompletion) -> ChoreListing:
        listing = self.listing(listing_id)
        listing.submit(child_name, completion)
        return listing

    def cancel(self, listing_id: str, owner: str, *, when: datetime | None = None) -> ChoreListing:
        listing = self.listing(listing_id)
        listing.cancel(owner, when=when)
        return listing

    def approve(
        self,
        listing_id: str,
        *,
        payout: Decimal,
        resolved_by: str,
        note: str | None = None,
        when: datetime | None = None,
    ) -> ChoreListing:
        listing = self.listing(listing_id)
        listing.approve(payout=payout, resolved_by=resolved_by, note=note, when=when)
        return listing

    def reject(
        self,
        listing_id: str,
        *,
        resolved_by: str,
        note: str | None = None,
        when: datetime | None = None,
    ) -> ChoreListing:
        listing = self.listing(listing_id)
        listing.reject(resolved_by=resolved_by, note=note, when=when)
        return listing

    def active_listing_for(self, owner: str, chore_name: str) -> Optional[ChoreListing]:
        for listing in self._listings.values():
            if (
                listing.owner == owner
                and listing.chore_name == chore_name
                and listing.is_active
            ):
                return listing
        return None

    def submissions(self) -> Sequence[ChoreListing]:
        return tuple(
            listing
            for listing in self._listings.values()
            if listing.status is ChoreListingStatus.SUBMITTED
        )


class ChoreBoard:
    """Manage chores and completions for a single child."""

    def __init__(self) -> None:
        self._chores: Dict[str, Chore] = {}
        self._packs: Dict[str, ChorePack] = {}
        self._completions: List[ChoreCompletion] = []

    def add_chore(self, chore: Chore) -> None:
        if chore.name in self._chores:
            raise ValueError(f"Chore '{chore.name}' already exists.")
        self._chores[chore.name] = chore

    def chores(self) -> Sequence[Chore]:
        return tuple(self._chores.values())

    def remove_chore(self, name: str) -> None:
        self._chores.pop(name, None)

    def get(self, name: str) -> Chore:
        try:
            return self._chores[name]
        except KeyError as exc:
            raise KeyError(f"Unknown chore '{name}'.") from exc

    def schedule_pack(self, pack: ChorePack) -> None:
        if pack.name in self._packs:
            raise ValueError(f"Pack '{pack.name}' already exists.")
        for name in pack.chore_names:
            if name not in self._chores:
                raise ValueError(f"Chore '{name}' must exist before creating a pack.")
        self._packs[pack.name] = pack

    def deactivate_pack(self, name: str) -> None:
        self._packs.pop(name, None)

    def complete_chore(
        self,
        name: str,
        *,
        at: Optional[datetime] = None,
        proof: Optional[str] = None,
    ) -> ChoreCompletion:
        chore = self.get(name)
        completion = chore.mark_completed(at=at, proof=proof)
        self._completions.append(completion)
        return completion

    def complete_pack(self, name: str, *, at: Optional[datetime] = None) -> Decimal:
        pack = self._packs.get(name)
        if not pack:
            raise KeyError(f"Pack '{name}' does not exist.")
        moment = at or datetime.utcnow()
        chore_dates = [self._chores[chore_name].last_completed for chore_name in pack.chore_names]
        if not all(chore_dates):
            raise ValueError("All chores in the pack must be completed before claiming the pack bonus.")
        reference_date = moment.date()
        if not all(entry.date() == reference_date for entry in chore_dates if entry):
            raise ValueError("Pack bonus must be claimed on the day chores were completed.")
        pack.last_completed = moment
        bonus = Decimal(pack.bonus_value).quantize(Decimal("0.01"))
        if bonus <= Decimal("0.00"):
            return Decimal("0.00")
        completion = ChoreCompletion(
            chore_name=pack.name,
            timestamp=moment,
            base_value=bonus,
            multiplier=Decimal("1.0"),
            proof=None,
        )
        self._completions.append(completion)
        return completion.awarded_value

    def auto_republish(self, *, at: Optional[datetime] = None) -> Sequence[str]:
        moment = at or datetime.utcnow()
        republished: List[str] = []
        for chore in self._chores.values():
            if chore.advance_day(at=moment):
                republished.append(chore.name)
        return tuple(republished)

    def pending(self, *, at: Optional[datetime] = None) -> Sequence[Chore]:
        moment = at or datetime.utcnow()
        return tuple(chore for chore in self._chores.values() if chore.advance_day(at=moment) or chore.pending_since)

    def pending_overdue(self, hours: int, *, at: Optional[datetime] = None) -> Sequence[Chore]:
        moment = at or datetime.utcnow()
        overdue: List[Chore] = []
        for chore in self._chores.values():
            chore.advance_day(at=moment)
            if chore.pending_for_hours(hours, at=moment):
                overdue.append(chore)
        return tuple(overdue)

    def completions_since(self, since: datetime) -> Sequence[ChoreCompletion]:
        return tuple(entry for entry in self._completions if entry.timestamp >= since)

    def streak(self, name: str) -> int:
        return self.get(name).streak

    def multiplier(self, name: str) -> Decimal:
        return self.get(name).multiplier()

    def special_chores(self) -> Sequence[Chore]:
        return tuple(chore for chore in self._chores.values() if chore.requires_proof)

    def completion_rate(self, *, days: int = 7, at: Optional[datetime] = None) -> Decimal:
        if not self._chores:
            return Decimal("0")
        moment = at or datetime.utcnow()
        start = moment - timedelta(days=days)
        completions = [entry for entry in self._completions if entry.timestamp >= start]
        expected = len(self._chores) * days
        if expected == 0:
            return Decimal("0")
        ratio = Decimal(len(completions)) / Decimal(expected)
        return max(Decimal("0"), min(ratio.quantize(Decimal("0.01")), Decimal("1.00")))

    def total_completions(self) -> int:
        return len(self._completions)

    def leaderboard_score(self) -> int:
        return sum(int(entry.awarded_value) for entry in self._completions)

    def goal_gallery(self) -> Sequence[str]:
        return tuple(self._chores)


__all__ = [
    "Chore",
    "ChoreBoard",
    "ChoreCompletion",
    "ChorePack",
    "ChoreSchedule",
    "TimeWindow",
    "Weekday",
    "GlobalChore",
    "GlobalChoreSubmission",
    "ChoreListing",
    "ChoreListingStatus",
    "ChoreMarketplace",
]
