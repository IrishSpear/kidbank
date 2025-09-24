"""High level service for coordinating multiple KidBank accounts."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from .account import Account
from .admin import AuditLog, FeatureFlagRegistry, UndoManager
from .api import ApiExporter, WebhookDispatcher
from .chores import (
    Chore,
    ChoreBoard,
    ChoreMarketplace,
    ChorePack,
    ChoreSchedule,
    ChoreListing,
    ChoreListingStatus,
    GlobalChore,
    GlobalChoreSubmission,
    TimeWindow,
    Weekday,
)
from .exceptions import AccountNotFoundError, DuplicateAccountError, InsufficientFundsError
from .i18n import Translator
from .investing import CertificateOfDeposit, InvestmentPortfolio
from .models import (
    EventCategory,
    Goal,
    KidSummary,
    LeaderboardEntry,
    NFCEvent,
    Reward,
    PayoutRequest,
    PayoutStatus,
    TransferRequest,
    TransferRequestStatus,
    ScheduledDigest,
    Transaction,
    TransactionType,
    BackupMetadata,
)
from .money import AmountLike, format_currency, to_decimal
from .notifications import Notification, NotificationCenter, NotificationChannel, NotificationType
from .ops import BackupManager, HealthMonitor, StructuredLogger
from .security import AuthManager


class KidBank:
    """Manage accounts, chores, investing and administrative workflows."""

    __slots__ = (
        "_accounts",
        "_chores",
        "_global_chores",
        "_portfolios",
        "_notifications",
        "_audit_log",
        "_undo",
        "_feature_flags",
        "_webhooks",
        "_auth",
        "_backups",
        "_health",
        "_logger",
        "_pending_payouts",
        "_soft_limits",
        "_parent_contacts",
        "_kid_contacts",
        "_goal_milestones",
        "_translator",
        "_api",
        "_nfc_events",
        "_badge_thresholds",
        "_earned_badges",
        "_next_allowance_eta",
        "_default_soft_limit",
        "_transfer_requests",
        "_marketplace",
    )

    def __init__(self, *, soft_limit: AmountLike = Decimal("20")) -> None:
        self._accounts: Dict[str, Account] = {}
        self._chores: Dict[str, ChoreBoard] = {}
        self._global_chores: Dict[str, GlobalChore] = {}
        self._portfolios: Dict[str, InvestmentPortfolio] = {}
        self._notifications = NotificationCenter()
        self._audit_log = AuditLog()
        self._undo = UndoManager()
        self._feature_flags = FeatureFlagRegistry(
            {
                "investing": True,
                "badges": True,
                "sms": True,
                "leaderboard": True,
            }
        )
        self._webhooks = WebhookDispatcher()
        self._auth = AuthManager()
        self._backups = BackupManager()
        self._health = HealthMonitor()
        self._logger = StructuredLogger()
        self._pending_payouts: Dict[str, PayoutRequest] = {}
        self._soft_limits: Dict[str, Decimal] = {}
        self._parent_contacts: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._kid_contacts: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._goal_milestones: Dict[tuple[str, str], set[int]] = defaultdict(set)
        self._translator = Translator()
        self._api = ApiExporter()
        self._nfc_events: list[NFCEvent] = []
        self._badge_thresholds: Dict[str, int] = {
            "Rising Star": 5,
            "Chore Champion": 25,
            "Legend": 50,
        }
        self._earned_badges: Dict[str, set[str]] = defaultdict(set)
        self._next_allowance_eta: Dict[str, Optional[datetime]] = {}
        self._default_soft_limit = to_decimal(soft_limit)
        self._transfer_requests: Dict[str, TransferRequest] = {}
        self._marketplace = ChoreMarketplace()

    # ------------------------------------------------------------------
    # Core account operations
    # ------------------------------------------------------------------
    def create_account(self, child_name: str, *, starting_balance: AmountLike = 0) -> Account:
        if child_name in self._accounts:
            raise DuplicateAccountError(f"Account '{child_name}' already exists.")
        account = Account(child_name, starting_balance=starting_balance)
        self._accounts[child_name] = account
        self._chores[child_name] = ChoreBoard()
        self._portfolios[child_name] = InvestmentPortfolio()
        self._soft_limits[child_name] = self._default_soft_limit
        self._next_allowance_eta[child_name] = None
        self._audit_log.record("system", "create_account", child_name)
        self._logger.log("account_created", child=child_name, balance=float(account.balance))
        self._webhooks.dispatch({"event": "account_created", "child": child_name})
        return account

    def list_accounts(self) -> Tuple[str, ...]:
        return tuple(sorted(self._accounts))

    def has_account(self, child_name: str) -> bool:
        return child_name in self._accounts

    def get_account(self, child_name: str) -> Account:
        try:
            return self._accounts[child_name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise AccountNotFoundError(f"Account '{child_name}' does not exist.") from exc

    def deposit(
        self,
        child_name: str,
        amount: AmountLike,
        description: str = "Deposit",
        *,
        category: EventCategory | None = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Transaction:
        transaction = self.get_account(child_name).deposit(
            amount,
            description,
            category=category,
            metadata=metadata,
        )
        self._after_transaction(child_name, transaction)
        swept = self._portfolios[child_name].record_deposit(transaction.amount)
        if swept > Decimal("0"):
            self._logger.log("auto_sweep", child=child_name, amount=float(swept))
        self._notify_payout(child_name, transaction)
        return transaction

    def withdraw(
        self,
        child_name: str,
        amount: AmountLike,
        description: str = "Withdrawal",
        *,
        category: EventCategory | None = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Transaction:
        transaction = self.get_account(child_name).withdraw(
            amount,
            description,
            category=category,
            metadata=metadata,
        )
        self._after_transaction(child_name, transaction)
        return transaction

    def transfer(
        self,
        sender: str,
        recipient: str,
        amount: AmountLike,
        description: str | None = None,
        *,
        comment: str | None = None,
    ) -> tuple[Transaction, Transaction]:
        source = self.get_account(sender)
        destination = self.get_account(recipient)
        note = description or comment or f"Transfer to {recipient}"
        metadata = {"comment": comment} if comment else None
        outgoing, incoming = source.transfer_to(
            destination,
            amount,
            note,
            metadata=metadata,
        )
        self._after_transaction(sender, outgoing)
        self._after_transaction(recipient, incoming)
        return outgoing, incoming

    def request_money(
        self,
        requester: str,
        from_child: str,
        amount: AmountLike,
        *,
        comment: str = "",
    ) -> TransferRequest:
        if requester == from_child:
            raise ValueError("Cannot request money from the same account.")
        self.get_account(requester)
        self.get_account(from_child)
        value = to_decimal(amount)
        request = TransferRequest(
            request_id=str(uuid4()),
            sender=from_child,
            recipient=requester,
            amount=value,
            comment=comment.strip(),
        )
        self._transfer_requests[request.request_id] = request
        self._audit_log.record(requester, "request_transfer", request.request_id)
        return request

    def transfer_requests(
        self,
        *,
        for_child: str | None = None,
        status: TransferRequestStatus | None = None,
    ) -> Sequence[TransferRequest]:
        requests = self._transfer_requests.values()
        if for_child is not None:
            requests = [
                request
                for request in requests
                if request.sender == for_child or request.recipient == for_child
            ]
        if status is not None:
            requests = [request for request in requests if request.status is status]
        return tuple(requests)

    def respond_money_request(
        self,
        request_id: str,
        *,
        responder: str,
        approve: bool,
    ) -> TransferRequest:
        request = self._transfer_requests[request_id]
        if request.status is not TransferRequestStatus.PENDING:
            return request
        if responder != request.sender:
            raise PermissionError("Only the requested sender may respond.")
        if approve:
            request.approve(responder)
            self.transfer(
                request.sender,
                request.recipient,
                request.amount,
                description=request.comment or None,
                comment=request.comment or None,
            )
            self._audit_log.record(responder, "approve_transfer_request", request_id)
        else:
            request.decline(responder)
            self._audit_log.record(responder, "decline_transfer_request", request_id)
        return request

    def redeem_reward(
        self,
        child_name: str,
        *,
        name: str,
        cost: AmountLike,
        description: str = "",
    ) -> Reward:
        reward = self.get_account(child_name).redeem_reward(name, cost, description=description)
        self._audit_log.record("guardian", "redeem_reward", f"{child_name}:{name}")
        self._logger.log("reward_redeemed", child=child_name, reward=name, cost=float(reward.cost))
        return reward

    def add_goal(
        self,
        child_name: str,
        *,
        name: str,
        target_amount: AmountLike,
        description: str = "",
        image_url: str = "",
    ) -> Goal:
        goal = self.get_account(child_name).add_goal(
            name,
            target_amount,
            description=description,
            image_url=image_url,
        )
        self._audit_log.record("guardian", "add_goal", f"{child_name}:{name}")
        self._logger.log("goal_added", child=child_name, goal=name, target=float(goal.target_amount))
        return goal

    def contribute_to_goal(
        self,
        child_name: str,
        *,
        name: str,
        amount: AmountLike,
        description: str | None = None,
    ) -> Goal:
        goal = self.get_account(child_name).contribute_to_goal(name, amount, description=description)
        self._audit_log.record("guardian", "contribute_goal", f"{child_name}:{name}")
        self._logger.log("goal_contribution", child=child_name, goal=name, saved=float(goal.saved_amount))
        self._check_goal_milestones(child_name, goal)
        return goal

    def total_balance(self) -> Decimal:
        return sum((account.balance for account in self._accounts.values()), Decimal("0.00"))

    def summary(self) -> str:
        if not self._accounts:
            return "No accounts have been created yet."
        lines = ["KidBank summary:"]
        for name in sorted(self._accounts):
            account = self._accounts[name]
            lines.append(f"- {account.child_name}: {format_currency(account.balance)}")
        lines.append(f"Total balance: {format_currency(self.total_balance())}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Chore management and gamification
    # ------------------------------------------------------------------
    def schedule_chore(
        self,
        child_name: str,
        *,
        name: str,
        value: AmountLike,
        weekdays: Sequence[int | Weekday] | Weekday | int | None = None,
        dates: Sequence[date] | date | None = None,
        after: Optional[TimeWindow | Tuple[int, int]] = None,
        before: Optional[Tuple[int, int]] = None,
        requires_proof: bool = False,
        proof_type: str | None = None,
    ) -> Chore:
        board = self._chores[child_name]
        if dates is not None:
            if isinstance(dates, date):
                date_values = {dates}
            else:
                date_values = {d for d in dates}
        else:
            date_values = None

        if weekdays is None and date_values is None:
            weekday_iterable = Weekday
        elif weekdays is None:
            weekday_iterable = []
        elif isinstance(weekdays, (Weekday, int)):
            weekday_iterable = [weekdays]
        else:
            weekday_iterable = weekdays

        weekday_set = frozenset(Weekday(day) for day in weekday_iterable)
        window: TimeWindow | None
        if isinstance(after, TimeWindow):
            window = after
        elif after or before:
            start = None
            end = None
            if isinstance(after, tuple):
                start = datetime.utcnow().replace(hour=after[0], minute=after[1], second=0, microsecond=0).time()
            if isinstance(before, tuple):
                end = datetime.utcnow().replace(hour=before[0], minute=before[1], second=0, microsecond=0).time()
            window = TimeWindow(start=start, end=end)
        else:
            window = None
        chore = Chore(
            name=name,
            value=to_decimal(value),
            schedule=ChoreSchedule(
                weekdays=weekday_set,
                window=window,
                specific_dates=frozenset(date_values) if date_values is not None else None,
            ),
            requires_proof=requires_proof,
            proof_type=proof_type,
        )
        board.add_chore(chore)
        self._audit_log.record("guardian", "add_chore", f"{child_name}:{name}")
        self._logger.log("chore_created", child=child_name, chore=name, value=float(chore.value))
        return chore

    def complete_chore(
        self,
        child_name: str,
        name: str,
        *,
        proof: Optional[str] = None,
        at: Optional[datetime] = None,
    ) -> Decimal:
        completion = self._chores[child_name].complete_chore(name, at=at, proof=proof)
        metadata = {"chore": name, "multiplier": str(completion.multiplier)}
        transaction = self.deposit(
            child_name,
            completion.awarded_value,
            f"Chore completed: {name}",
            category=EventCategory.CHORE,
            metadata=metadata,
        )
        self._logger.log(
            "chore_completed",
            child=child_name,
            chore=name,
            awarded=float(completion.awarded_value),
            multiplier=float(completion.multiplier),
        )
        self._webhooks.dispatch(
            {
                "event": "chore_completed",
                "child": child_name,
                "chore": name,
                "value": float(completion.awarded_value),
                "transaction_id": transaction.timestamp.isoformat(),
            }
        )
        self.evaluate_badges(child_name)
        return completion.awarded_value

    def schedule_pack(
        self,
        child_name: str,
        *,
        name: str,
        chore_names: Sequence[str],
        bonus_value: AmountLike,
        description: str = "",
    ) -> ChorePack:
        pack = ChorePack(name=name, chore_names=tuple(chore_names), bonus_value=to_decimal(bonus_value), description=description)
        self._chores[child_name].schedule_pack(pack)
        self._audit_log.record("guardian", "add_pack", f"{child_name}:{name}")
        return pack

    def complete_pack(self, child_name: str, name: str, *, at: Optional[datetime] = None) -> Decimal:
        bonus = self._chores[child_name].complete_pack(name, at=at)
        if bonus > Decimal("0"):
            self.deposit(
                child_name,
                bonus,
                f"Pack bonus: {name}",
                category=EventCategory.BONUS,
                metadata={"pack": name},
            )
        return bonus

    def create_global_chore(
        self,
        *,
        name: str,
        reward: AmountLike,
        max_claims: int,
        description: str = "",
    ) -> GlobalChore:
        if name in self._global_chores:
            raise ValueError(f"Global chore '{name}' already exists.")
        chore = GlobalChore(
            name=name,
            reward=to_decimal(reward),
            max_claims=max_claims,
            description=description,
        )
        self._global_chores[name] = chore
        return chore

    def global_chores(self, *, include_inactive: bool = False) -> Sequence[GlobalChore]:
        return tuple(
            chore
            for chore in self._global_chores.values()
            if include_inactive or chore.active
        )

    def submit_global_chore(
        self,
        child_name: str,
        chore_name: str,
        *,
        comment: str = "",
        proof: Optional[str] = None,
    ) -> GlobalChoreSubmission:
        chore = self._global_chores.get(chore_name)
        if not chore:
            raise KeyError(f"Global chore '{chore_name}' does not exist.")
        submission = chore.submit(child_name, comment=comment, proof=proof)
        return submission

    def global_chore_submissions(self, chore_name: str) -> Sequence[GlobalChoreSubmission]:
        chore = self._global_chores.get(chore_name)
        if not chore:
            raise KeyError(f"Global chore '{chore_name}' does not exist.")
        return tuple(chore.submissions.values())

    def approve_global_chore(
        self,
        chore_name: str,
        participants: Sequence[str],
        *,
        amount_override: Mapping[str, AmountLike] | None = None,
        approved_by: str = "guardian",
    ) -> Dict[str, Decimal]:
        chore = self._global_chores.get(chore_name)
        if not chore:
            raise KeyError(f"Global chore '{chore_name}' does not exist.")
        override_decimals: Mapping[str, Decimal] | None = None
        if amount_override is not None:
            override_decimals = {child: to_decimal(amount) for child, amount in amount_override.items()}
        payouts = chore.approve(participants, amount_override=override_decimals)
        for child, amount in payouts.items():
            metadata = {"global_chore": chore.name, "approved_by": approved_by}
            self.deposit(
                child,
                amount,
                f"Global chore: {chore.name}",
                category=EventCategory.CHORE,
                metadata=metadata,
            )
        self._audit_log.record(approved_by, "approve_global_chore", chore_name)
        return payouts

    def marketplace_listings(self, *, include_closed: bool = False) -> Sequence[ChoreListing]:
        """Return marketplace listings, optionally including closed ones."""

        return self._marketplace.listings(include_closed=include_closed)

    def list_marketplace_chore(
        self,
        owner: str,
        chore_name: str,
        *,
        offer: AmountLike,
    ) -> ChoreListing:
        """Publish one of the owner's chores to the marketplace."""

        self.get_account(owner)
        board = self._chores[owner]
        board.get(chore_name)
        if self._marketplace.active_listing_for(owner, chore_name):
            raise ValueError(f"Chore '{chore_name}' already has an active marketplace listing.")
        offer_value = to_decimal(offer)
        if offer_value <= Decimal("0.00"):
            raise ValueError("Marketplace offer must be positive.")
        escrow = self.withdraw(
            owner,
            offer_value,
            f"Marketplace listing escrow: {chore_name}",
            category=EventCategory.CHORE,
            metadata={"marketplace": "escrow", "chore": chore_name},
        )
        listing = self._marketplace.create_listing(owner, chore_name, offer_value)
        escrow.metadata["listing"] = listing.listing_id
        self._audit_log.record(owner, "marketplace_list", f"{chore_name}:{offer_value}")
        self._logger.log("marketplace_listed", owner=owner, chore=chore_name, offer=float(offer_value))
        return listing

    def claim_marketplace_chore(self, child_name: str, listing_id: str) -> ChoreListing:
        """Allow another child to claim an open marketplace listing."""

        self.get_account(child_name)
        listing = self._marketplace.claim(listing_id, child_name)
        self._audit_log.record(child_name, "marketplace_claim", listing.chore_name)
        self._logger.log("marketplace_claimed", listing=listing_id, child=child_name)
        return listing

    def complete_marketplace_chore(
        self,
        child_name: str,
        listing_id: str,
        *,
        proof: Optional[str] = None,
        at: Optional[datetime] = None,
    ) -> Decimal:
        """Submit a marketplace chore completion for administrator approval."""

        self.get_account(child_name)
        listing = self._marketplace.listing(listing_id)
        if listing.status is not ChoreListingStatus.CLAIMED:
            raise ValueError("Marketplace listing is not ready for completion.")
        if listing.claimed_by != child_name:
            raise ValueError("This marketplace listing was claimed by another child.")
        board = self._chores[listing.owner]
        completion = board.complete_chore(listing.chore_name, at=at, proof=proof)
        self._marketplace.submit(listing_id, child_name, completion)
        total = (completion.awarded_value + listing.offer).quantize(Decimal("0.01"))
        self._audit_log.record(child_name, "marketplace_submit", listing.chore_name)
        self._logger.log(
            "marketplace_submitted",
            listing=listing.listing_id,
            worker=child_name,
            owner=listing.owner,
            expected_payout=float(total),
        )
        self.evaluate_badges(listing.owner)
        return total

    def cancel_marketplace_listing(self, owner: str, listing_id: str) -> ChoreListing:
        """Cancel an open marketplace listing and refund the escrowed funds."""

        self.get_account(owner)
        listing = self._marketplace.listing(listing_id)
        if listing.owner != owner:
            raise ValueError("Only the owner can cancel this marketplace listing.")
        self._marketplace.cancel(listing_id, owner)
        self.deposit(
            owner,
            listing.offer,
            f"Marketplace listing cancelled: {listing.chore_name}",
            category=EventCategory.CHORE,
            metadata={"marketplace": "refund", "listing": listing.listing_id, "chore": listing.chore_name},
        )
        self._audit_log.record(owner, "marketplace_cancel", listing.chore_name)
        self._logger.log("marketplace_cancelled", owner=owner, listing=listing.listing_id)
        return listing

    def pending_marketplace_submissions(self) -> Sequence[ChoreListing]:
        """Return marketplace listings waiting on administrator approval."""

        return self._marketplace.submissions()

    def approve_marketplace_submission(
        self,
        listing_id: str,
        *,
        approver: str,
        payout_override: AmountLike | None = None,
        note: str | None = None,
    ) -> Decimal:
        listing = self._marketplace.listing(listing_id)
        if listing.status is not ChoreListingStatus.SUBMITTED:
            raise ValueError("Marketplace submission is not awaiting approval.")
        if not listing.pending_completion:
            raise ValueError("Marketplace submission details were missing.")
        completion = listing.pending_completion
        default_total = (completion.awarded_value + listing.offer).quantize(Decimal("0.01"))
        if payout_override is None:
            final_total = default_total
        else:
            final_total = to_decimal(payout_override).quantize(Decimal("0.01"))
            if final_total < Decimal("0.00"):
                raise ValueError("Marketplace payout override cannot be negative.")
        award_value = completion.awarded_value
        offer_component = max(final_total - award_value, Decimal("0.00"))
        offer_delta = offer_component - listing.offer
        resolution_time = datetime.utcnow()
        if offer_delta > Decimal("0.00"):
            # Withdraw additional funds from the owner if the payout exceeds the escrowed offer.
            self.withdraw(
                listing.owner,
                offer_delta,
                f"Marketplace offer increase: {listing.chore_name}",
                category=EventCategory.CHORE,
                metadata={"marketplace": "offer_increase", "listing": listing.listing_id},
            )
        elif offer_delta < Decimal("0.00"):
            self.deposit(
                listing.owner,
                -offer_delta,
                f"Marketplace offer refund: {listing.chore_name}",
                category=EventCategory.CHORE,
                metadata={"marketplace": "offer_refund", "listing": listing.listing_id},
            )
        metadata = {
            "marketplace": "payout",
            "listing": listing.listing_id,
            "chore": listing.chore_name,
            "owner": listing.owner,
            "worker": listing.claimed_by or "",
            "award": str(award_value),
            "offer": str(offer_component),
            "final": str(final_total),
        }
        if note:
            metadata["note"] = note
        if final_total > Decimal("0.00"):
            transaction = self.deposit(
                listing.claimed_by or listing.owner,
                final_total,
                f"Marketplace chore payout: {listing.chore_name}",
                category=EventCategory.CHORE,
                metadata=metadata,
            )
            transaction_id = transaction.timestamp.isoformat()
        else:
            transaction_id = ""
        self._marketplace.approve(
            listing_id,
            payout=final_total,
            resolved_by=approver,
            note=note,
            when=resolution_time,
        )
        self._audit_log.record(approver, "marketplace_approve", f"{listing.chore_name}:{final_total}")
        self._logger.log(
            "marketplace_approved",
            listing=listing.listing_id,
            worker=listing.claimed_by,
            owner=listing.owner,
            payout=float(final_total),
            transaction_id=transaction_id,
        )
        self.evaluate_badges(listing.owner)
        if listing.claimed_by:
            self.evaluate_badges(listing.claimed_by)
        return final_total

    def reject_marketplace_submission(
        self,
        listing_id: str,
        *,
        approver: str,
        note: str | None = None,
    ) -> None:
        listing = self._marketplace.listing(listing_id)
        if listing.status is not ChoreListingStatus.SUBMITTED:
            raise ValueError("Marketplace submission is not awaiting approval.")
        resolution_time = datetime.utcnow()
        if listing.offer > Decimal("0.00"):
            self.deposit(
                listing.owner,
                listing.offer,
                f"Marketplace listing rejected: {listing.chore_name}",
                category=EventCategory.CHORE,
                metadata={"marketplace": "reject_refund", "listing": listing.listing_id},
            )
        self._marketplace.reject(
            listing_id,
            resolved_by=approver,
            note=note,
            when=resolution_time,
        )
        self._audit_log.record(approver, "marketplace_reject", listing.chore_name)
        self._logger.log(
            "marketplace_rejected",
            listing=listing.listing_id,
            worker=listing.claimed_by,
            owner=listing.owner,
            note=note or "",
        )

    def auto_republish_chores(self, *, at: Optional[datetime] = None) -> Dict[str, Sequence[str]]:
        republished: Dict[str, Sequence[str]] = {}
        for child, board in self._chores.items():
            refreshed = board.auto_republish(at=at)
            if refreshed:
                republished[child] = refreshed
        if republished:
            self._logger.log("chores_republished", details=republished)
        return republished

    def send_pending_chore_alerts(
        self,
        *,
        hours: int,
        at: Optional[datetime] = None,
    ) -> Sequence[Notification]:
        alerts: list[Notification] = []
        for child, board in self._chores.items():
            overdue = board.pending_overdue(hours, at=at)
            if not overdue:
                continue
            contact = self._parent_contacts.get(child, {})
            recipient = contact.get("email") or contact.get("sms")
            if not recipient:
                continue
            channel = NotificationChannel.EMAIL if contact.get("email") else NotificationChannel.SMS
            for chore in overdue:
                notification = Notification(
                    recipient=recipient,
                    channel=channel,
                    type=NotificationType.CHORE_REMINDER,
                    subject=f"{child} has a pending chore",
                    body=f"Chore '{chore.name}' has been waiting since {chore.pending_since:%Y-%m-%d %H:%M}.",
                    metadata={"child": child, "chore": chore.name},
                )
                self._notifications.queue(notification)
                alerts.append(notification)
        return tuple(alerts)

    def evaluate_badges(self, child_name: str) -> Sequence[str]:
        if not self._feature_flags.is_enabled("badges"):
            return tuple(self._earned_badges.get(child_name, set()))
        board = self._chores[child_name]
        earned = self._earned_badges[child_name]
        total_completions = board.total_completions()
        for badge, threshold in self._badge_thresholds.items():
            if total_completions >= threshold:
                earned.add(badge)
        for chore in board.chores():
            if chore.streak >= 7:
                earned.add("Streak Master")
        return tuple(sorted(earned))

    def leaderboard(self) -> Sequence[LeaderboardEntry]:
        if not self._feature_flags.is_enabled("leaderboard"):
            return tuple()
        entries = [
            LeaderboardEntry(name=child, score=board.leaderboard_score())
            for child, board in self._chores.items()
        ]
        return tuple(sorted(entries, key=lambda entry: entry.score, reverse=True))

    def goal_gallery(self, child_name: str) -> Sequence[dict]:
        account = self.get_account(child_name)
        return tuple(
            {
                "name": goal.name,
                "progress": float(goal.progress()),
                "image": goal.image_url,
            }
            for goal in account.goals
        )

    def record_nfc_tap(self, child_name: str, *, bonus: AmountLike = 0) -> NFCEvent:
        bonus_amount = to_decimal(bonus)
        event = NFCEvent(child_name=child_name, bonus_awarded=bonus_amount)
        self._nfc_events.append(event)
        if bonus_amount > Decimal("0"):
            self.deposit(
                child_name,
                bonus_amount,
                "NFC tap bonus",
                category=EventCategory.BONUS,
                metadata={"source": "nfc"},
            )
        return event

    def nfc_log(self, child_name: Optional[str] = None) -> Sequence[NFCEvent]:
        if child_name is None:
            return tuple(self._nfc_events)
        return tuple(event for event in self._nfc_events if event.child_name == child_name)

    # ------------------------------------------------------------------
    # Notifications, contacts and digests
    # ------------------------------------------------------------------
    @property
    def notifications(self) -> NotificationCenter:
        return self._notifications

    def register_contacts(
        self,
        child_name: str,
        *,
        parent_email: str | None = None,
        parent_sms: str | None = None,
        kid_email: str | None = None,
        kid_sms: str | None = None,
    ) -> None:
        if parent_email:
            self._parent_contacts[child_name]["email"] = parent_email
        if parent_sms:
            self._parent_contacts[child_name]["sms"] = parent_sms
        if kid_email:
            self._kid_contacts[child_name]["email"] = kid_email
        if kid_sms:
            self._kid_contacts[child_name]["sms"] = kid_sms

    def schedule_weekly_digest(
        self,
        recipient: str,
        *,
        summary: Mapping[str, object],
        send_on: datetime,
    ) -> None:
        self._notifications.schedule_digest(
            ScheduledDigest(recipient=recipient, send_on=send_on, summary=dict(summary))
        )

    # ------------------------------------------------------------------
    # Payout approvals and exports
    # ------------------------------------------------------------------
    def set_soft_limit(self, child_name: str, amount: AmountLike) -> None:
        self._soft_limits[child_name] = to_decimal(amount)

    def request_payout(
        self,
        child_name: str,
        amount: AmountLike,
        *,
        description: str,
        requested_by: str,
    ) -> PayoutRequest:
        value = to_decimal(amount)
        limit = self._soft_limits.get(child_name, self._default_soft_limit)
        if value <= limit:
            transaction = self.withdraw(
                child_name,
                value,
                description,
                category=EventCategory.MANUAL,
                metadata={"approved": "auto"},
            )
            self._logger.log("payout_auto", child=child_name, amount=float(transaction.amount))
            return PayoutRequest(
                request_id="auto",
                child_name=child_name,
                amount=value,
                description=description,
                created_by=requested_by,
                status=PayoutStatus.APPROVED,
                resolved_at=datetime.utcnow(),
                resolved_by="system",
            )
        request = PayoutRequest(
            request_id=str(uuid4()),
            child_name=child_name,
            amount=value,
            description=description,
            created_by=requested_by,
        )
        self._pending_payouts[request.request_id] = request
        self._audit_log.record(requested_by, "request_payout", f"{child_name}:{value}")
        self._logger.log("payout_requested", child=child_name, amount=float(value))
        return request

    def approve_payout(self, request_id: str, *, approver: str) -> PayoutRequest:
        request = self._pending_payouts[request_id]
        request.approve(approver)
        self.withdraw(
            request.child_name,
            request.amount,
            request.description,
            category=EventCategory.MANUAL,
            metadata={"payout_request": request_id},
        )
        self._audit_log.record(approver, "approve_payout", request.child_name)
        self._pending_payouts.pop(request_id, None)
        return request

    def reject_payout(self, request_id: str, *, approver: str) -> PayoutRequest:
        request = self._pending_payouts[request_id]
        request.reject(approver)
        self._audit_log.record(approver, "reject_payout", request.child_name)
        self._pending_payouts.pop(request_id, None)
        return request

    def pending_payouts(self) -> Sequence[PayoutRequest]:
        return tuple(self._pending_payouts.values())

    def bulk_approve_payouts(self, request_ids: Sequence[str], *, approver: str) -> Sequence[str]:
        approved: list[str] = []
        for request_id in request_ids:
            if request_id in self._pending_payouts:
                self.approve_payout(request_id, approver=approver)
                approved.append(request_id)
        return tuple(approved)

    def export_transactions_csv(
        self,
        child_name: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        category: EventCategory | None = None,
    ) -> str:
        return self.get_account(child_name).export_transactions_csv(start=start, end=end, category=category)

    def search_transactions(
        self,
        *,
        child: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        categories: Sequence[EventCategory] | None = None,
        types: Sequence[TransactionType] | None = None,
    ) -> Sequence[tuple[str, Transaction]]:
        names = [child] if child else self.list_accounts()
        results: list[tuple[str, Transaction]] = []
        for name in names:
            account = self.get_account(name)
            for transaction in account.filter_transactions(start=start, end=end, categories=categories, types=types):
                results.append((name, transaction))
        return tuple(results)

    def bulk_deactivate_chores(self, child_name: str, names: Sequence[str]) -> Sequence[str]:
        board = self._chores[child_name]
        removed: list[str] = []
        for chore_name in names:
            if chore_name in {chore.name for chore in board.chores()}:
                board.remove_chore(chore_name)
                removed.append(chore_name)
        if removed:
            self._audit_log.record("guardian", "remove_chore", f"{child_name}:{','.join(removed)}")
        return tuple(removed)

    # ------------------------------------------------------------------
    # Dashboard summaries and undo support
    # ------------------------------------------------------------------
    def kid_summary(self, child_name: str) -> KidSummary:
        board = self._chores[child_name]
        account = self.get_account(child_name)
        last_payout = account.last_transaction(transaction_type=TransactionType.DEPOSIT)
        pending = len(board.pending())
        return KidSummary(
            completion_percentage=board.completion_rate(),
            last_payout=last_payout.timestamp if last_payout else None,
            next_allowance_eta=self._next_allowance_eta.get(child_name),
            pending_chores=pending,
        )

    def set_next_allowance_eta(self, child_name: str, eta: Optional[datetime]) -> None:
        self._next_allowance_eta[child_name] = eta

    def undo_last_event(self) -> str:
        return self._undo.undo()

    # ------------------------------------------------------------------
    # Investing helpers
    # ------------------------------------------------------------------
    def portfolio(self, child_name: str) -> InvestmentPortfolio:
        return self._portfolios[child_name]

    def certificates(self, child_name: str) -> tuple[CertificateOfDeposit, ...]:
        return self._portfolios[child_name].certificates()

    def set_certificate_rate(
        self,
        child_name: str,
        rate: float,
        *,
        term_months: int | None = None,
        update_existing: bool = False,
    ) -> Decimal:
        return self._portfolios[child_name].set_cd_rate(
            rate, term_months=term_months, update_existing=update_existing
        )

    def set_certificate_penalty(self, child_name: str, term_months: int, days: int) -> int:
        return self._portfolios[child_name].set_cd_penalty(term_months, days)

    def open_certificate(
        self,
        child_name: str,
        amount: AmountLike,
        *,
        term_months: int = 12,
        description: str | None = None,
        opened_on: datetime | None = None,
    ) -> CertificateOfDeposit:
        account = self.get_account(child_name)
        portfolio = self._portfolios[child_name]
        value = to_decimal(amount)
        if portfolio.available_cash() < value:
            raise InsufficientFundsError(
                f"Portfolio for '{child_name}' has insufficient cash for certificate purchase."
            )
        metadata = {
            "investment": "certificate_of_deposit",
            "term_months": str(term_months),
        }
        transaction = account.withdraw(
            value,
            description or "Certificate of deposit purchase",
            category=EventCategory.INVEST,
            metadata=metadata,
        )
        self._after_transaction(child_name, transaction)
        certificate = portfolio.open_certificate(
            transaction.amount,
            term_months=term_months,
            opened_on=opened_on,
        )
        return certificate

    def withdraw_certificate(
        self,
        child_name: str,
        certificate: CertificateOfDeposit,
        *,
        at: datetime | None = None,
        note: str | None = None,
    ) -> tuple[Decimal, Decimal, Decimal]:
        portfolio = self._portfolios[child_name]
        gross, penalty, net = portfolio.close_certificate(certificate, at=at)
        account = self.get_account(child_name)
        metadata = {"investment": "certificate_of_deposit"}
        metadata["status"] = "early" if penalty > Decimal("0") else "matured"
        description = note or "Certificate of deposit withdrawal"
        deposit_tx = account.deposit(
            gross,
            description,
            category=EventCategory.INVEST,
            metadata=metadata,
        )
        self._after_transaction(child_name, deposit_tx)
        if penalty > Decimal("0"):
            penalty_tx = account.withdraw(
                penalty,
                "Certificate early withdrawal penalty",
                category=EventCategory.PENALTY,
                metadata={"investment": "certificate_of_deposit"},
            )
            self._after_transaction(child_name, penalty_tx)
        self._audit_log.record("system", "withdraw_certificate", f"{child_name}:{gross}")
        return gross, penalty, net

    def mature_certificates(self, child_name: str, *, at: datetime | None = None) -> Decimal:
        portfolio = self._portfolios[child_name]
        payout = portfolio.mature_certificates(at=at)
        if payout > Decimal("0"):
            transaction = self.get_account(child_name).deposit(
                payout,
                "Certificate of deposit matured",
                category=EventCategory.INVEST,
                metadata={"investment": "certificate_of_deposit"},
            )
            self._after_transaction(child_name, transaction)
        return payout

    def run_investing_jobs(self, *, at: Optional[datetime] = None) -> Dict[str, Decimal]:
        invested: Dict[str, Decimal] = {}
        for child, portfolio in self._portfolios.items():
            amount = portfolio.run_dca_if_due(at=at)
            if amount > Decimal("0"):
                invested[child] = amount
                self._logger.log("dca_executed", child=child, amount=float(amount))
        return invested

    # ------------------------------------------------------------------
    # Ops & integrations
    # ------------------------------------------------------------------
    @property
    def auth(self) -> AuthManager:
        return self._auth

    @property
    def feature_flags(self) -> FeatureFlagRegistry:
        return self._feature_flags

    @property
    def audit_log(self) -> AuditLog:
        return self._audit_log

    @property
    def backup_manager(self) -> BackupManager:
        return self._backups

    @property
    def health_monitor(self) -> HealthMonitor:
        return self._health

    @property
    def logger(self) -> StructuredLogger:
        return self._logger

    @property
    def translator(self) -> Translator:
        return self._translator

    def create_backup(self, *, label: str = "") -> BackupMetadata:
        payload = {
            "accounts": {
                name: {
                    "balance": float(account.balance),
                    "escrow": float(account.escrow_balance),
                    "goals": [goal.name for goal in account.goals],
                }
                for name, account in self._accounts.items()
            }
        }
        return self._backups.create_backup(payload, label=label)

    def health_status(self) -> dict:
        return self._health.status()

    def register_webhook(self, listener: Callable[[Dict[str, object]], None]) -> None:
        self._webhooks.register(listener)

    def api_account(self, child_name: str) -> dict:
        return self._api.account_snapshot(self.get_account(child_name))

    def translate(self, key: str, *, locale: Optional[str] = None) -> str:
        return self._translator.translate(key, locale=locale)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _after_transaction(self, child_name: str, transaction: Transaction) -> None:
        self._logger.log(
            "transaction",
            child=child_name,
            type=transaction.type.value,
            category=transaction.category.value if transaction.category else None,
            amount=float(transaction.amount),
        )
        self._webhooks.dispatch(
            {
                "event": "transaction",
                "child": child_name,
                "type": transaction.type.value,
                "amount": float(transaction.amount),
                "category": transaction.category.value if transaction.category else None,
                "timestamp": transaction.timestamp.isoformat(),
            }
        )
        self._register_undo(child_name, transaction)

    def _register_undo(self, child_name: str, transaction: Transaction) -> None:
        account = self.get_account(child_name)
        if transaction.type is TransactionType.DEPOSIT:
            self._undo.register(
                lambda: account.withdraw(
                    transaction.amount,
                    description=f"Undo {transaction.description}",
                    category=transaction.category,
                )
            )
        elif transaction.type is TransactionType.WITHDRAWAL:
            self._undo.register(
                lambda: account.deposit(
                    transaction.amount,
                    description=f"Undo {transaction.description}",
                    category=transaction.category,
                )
            )

    def _notify_payout(self, child_name: str, transaction: Transaction) -> None:
        contact = self._kid_contacts.get(child_name, {})
        recipient = contact.get("email") or contact.get("sms")
        if not recipient:
            return
        channel = NotificationChannel.EMAIL if contact.get("email") else NotificationChannel.SMS
        notification = Notification(
            recipient=recipient,
            channel=channel,
            type=NotificationType.PAYOUT_POSTED,
            subject="Allowance arrived!",
            body=f"{format_currency(transaction.amount)} was added to your account.",
            metadata={"child": child_name},
        )
        self._notifications.queue(notification)

    def _check_goal_milestones(self, child_name: str, goal: Goal) -> None:
        key = (child_name, goal.name)
        milestones = self._goal_milestones[key]
        for percentage in (25, 50, 75, 100):
            if percentage in milestones:
                continue
            if goal.milestone_reached(percentage):
                milestones.add(percentage)
                contact = self._kid_contacts.get(child_name, {})
                recipient = contact.get("email") or contact.get("sms")
                if recipient:
                    channel = NotificationChannel.EMAIL if contact.get("email") else NotificationChannel.SMS
                    notification = Notification(
                        recipient=recipient,
                        channel=channel,
                        type=NotificationType.GOAL_MILESTONE,
                        subject=f"{percentage}% milestone reached!",
                        body=f"Goal '{goal.name}' is now {percentage}% funded.",
                        metadata={"child": child_name, "goal": goal.name, "milestone": str(percentage)},
                    )
                    self._notifications.queue(notification)


__all__ = ["KidBank"]
