"""High level service for coordinating multiple KidBank accounts."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, Tuple

from .account import Account
from .exceptions import AccountNotFoundError, DuplicateAccountError
from .money import AmountLike, format_currency
from .models import Goal, Reward, Transaction


class KidBank:
    """Manage multiple :class:`~kidbank.account.Account` instances."""

    __slots__ = ("_accounts",)

    def __init__(self) -> None:
        self._accounts: Dict[str, Account] = {}

    def create_account(self, child_name: str, *, starting_balance: AmountLike = 0) -> Account:
        """Create and register a new account."""

        if child_name in self._accounts:
            raise DuplicateAccountError(f"Account '{child_name}' already exists.")
        account = Account(child_name, starting_balance=starting_balance)
        self._accounts[child_name] = account
        return account

    def list_accounts(self) -> Tuple[str, ...]:
        """Return the names of all registered accounts."""

        return tuple(sorted(self._accounts))

    def has_account(self, child_name: str) -> bool:
        """Return ``True`` if an account exists for ``child_name``."""

        return child_name in self._accounts

    def get_account(self, child_name: str) -> Account:
        """Return the account for ``child_name`` or raise ``AccountNotFoundError``."""

        try:
            return self._accounts[child_name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise AccountNotFoundError(f"Account '{child_name}' does not exist.") from exc

    def deposit(self, child_name: str, amount: AmountLike, description: str = "Deposit") -> Transaction:
        """Record a deposit for the specified account."""

        return self.get_account(child_name).deposit(amount, description)

    def withdraw(self, child_name: str, amount: AmountLike, description: str = "Withdrawal") -> Transaction:
        """Record a withdrawal for the specified account."""

        return self.get_account(child_name).withdraw(amount, description)

    def transfer(
        self,
        sender: str,
        recipient: str,
        amount: AmountLike,
        description: str | None = None,
    ) -> tuple[Transaction, Transaction]:
        """Transfer funds between two accounts."""

        source = self.get_account(sender)
        destination = self.get_account(recipient)
        return source.transfer_to(destination, amount, description)

    def redeem_reward(
        self,
        child_name: str,
        *,
        name: str,
        cost: AmountLike,
        description: str = "",
    ) -> Reward:
        """Redeem a reward for the specified account."""

        return self.get_account(child_name).redeem_reward(name, cost, description)

    def add_goal(
        self,
        child_name: str,
        *,
        name: str,
        target_amount: AmountLike,
        description: str = "",
    ) -> Goal:
        """Add a new goal to an account."""

        return self.get_account(child_name).add_goal(name, target_amount, description)

    def contribute_to_goal(
        self,
        child_name: str,
        *,
        name: str,
        amount: AmountLike,
        description: str | None = None,
    ) -> Goal:
        """Contribute funds to a specific goal for ``child_name``."""

        return self.get_account(child_name).contribute_to_goal(name, amount, description=description)

    def total_balance(self) -> Decimal:
        """Return the aggregate balance across all accounts."""

        return sum((account.balance for account in self._accounts.values()), Decimal("0.00"))

    def summary(self) -> str:
        """Return a multi-line report of all accounts and balances."""

        if not self._accounts:
            return "No accounts have been created yet."
        lines = ["KidBank summary:"]
        for name in sorted(self._accounts):
            account = self._accounts[name]
            lines.append(f"- {account.child_name}: {format_currency(account.balance)}")
        lines.append(f"Total balance: {format_currency(self.total_balance())}")
        return "\n".join(lines)
