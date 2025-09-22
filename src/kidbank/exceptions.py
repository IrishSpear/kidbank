"""Custom exception hierarchy for the KidBank package."""

from __future__ import annotations


class KidBankError(Exception):
    """Base class for all KidBank specific errors."""


class DuplicateAccountError(KidBankError):
    """Raised when attempting to create an account that already exists."""


class AccountNotFoundError(KidBankError):
    """Raised when an account lookup fails."""


class InsufficientFundsError(KidBankError):
    """Raised when an account operation would result in a negative balance."""


class GoalNotFoundError(KidBankError):
    """Raised when a requested savings goal cannot be found."""
