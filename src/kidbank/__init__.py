"""KidBank package for managing reward-based banking for children."""

from .account import Account
from .exceptions import (
    AccountNotFoundError,
    DuplicateAccountError,
    GoalNotFoundError,
    InsufficientFundsError,
    KidBankError,
)
from .models import Goal, Reward, Transaction, TransactionType
from .service import KidBank

__all__ = [
    "Account",
    "KidBank",
    "Goal",
    "Reward",
    "Transaction",
    "TransactionType",
    "KidBankError",
    "AccountNotFoundError",
    "DuplicateAccountError",
    "GoalNotFoundError",
    "InsufficientFundsError",
]
