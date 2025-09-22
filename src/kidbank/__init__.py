"""KidBank package for managing reward-based banking for children."""

from .account import Account
from .admin import AuditLog, FeatureFlagRegistry, UndoManager
from .api import ApiExporter, WebhookDispatcher
from .chores import Chore, ChoreBoard, ChorePack, ChoreSchedule, TimeWindow, Weekday
from .emailing import EmailClient
from .exceptions import (
    AccountNotFoundError,
    DuplicateAccountError,
    GoalNotFoundError,
    InsufficientFundsError,
    KidBankError,
)
from .i18n import Translator
from .investing import InvestmentPortfolio
from .models import (
    BackupMetadata,
    EventCategory,
    FeatureFlag,
    Goal,
    KidSummary,
    LeaderboardEntry,
    NFCEvent,
    PayoutRequest,
    PayoutStatus,
    Reward,
    ScheduledDigest,
    Transaction,
    TransactionType,
)
from .notifications import Notification, NotificationCenter, NotificationChannel, NotificationType
from .ops import BackupManager, HealthMonitor, StructuredLogger
from .security import AuthManager
from .service import KidBank

__all__ = [
    "Account",
    "AuditLog",
    "AuthManager",
    "ApiExporter",
    "BackupManager",
    "BackupMetadata",
    "Chore",
    "ChoreBoard",
    "ChorePack",
    "ChoreSchedule",
    "EmailClient",
    "EventCategory",
    "FeatureFlag",
    "FeatureFlagRegistry",
    "KidBank",
    "KidSummary",
    "Goal",
    "LeaderboardEntry",
    "NFCEvent",
    "Notification",
    "NotificationCenter",
    "NotificationChannel",
    "NotificationType",
    "InvestmentPortfolio",
    "ScheduledDigest",
    "Reward",
    "Transaction",
    "TransactionType",
    "KidBankError",
    "AccountNotFoundError",
    "DuplicateAccountError",
    "GoalNotFoundError",
    "InsufficientFundsError",
    "PayoutRequest",
    "PayoutStatus",
    "StructuredLogger",
    "HealthMonitor",
    "TimeWindow",
    "Translator",
    "UndoManager",
    "Weekday",
    "WebhookDispatcher",
]
