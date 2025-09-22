"""API helpers and webhook integrations for KidBank."""

from __future__ import annotations

import json
from typing import Callable, Dict, TYPE_CHECKING

from .models import Transaction

if TYPE_CHECKING:  # pragma: no cover
    from .account import Account


class ApiExporter:
    """Convert KidBank data structures to JSON friendly dictionaries."""

    def account_snapshot(self, account: "Account") -> Dict[str, object]:
        return {
            "child": account.child_name,
            "balance": float(account.balance),
            "escrow_balance": float(account.escrow_balance),
            "transactions": [self._serialise_transaction(tx) for tx in account.transactions],
            "goals": [
                {
                    "name": goal.name,
                    "target": float(goal.target_amount),
                    "saved": float(goal.saved_amount),
                    "progress": float(goal.progress()),
                    "image": goal.image_url,
                }
                for goal in account.goals
            ],
            "rewards": [
                {
                    "name": reward.name,
                    "cost": float(reward.cost),
                    "redeemed_at": reward.redeemed_at.isoformat(),
                }
                for reward in account.rewards
            ],
        }

    def to_json(self, payload: Dict[str, object]) -> str:
        return json.dumps(payload, sort_keys=True)

    def _serialise_transaction(self, transaction: Transaction) -> Dict[str, object]:
        return {
            "timestamp": transaction.timestamp.isoformat(),
            "type": transaction.type.value,
            "category": transaction.category.value if transaction.category else None,
            "amount": float(transaction.amount),
            "description": transaction.description,
        }


class WebhookDispatcher:
    """Simple synchronous webhook broadcaster."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[Dict[str, object]], None]] = []

    def register(self, listener: Callable[[Dict[str, object]], None]) -> None:
        self._listeners.append(listener)

    def unregister(self, listener: Callable[[Dict[str, object]], None]) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def dispatch(self, event: Dict[str, object]) -> None:
        for listener in list(self._listeners):
            listener(event)


__all__ = ["ApiExporter", "WebhookDispatcher"]
