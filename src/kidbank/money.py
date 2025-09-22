"""Utilities for working with monetary values in KidBank."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Union

CENT = Decimal("0.01")

AmountLike = Union[Decimal, int, float, str]


def to_decimal(value: AmountLike) -> Decimal:
    """Convert ``value`` to a :class:`~decimal.Decimal` with two decimal places."""

    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, float)):
        result = Decimal(str(value))
    elif isinstance(value, str):
        result = Decimal(value)
    else:  # pragma: no cover - defensive programming branch
        raise TypeError(f"Unsupported amount type: {type(value)!r}")

    return result.quantize(CENT, rounding=ROUND_HALF_UP)


def require_positive(amount: Decimal, *, allow_zero: bool = False) -> Decimal:
    """Ensure ``amount`` is positive (or non-negative when ``allow_zero`` is true)."""

    if allow_zero:
        if amount < Decimal("0"):
            raise ValueError("Amount must be zero or greater.")
    else:
        if amount <= Decimal("0"):
            raise ValueError("Amount must be greater than zero.")
    return amount


def format_currency(amount: Decimal) -> str:
    """Return ``amount`` as a currency formatted string (e.g. ``$12.34``)."""

    return f"${amount.quantize(CENT, rounding=ROUND_HALF_UP):,.2f}"
