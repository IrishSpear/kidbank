"""Investing helpers for KidBank."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_UP
from typing import Dict, Optional


@dataclass(slots=True)
class DCAConfig:
    amount: Decimal
    interval: timedelta
    next_run: datetime
    active: bool = True

    def schedule_next(self) -> None:
        self.next_run += self.interval


class InvestmentPortfolio:
    """Track cash and simulated stock/bond balances for a child."""

    def __init__(self) -> None:
        self.positions: Dict[str, Decimal] = {
            "cash": Decimal("0.00"),
            "stock": Decimal("0.00"),
            "bond": Decimal("0.00"),
        }
        self.auto_sweep_enabled = True
        self.allocation: Dict[str, Decimal] = {"stock": Decimal("0.70"), "bond": Decimal("0.30")}
        self.dca: Optional[DCAConfig] = None
        self.explainers: Dict[str, str] = {
            "risk": "Investing involves ups and downs. Holding both stocks and bonds spreads the risk.",
            "p_l": "Profit and loss shows how much your investments changed from what you put in.",
            "diversification": "Putting money in different buckets keeps your eggs out of one basket.",
        }

    # Cash handling ---------------------------------------------------------
    def record_deposit(self, amount: Decimal) -> Decimal:
        amount = Decimal(amount).quantize(Decimal("0.01"))
        self.positions["cash"] += amount
        if self.auto_sweep_enabled:
            swept = self.auto_cash_sweep(amount)
        else:
            swept = Decimal("0.00")
        return swept

    def auto_cash_sweep(self, deposit_amount: Decimal) -> Decimal:
        value = Decimal(deposit_amount).quantize(Decimal("0.01"))
        ceiling = value.quantize(Decimal("1"), rounding=ROUND_UP)
        spare = (ceiling - value).quantize(Decimal("0.01"))
        if spare <= Decimal("0.00"):
            return Decimal("0.00")
        if self.positions["cash"] < spare:
            return Decimal("0.00")
        self.positions["cash"] -= spare
        self._invest(spare)
        return spare

    def _invest(self, amount: Decimal) -> None:
        total_weight = sum(self.allocation.values())
        if total_weight == 0:
            self.positions["stock"] += amount
            return
        allocated = Decimal("0.00")
        for asset, weight in self.allocation.items():
            portion = (amount * (weight / total_weight)).quantize(Decimal("0.01"))
            self.positions.setdefault(asset, Decimal("0.00"))
            self.positions[asset] += portion
            allocated += portion
        residual = amount - allocated
        if residual != Decimal("0.00"):
            self.positions["cash"] += residual

    # DCA -------------------------------------------------------------------
    def configure_dca(
        self,
        amount: Decimal,
        *,
        start: Optional[datetime] = None,
        interval_days: int = 7,
    ) -> DCAConfig:
        config = DCAConfig(
            amount=Decimal(amount).quantize(Decimal("0.01")),
            interval=timedelta(days=interval_days),
            next_run=start or datetime.utcnow(),
        )
        self.dca = config
        return config

    def toggle_dca(self, active: bool) -> None:
        if not self.dca:
            raise RuntimeError("No DCA configuration set.")
        self.dca.active = active

    def run_dca_if_due(self, *, at: Optional[datetime] = None) -> Decimal:
        if not self.dca or not self.dca.active:
            return Decimal("0.00")
        moment = at or datetime.utcnow()
        if moment < self.dca.next_run:
            return Decimal("0.00")
        amount = self.dca.amount
        if self.positions["cash"] < amount:
            return Decimal("0.00")
        self.positions["cash"] -= amount
        self._invest(amount)
        self.dca.schedule_next()
        return amount

    # Portfolio summaries ---------------------------------------------------
    def total_value(self) -> Decimal:
        return sum(self.positions.values())

    def allocation_breakdown(self) -> Dict[str, float]:
        total = float(self.total_value())
        if total == 0:
            return {asset: 0.0 for asset in self.positions}
        return {asset: float(balance / total) for asset, balance in self.positions.items()}

    def set_allocation(self, *, stock: float, bond: float) -> None:
        total = stock + bond
        if total <= 0:
            raise ValueError("Allocation weights must be positive.")
        self.allocation["stock"] = Decimal(stock / total).quantize(Decimal("0.01"))
        self.allocation["bond"] = Decimal(bond / total).quantize(Decimal("0.01"))

    def explain(self, topic: str) -> str:
        return self.explainers.get(topic, "Ask a grown-up for details about this topic!")


__all__ = ["DCAConfig", "InvestmentPortfolio"]
