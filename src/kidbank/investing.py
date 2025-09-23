"""Investing helpers for KidBank."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_UP
from typing import Dict, Optional, Tuple

from .money import require_positive


def _to_decimal(value: Decimal | float | int) -> Decimal:
    return Decimal(value).quantize(Decimal("0.01"))


@dataclass(slots=True)
class CertificateOfDeposit:
    """Represents a time-bound savings product with a fixed rate."""

    principal: Decimal
    rate: Decimal
    term_months: int
    opened_on: datetime = field(default_factory=datetime.utcnow)
    penalty_days: int = 0

    def __post_init__(self) -> None:
        principal = _to_decimal(self.principal)
        require_positive(principal)
        if self.term_months <= 0:
            raise ValueError("term_months must be positive")
        rate = Decimal(self.rate).quantize(Decimal("0.0001"))
        object.__setattr__(self, "principal", principal)
        object.__setattr__(self, "rate", rate)
        if self.penalty_days < 0:
            raise ValueError("penalty_days cannot be negative")

    @property
    def term(self) -> timedelta:
        return timedelta(days=self.term_months * 30)

    @property
    def matures_on(self) -> datetime:
        return self.opened_on + self.term

    def value(self, *, at: Optional[datetime] = None) -> Decimal:
        """Return the accrued value as of ``at`` using simple interest."""

        moment = at or datetime.utcnow()
        total_days = self.term.days
        if total_days == 0:
            return self.payout_amount()
        elapsed_days = max(0, min((moment - self.opened_on).days, total_days))
        progress = Decimal(elapsed_days) / Decimal(total_days)
        interest = (self.principal * self.rate * progress).quantize(Decimal("0.01"))
        return self.principal + interest

    def payout_amount(self) -> Decimal:
        """Return the amount released upon maturity."""

        interest = (self.principal * self.rate).quantize(Decimal("0.01"))
        return self.principal + interest


@dataclass(slots=True)
class DCAConfig:
    amount: Decimal
    interval: timedelta
    next_run: datetime
    active: bool = True

    def schedule_next(self) -> None:
        self.next_run += self.interval


class InvestmentPortfolio:
    """Track cash, simulated stock/bond balances, and time deposits for a child."""

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
        self._default_cd_rate = Decimal("0.02")
        self._cd_rates: Dict[int, Decimal] = {}
        self._cd_penalties: Dict[int, int] = {}
        self._certificates: list[CertificateOfDeposit] = []

    # Cash handling ---------------------------------------------------------
    def record_deposit(self, amount: Decimal) -> Decimal:
        amount = _to_decimal(amount)
        self.positions["cash"] += amount
        if self.auto_sweep_enabled:
            swept = self.auto_cash_sweep(amount)
        else:
            swept = Decimal("0.00")
        return swept

    def available_cash(self) -> Decimal:
        return self.positions["cash"]

    def auto_cash_sweep(self, deposit_amount: Decimal) -> Decimal:
        value = _to_decimal(deposit_amount)
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

    # Certificates of deposit ------------------------------------------------
    def set_cd_rate(
        self,
        rate: float,
        *,
        term_months: int | None = None,
        update_existing: bool = False,
    ) -> Decimal:
        """Update the annual rate used for certificates of deposit."""

        new_rate = Decimal(rate).quantize(Decimal("0.0001"))
        if new_rate < Decimal("0"):
            raise ValueError("rate cannot be negative")
        if term_months is None:
            self._default_cd_rate = new_rate
            if update_existing:
                for certificate in self._certificates:
                    object.__setattr__(certificate, "rate", new_rate)
        else:
            if term_months <= 0:
                raise ValueError("term_months must be positive")
            self._cd_rates[term_months] = new_rate
            if update_existing:
                for certificate in self._certificates:
                    if certificate.term_months == term_months:
                        object.__setattr__(certificate, "rate", new_rate)
        return new_rate

    def set_cd_penalty(self, term_months: int, days: int) -> int:
        if term_months <= 0:
            raise ValueError("term_months must be positive")
        if days < 0:
            raise ValueError("Penalty days cannot be negative.")
        self._cd_penalties[term_months] = days
        return days

    def _rate_for_term(self, term_months: int) -> Decimal:
        return self._cd_rates.get(term_months, self._default_cd_rate)

    def _penalty_for_term(self, term_months: int) -> int:
        return self._cd_penalties.get(term_months, 0)

    def open_certificate(
        self,
        amount: Decimal,
        *,
        term_months: int = 12,
        opened_on: Optional[datetime] = None,
    ) -> CertificateOfDeposit:
        value = _to_decimal(amount)
        require_positive(value)
        if self.positions["cash"] < value:
            raise ValueError("Insufficient cash to purchase certificate.")
        self.positions["cash"] -= value
        certificate = CertificateOfDeposit(
            principal=value,
            rate=self._rate_for_term(term_months),
            term_months=term_months,
            opened_on=opened_on or datetime.utcnow(),
            penalty_days=self._penalty_for_term(term_months),
        )
        self._certificates.append(certificate)
        return certificate

    def certificates(self) -> Tuple[CertificateOfDeposit, ...]:
        return tuple(self._certificates)

    def total_certificate_value(self, *, at: Optional[datetime] = None) -> Decimal:
        return sum((certificate.value(at=at) for certificate in self._certificates), Decimal("0.00"))

    def close_certificate(
        self, certificate: CertificateOfDeposit, *, at: Optional[datetime] = None
    ) -> Tuple[Decimal, Decimal, Decimal]:
        try:
            self._certificates.remove(certificate)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Certificate not found in portfolio.") from exc
        moment = at or datetime.utcnow()
        if moment >= certificate.matures_on:
            gross = certificate.payout_amount()
            penalty = Decimal("0.00")
        else:
            gross = certificate.value(at=moment)
            total_days = certificate.term.days or 1
            daily_interest = (certificate.principal * certificate.rate) / Decimal(total_days)
            penalty = (
                daily_interest * Decimal(min(certificate.penalty_days, total_days))
            ).quantize(Decimal("0.01"))
            accrued_interest = max(Decimal("0.00"), gross - certificate.principal)
            penalty = min(penalty, accrued_interest)
        net = (gross - penalty).quantize(Decimal("0.01"))
        self.positions["cash"] = (self.positions["cash"] + net).quantize(Decimal("0.01"))
        gross = gross.quantize(Decimal("0.01"))
        penalty = penalty.quantize(Decimal("0.01"))
        return gross, penalty, net

    def mature_certificates(self, *, at: Optional[datetime] = None) -> Decimal:
        """Return matured certificates to cash and remove them from the ladder."""

        matured_total = Decimal("0.00")
        moment = at or datetime.utcnow()
        certificates = list(self._certificates)
        for certificate in certificates:
            if moment >= certificate.matures_on:
                _, _, net = self.close_certificate(certificate, at=moment)
                matured_total += net
        return matured_total

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
    def total_value(self, *, at: Optional[datetime] = None) -> Decimal:
        return sum(self.positions.values()) + self.total_certificate_value(at=at)

    def allocation_breakdown(self, *, at: Optional[datetime] = None) -> Dict[str, float]:
        total = float(self.total_value(at=at))
        assets = dict(self.positions)
        assets["cd"] = self.total_certificate_value(at=at)
        if total == 0:
            return {asset: 0.0 for asset in assets}
        return {asset: float(balance / total) for asset, balance in assets.items()}

    def set_allocation(self, *, stock: float, bond: float) -> None:
        total = stock + bond
        if total <= 0:
            raise ValueError("Allocation weights must be positive.")
        self.allocation["stock"] = Decimal(stock / total).quantize(Decimal("0.01"))
        self.allocation["bond"] = Decimal(bond / total).quantize(Decimal("0.01"))

    def explain(self, topic: str) -> str:
        return self.explainers.get(topic, "Ask a grown-up for details about this topic!")


__all__ = ["CertificateOfDeposit", "DCAConfig", "InvestmentPortfolio"]
