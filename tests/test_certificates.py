import math
import sys
import types
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda: None  # type: ignore[attr-defined]
sys.modules.setdefault("dotenv", dotenv_stub)

from kidbank.webapp import (  # noqa: E402
    Certificate,
    certificate_penalty_cents,
    certificate_sale_breakdown_cents,
    certificate_term_days,
    certificate_maturity_date,
    certificate_value_cents,
    certificate_maturity_value_cents,
    certificate_progress_percent,
)


def test_certificate_penalty_caps_at_accrued_interest() -> None:
    opened_at = datetime.utcnow() - timedelta(days=60)
    certificate = Certificate(
        id=1,
        kid_id="ava",
        principal_cents=10_000,
        rate_bps=300,
        term_months=6,
        opened_at=opened_at,
        penalty_days=90,
    )
    moment = opened_at + timedelta(days=60)

    penalty = certificate_penalty_cents(certificate, at=moment)
    gross, _, net = certificate_sale_breakdown_cents(certificate, at=moment)
    accrued = gross - certificate.principal_cents

    assert penalty == accrued
    assert net == certificate.principal_cents


def test_certificate_penalty_ignored_after_maturity() -> None:
    opened_at = datetime.utcnow() - timedelta(days=200)
    certificate = Certificate(
        id=2,
        kid_id="ben",
        principal_cents=5_000,
        rate_bps=250,
        term_months=6,
        opened_at=opened_at,
        penalty_days=60,
    )
    moment = datetime.utcnow()

    assert certificate_penalty_cents(certificate, at=moment) == 0
    certificate.matured_at = moment
    assert certificate_penalty_cents(certificate, at=moment) == 0


def test_certificate_one_week_term_uses_days() -> None:
    opened_at = datetime.utcnow() - timedelta(days=5)
    certificate = Certificate(
        id=3,
        kid_id="cody",
        principal_cents=10_000,
        rate_bps=200,
        term_months=0,
        term_days=7,
        opened_at=opened_at,
        penalty_days=3,
    )

    assert certificate_term_days(certificate) == 7
    assert certificate_maturity_date(certificate) == opened_at + timedelta(days=7)
    mid_value = certificate_value_cents(certificate, at=opened_at + timedelta(days=3))
    mature_value = certificate_maturity_value_cents(certificate)

    assert certificate.principal_cents < mid_value < mature_value


def test_certificate_value_uses_compound_interest() -> None:
    opened_at = datetime.utcnow() - timedelta(days=15)
    certificate = Certificate(
        id=4,
        kid_id="dana",
        principal_cents=10_000,
        rate_bps=600,
        term_months=0,
        term_days=30,
        opened_at=opened_at,
    )
    moment = opened_at + timedelta(days=15)
    value_cents = certificate_value_cents(certificate, at=moment)
    principal = Decimal(certificate.principal_cents)
    rate = Decimal(certificate.rate_bps) / Decimal(10000)
    factor = Decimal(str(math.exp(0.5 * math.log1p(float(rate)))))
    expected = (principal * factor).quantize(Decimal("1"), rounding=ROUND_HALF_UP)

    assert value_cents == int(expected)


def test_certificate_progress_uses_fractional_days() -> None:
    opened_at = datetime.utcnow() - timedelta(hours=12)
    certificate = Certificate(
        id=5,
        kid_id="elsa",
        principal_cents=5_000,
        rate_bps=400,
        term_months=0,
        term_days=1,
        opened_at=opened_at,
    )

    pct = certificate_progress_percent(certificate, at=datetime.utcnow())

    assert 40.0 < pct < 60.0
