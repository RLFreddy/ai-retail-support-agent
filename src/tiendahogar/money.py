"""Money amounts in free text, parsed exactly with ``Decimal``.

Recognized forms: ``$1,200.50``, ``$ 80``, ``USD 1500``, ``1200 dollars``,
``1,500 USD``, ``1200 dólares``. Bare numbers (order ids, days, months) are never amounts.
"""

from __future__ import annotations

import re
from decimal import Decimal

#: Refunds above this amount need a human supervisor's approval.
REFUND_LIMIT = Decimal(500)

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_AMOUNT_RE = re.compile(
    rf"(?:\$|\busd\b)\s*(?P<prefixed>{_NUMBER})"
    rf"|(?P<suffixed>{_NUMBER})\s*(?:usd|dollars?|bucks|d[oó]lares)\b",
    re.IGNORECASE,
)


def amounts(text: str) -> list[Decimal]:
    """Every money amount in ``text``, in order of appearance."""
    return [
        Decimal((match["prefixed"] or match["suffixed"]).replace(",", ""))
        for match in _AMOUNT_RE.finditer(text)
    ]


def largest_amount(text: str) -> Decimal | None:
    found = amounts(text)
    return max(found) if found else None
