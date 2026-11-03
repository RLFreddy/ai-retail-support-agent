"""Tests of the money amount parser."""

from decimal import Decimal

import pytest

from tiendahogar.money import amounts, largest_amount


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a refund of $1200", ["1200"]),
        ("a refund of $1,500.50", ["1500.50"]),
        ("$ 80 for the toaster", ["80"]),
        ("USD 1500", ["1500"]),
        ("1200 dollars", ["1200"]),
        ("1,500 USD", ["1500"]),
        ("$1200 for the fridge and $30 of delivery", ["1200", "30"]),
    ],
)
def test_amounts(text, expected):
    assert amounts(text) == [Decimal(value) for value in expected]


@pytest.mark.parametrize(
    "text", ["order ORD-1001", "within 30 days", "a warranty of 12 months", "refund please"]
)
def test_numbers_without_currency_are_not_amounts(text):
    assert amounts(text) == []


def test_largest_amount():
    assert largest_amount("$1200 for the fridge, $30 of delivery") == Decimal(1200)
    assert largest_amount("no money here") is None


def test_amounts_are_exact():
    assert amounts("$0.10")[0] + amounts("$0.20")[0] == Decimal("0.30")
