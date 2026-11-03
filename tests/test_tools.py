"""Tests of the ``get_order_status`` tool: valid ids are answered, invalid ones
are reported as not found without invented data."""

import pytest

from tiendahogar.tools import ORDERS, get_order_status


def test_existing_order_returns_its_status():
    result = get_order_status("ORD-1001")
    assert result["found"] is True
    assert result["product"] == "Refrigerator"
    assert result["status"] == "In transit"
    assert result["estimated_delivery"] == "3 business days"


@pytest.mark.parametrize("order_id", list(ORDERS))
def test_every_mock_order_is_found(order_id):
    assert get_order_status(order_id)["found"] is True


def test_unknown_order_is_not_invented():
    result = get_order_status("ORD-9999")
    assert result == {"order_id": "ORD-9999", "status": "not found", "found": False}


@pytest.mark.parametrize("order_id", ["", "   ", None])
def test_empty_id_is_not_found(order_id):
    assert get_order_status(order_id)["found"] is False


def test_id_is_case_and_space_insensitive():
    result = get_order_status(" ord-1002 ")
    assert result["order_id"] == "ORD-1002"
    assert result["found"] is True
