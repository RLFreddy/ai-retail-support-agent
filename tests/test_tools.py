"""Tests de la tool `consultar_estado_pedido`.

Cubre el requisito mínimo de la prueba técnica:
- responde correctamente para un order_id válido,
- maneja correctamente uno inválido, sin inventar datos.
"""

import inspect

from tiendahogar.tools import ORDERS, consultar_estado_pedido


def test_firma_exacta_requerida():
    # El enunciado pide esta firma exacta: consultar_estado_pedido(order_id: str) -> dict
    # (con `from __future__ import annotations`, las anotaciones quedan como str).
    sig = inspect.signature(consultar_estado_pedido)
    assert list(sig.parameters.keys()) == ["order_id"]
    assert sig.parameters["order_id"].annotation in (str, "str")
    assert sig.return_annotation in (dict, "dict")


def test_pedido_valido_devuelve_estado_correcto():
    result = consultar_estado_pedido("ORD-1001")
    assert result["encontrado"] is True
    assert result["producto"] == "Refrigeradora"
    assert result["estado"] == "En tránsito"
    assert result["entrega_estimada"] == "3 días hábiles"


def test_todos_los_pedidos_mock_son_validos():
    for order_id in ORDERS:
        assert consultar_estado_pedido(order_id)["encontrado"] is True


def test_pedido_inexistente_no_inventa():
    result = consultar_estado_pedido("ORD-9999")
    assert result["encontrado"] is False
    assert result["estado"] == "no encontrado"
    assert "producto" not in result
    assert "entrega_estimada" not in result


def test_pedido_invalido_formato_vacio():
    assert consultar_estado_pedido("")["encontrado"] is False
    assert consultar_estado_pedido("   ")["encontrado"] is False
    assert consultar_estado_pedido(None)["encontrado"] is False


def test_insensible_a_mayusculas_y_espacios():
    assert consultar_estado_pedido(" ord-1002 ")["order_id"] == "ORD-1002"
    assert consultar_estado_pedido("ord-1002")["encontrado"] is True