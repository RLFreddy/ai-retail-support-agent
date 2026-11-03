"""Guardrails del agente TiendaHogar.

Reglas puras (sin red, deterministas) aplicadas en dos momentos:
- **Entrada** (``check_input``): decide si el mensaje debe derivarse a soporte
  humano, aprobarse como simple, o bloquearse (inyección de prompt / PII).
- **Salida** (``check_output``): valida que la respuesta final cumpla las
  políticas (no inventar, no escalar montos, derivar temas humanos) y marca
  una acción de derivación si corresponde.

Diseño: cada regla es una función pura ``(mensaje) -> motivo | None``. Las
reglas de salida miran el texto final que el agente va a entregar.

Políticas (del enunciado):
- Reembolso > $500 -> derivar a supervisor humano.
- Queja de trato de empleado, disputa de facturación, tema legal -> referir a
  soporte@tiendahogar.example (un humano).
- El agente no debe inventar información: si no hay grounding, no responder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tiendahogar.knowledge import DOCUMENTS

SUPPORT_EMAIL = "soporte@tiendahogar.example"

#: Umbral de reembolso que requiere aprobación humana (policy).
REFUND_THRESHOLD = 500.0

_AMOUNT_RE = re.compile(r"\$\s*([\d.,]+)")

#: Menciones del agente como humano / trato de empleado.
_EMPLOYEE_TRATO_RE = re.compile(
    r"\b(trato|actitud|groser[oa]|insult[oa]|amable|amabilidad)\b",
    re.IGNORECASE,
)
_PERSONA_RE = re.compile(
    r"\b(agente|empleado|ejecutivo|asesor|personal|representante)\b",
    re.IGNORECASE,
)

_BILLING_RE = re.compile(
    r"\b(disputa|cargos|facturaci[oó]n|factura|recibo|estado de cuenta|pago duplicado)\b",
    re.IGNORECASE,
)

_LEGAL_RE = re.compile(
    r"\b(legal|abogado|demanda|querella|juzgado|corte|consumidor y sociedad)\b",
    re.IGNORECASE,
)

#: Pares "(cantidad, unidad de tiempo)" para grounding de la salida.
_CLAIM_RE = re.compile(
    r"(\d+\s*(?:d[ií]as?\s+h[aá]biles?|meses|semanas|a[uñ]os?))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardrailVerdict:
    """Resultado del guardrail."""

    action: str  # "proceed" | "escalate" | "refer" | "block" | "off_topic"
    reason: str = ""
    #: Monto de reembolso detectado (para "escalate").
    amount: float | None = None


def _parse_amount(text: str) -> tuple[bool, float]:
    """Devuelve (encontrado?, monto). Usa el ÚLTIMO monto si hay varios."""
    found = False
    amount = 0.0
    for match in _AMOUNT_RE.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            amount = float(raw)
            found = True
        except ValueError:
            continue
    return found, amount


# --------------------------------------------------------------------------- #
# Reglas de entrada
# --------------------------------------------------------------------------- #
def _rule_refund_escalation(message: str) -> tuple[str | None, float | None]:
    """Reembolso > $500 -> escalar a supervisor humano (regla 5).

    Devuelve (motivo, monto) para evitar re-parsear el mensaje en ``check_input``.
    """
    found, amount = _parse_amount(message)
    if found and amount > REFUND_THRESHOLD:
        return (
            f"reembolso mayor a ${REFUND_THRESHOLD:,.0f} detectado (${amount:,.0f})",
            amount,
        )
    return None, None


def _rule_human_topics(message: str) -> str | None:
    """Trato de empleado / facturación / legal -> referir a soporte (regla 6)."""
    trato = bool(_EMPLOYEE_TRATO_RE.search(message) and _PERSONA_RE.search(message))
    if trato:
        return "queja sobre trato de un empleado"
    if _BILLING_RE.search(message):
        return "disputa de facturación"
    if _LEGAL_RE.search(message):
        return "tema legal"
    return None


def _rule_injection_or_pii(message: str) -> str | None:
    """Bloquear escaladas de privilegio / intentos de manipulación / datos sensibles."""
    lowered = message.lower()
    if re.search(r"\b(system prompt|ignore (previous|all)|reveal instructions)\b", lowered):
        return "posible inyección de prompt (escalada de sistema)"
    if re.search(r"\b(creds|password|clave|cvv|(tarjeta|card)\s*(est[aá]|es\s+|\s+)?\d{4}|\d{15,16})\b", lowered):
        return "datos sensibles (credenciales/tarjeta) en el mensaje"
    return None


def _rule_off_topic(message: str) -> str | None:
    """Preguntas fuera del dominio comercial de TiendaHogar."""
    if re.search(r"\b(filtrar|decrypt|accesar|pentest|hackeo)\b", message, re.IGNORECASE):
        return "solicitud fuera del dominio de soporte"
    return None


def check_input(message: str) -> GuardrailVerdict:
    """Evalúa el mensaje de entrada y decide el flujo."""
    if not message or not message.strip():
        return GuardrailVerdict(action="proceed")

    for label, rule in (
        ("block", _rule_injection_or_pii),
        ("block", _rule_off_topic),
    ):
        reason = rule(message)
        if reason:
            return GuardrailVerdict(action=label, reason=reason)

    reason = _rule_human_topics(message)
    if reason:
        return GuardrailVerdict(action="refer", reason=reason)

    reason, amount = _rule_refund_escalation(message)
    if reason:
        return GuardrailVerdict(action="escalate", reason=reason, amount=amount)

    return GuardrailVerdict(action="proceed")


# --------------------------------------------------------------------------- #
# Reglas de salida
# --------------------------------------------------------------------------- #
def _output_mentions_human_refund(answer: str) -> bool:
    """Detecta que la respuesta se autoprocura aprobar un reembolso > $500."""
    if not re.search(r"\baprueb[oa]|aprobar", answer.lower()):
        return False
    found, amount = _parse_amount(answer)
    mentions_refund = (
        re.search(r"\b(reembolso|devoluci[oó]n)\b", answer, re.IGNORECASE) is not None
    )
    return found and amount > REFUND_THRESHOLD and mentions_refund


def extract_known_claims(text: str) -> set[str]:
    """Extrae pares "(cantidad, unidad)" normalizados (minúsculas, un solo espacio)."""
    return {re.sub(r"\s+", " ", c).lower() for c in _CLAIM_RE.findall(text)}


def _output_claims_unknown_fact(answer: str, known_extra: set[str] | None = None) -> bool:
    """Detecta hechos no soportados por las fuentes autorizadas.

    Extrae pares "(cantidad, unidad)" nombrados en la respuesta y verifica que
    existan en el corpus (documentos de política) o en ``known_extra`` (datos
    transaccionales de la tool, p. ej. ``entrega_estimada`` de un pedido).
    Si la respuesta afirma un plazo/política que no aparece en ninguna fuente,
    se considera inventado.
    """
    if "soporte@tiendahogar.example" in answer.lower():
        return False
    claims = _CLAIM_RE.findall(answer)
    if not claims:
        return False
    corpus_text = " ".join(doc.content.lower() for doc in DOCUMENTS)
    known = extract_known_claims(corpus_text)
    if known_extra:
        known |= set(known_extra)
    return any(re.sub(r"\s+", " ", claim).lower() not in known for claim in claims)


def check_output(answer: str, known_extra: set[str] | None = None) -> GuardrailVerdict:
    """Valida la respuesta final antes de entregarla al usuario.

    ``known_extra``: claims (pares cantidad/unidad) provenientes de la tool
    ``consultar_estado_pedido`` — son fuente autorizada, igual que el corpus.
    """
    if not answer or not answer.strip():
        return GuardrailVerdict(
            action="block", reason="la respuesta generada está vacía"
        )

    if _output_mentions_human_refund(answer):
        return GuardrailVerdict(
            action="escalate",
            reason="la respuesta autoaprueba un reembolso mayor a $500",
        )

    if _output_claims_unknown_fact(answer, known_extra=known_extra):
        return GuardrailVerdict(
            action="refer",
            reason="la respuesta afirma plazos/políticas sin grounding verificado",
        )

    if re.search(r"\b(trato inadecuado|groser[oa])\b", answer, re.IGNORECASE):
        return GuardrailVerdict(
            action="refer",
            reason="tema de trato de empleado debe ir a soporte humano",
        )

    return GuardrailVerdict(action="proceed")