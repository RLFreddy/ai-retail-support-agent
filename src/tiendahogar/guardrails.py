"""Guardrails of the TiendaHogar agent.

Pure, deterministic rules applied at two points, in English and Spanish at
once (a message cannot dodge a rule by switching language):

- **Input** (``check_input``): refer the message to a human, escalate it, block
  it (prompt injection, sensitive data) or let it through.
- **Output** (``check_output``): check that the final answer follows the
  policies (no invented facts, no self-approved large refunds).

Policies from the technical test:

- Refunds over $500 are escalated to a human supervisor.
- Complaints about an employee, billing disputes and legal matters are referred
  to a human at support@tiendahogar.example.
- The agent must not invent information.
"""

from __future__ import annotations

import re
import unicodedata

from tiendahogar.domain import Action, Verdict
from tiendahogar.knowledge import KNOWLEDGE
from tiendahogar.money import REFUND_LIMIT, largest_amount

_REFUND_RE = re.compile(
    r"\b(refund(ed|s)?|reimburse(ment)?|money back"
    r"|reembols\w*|(devol(ver|verme|uci[oó]n del?)|devuelv\w*|devu[eé]lv\w*) (el |mi )?dinero)\b",
    re.IGNORECASE,
)

_TREATMENT_RE = re.compile(
    r"\b(rude|attitude|insult(ed|ing)?|disrespectful|treated|unfriendly"
    r"|groser[oa]s?|groser[ií]a|maltrat\w*|me trat(ó|o|aron) mal|falta de respeto"
    r"|irrespetuos[oa]|insult(ó|o|aron)|mala actitud)\b",
    re.IGNORECASE,
)
_STAFF_RE = re.compile(
    r"\b(agent|employee|staff|representative|advisor|cashier"
    r"|agente|emplead[oa]|vendedor(a)?|asesor(a)?|cajer[oa]|personal|trabajador(a)?)\b",
    re.IGNORECASE,
)
#: A contested charge. Asking for an invoice or a receipt is not a dispute.
_BILLING_RE = re.compile(
    r"\b(disputed?|overcharged|charged (me )?(twice|double|wrong(ly)?)|double charged?"
    r"|unauthori[sz]ed charge|billing (error|issue|problem|mistake)"
    r"|disputa\w*|cobro (indebido|doble|duplicado|no autorizado)"
    r"|me cobraron (dos veces|doble|de m[aá]s)|cargo (no autorizado|indebido|duplicado)"
    r"|error (de|en (la|mi)) facturaci[oó]n|error en (la|mi) factura)\b",
    re.IGNORECASE,
)
_LEGAL_RE = re.compile(
    r"\b(legal|lawyer|attorney|lawsuit|sue|court"
    r"|abogad[oa]|demandar(é|e|los)?|denunciar(é|e|los)?|juicio|tribunal|juzgado)\b",
    re.IGNORECASE,
)

_INJECTION_RE = re.compile(
    r"\b(ignore|disregard|forget|override|bypass)\b[^.?!]{0,30}"
    r"\b(instructions?|rules|prompt|guidelines)\b"
    r"|\b(reveal|show|print|display|repeat|output|tell me)\b[^.?!]{0,30}\b(system prompt"
    r"|(hidden|internal|secret|initial|original|your) (instructions|prompt|rules))\b"
    r"|\bsystem prompt\b"
    r"|\b(you are now|pretend (to be|you are)|developer mode|jailbreak)\b"
    r"|\b(ignora|olvida|omite|s[aá]ltate|descarta)\b[^.?!]{0,30}"
    r"\b(instrucciones|reglas|indicaciones|prompt|directrices)\b"
    r"|\b(revela|muestra|mu[eé]strame|imprime|repite|dime)\b[^.?!]{0,30}"
    r"\b(prompt del sistema|tus (instrucciones|reglas)"
    r"|instrucciones (ocultas|internas|secretas|iniciales|originales))\b"
    r"|\bprompt del sistema\b"
    r"|\b(ahora eres|finge (ser|que eres)|modo desarrollador)\b",
    re.IGNORECASE,
)
_SENSITIVE_RE = re.compile(
    r"\b(password|passcode|creds|cvv|cvc|contraseña|clave de (acceso|mi tarjeta))\b"
    r"|\b\d(?:[ -]?\d){12,18}\b",  # 13 to 19 digits: a card number
    re.IGNORECASE,
)

#: A deadline: "30 days", "2-3 business days", "5 to 10 business days", "12-month",
#: "5 a 7 días hábiles", "12 meses".
_CLAIM_RE = re.compile(
    r"\b(?P<low>\d+)(?:\s*(?:[-‐‑‒–—]|to|a)\s*(?P<high>\d+))?[\s‐‑‒–—-]*"
    r"(?P<unit>business days?|days?|weeks?|months?|years?"
    r"|d[ií]as? h[aá]bil(es)?|d[ií]as?|semanas?|meses|mes|años?)\b",
    re.IGNORECASE,
)
#: Canonical (English, singular) unit for each spelling, so a Spanish answer is
#: checked against the same facts as an English one.
_UNIT_PREFIXES = (
    ("day", "day"),
    ("dia", "day"),
    ("week", "week"),
    ("semana", "week"),
    ("month", "month"),
    ("mes", "month"),
    ("year", "year"),
    ("ano", "year"),
)


# --------------------------------------------------------------------------- #
# Input rules
# --------------------------------------------------------------------------- #
def _rule_refund_escalation(message: str) -> str | None:
    amount = largest_amount(message)
    if _REFUND_RE.search(message) and amount is not None and amount > REFUND_LIMIT:
        return f"refund of ${amount:,} is over the ${REFUND_LIMIT} limit"
    return None


def _rule_human_topics(message: str) -> str | None:
    if _TREATMENT_RE.search(message) and _STAFF_RE.search(message):
        return "complaint about an employee"
    if _BILLING_RE.search(message):
        return "billing dispute"
    if _LEGAL_RE.search(message):
        return "legal matter"
    return None


def _rule_injection_or_pii(message: str) -> str | None:
    if _INJECTION_RE.search(message):
        return "possible prompt injection"
    if _SENSITIVE_RE.search(message):
        return "sensitive data (credentials or card) in the message"
    return None


def _rule_off_topic(message: str) -> str | None:
    if re.search(
        r"\b(decrypt|exfiltrate|pentest|hacking|descifrar|hackear|hackeo)\b",
        message,
        re.IGNORECASE,
    ):
        return "request outside the support domain"
    return None


def check_input(message: str) -> Verdict:
    """Evaluate the incoming message and decide the flow."""
    if not message or not message.strip():
        return Verdict()

    for rule in (_rule_injection_or_pii, _rule_off_topic):
        reason = rule(message)
        if reason:
            return Verdict(Action.BLOCK, reason)

    reason = _rule_human_topics(message)
    if reason:
        return Verdict(Action.REFER, reason)

    reason = _rule_refund_escalation(message)
    if reason:
        return Verdict(Action.ESCALATE, reason)

    return Verdict()


# --------------------------------------------------------------------------- #
# Output rules
# --------------------------------------------------------------------------- #
def _output_approves_large_refund(answer: str) -> bool:
    if not re.search(r"\b(approv(e|ed|ing)|aprob\w*|aprueb\w*)\b", answer, re.IGNORECASE):
        return False
    amount = largest_amount(answer)
    return bool(_REFUND_RE.search(answer)) and amount is not None and amount > REFUND_LIMIT


def _canonical_unit(unit: str) -> str:
    plain = unicodedata.normalize("NFKD", unit.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    if plain.startswith("business") or "habil" in plain:
        return "business day"
    return next(canonical for prefix, canonical in _UNIT_PREFIXES if plain.startswith(prefix))


def extract_known_claims(text: str) -> set[str]:
    """Deadlines in ``text``, normalized: "2 to 3 business days" and
    "2 a 3 días hábiles" → "2-3 business day"."""
    claims = set()
    for match in _CLAIM_RE.finditer(text):
        span = match["low"] + (f"-{match['high']}" if match["high"] else "")
        claims.add(f"{span} {_canonical_unit(match['unit'])}")
    return claims


#: Deadlines stated by the policy documents, in any language.
_CORPUS_CLAIMS = frozenset(
    extract_known_claims(" ".join(d.content for docs in KNOWLEDGE.values() for d in docs))
)


def unsupported_claims(answer: str, known_extra: set[str] | None = None) -> set[str]:
    """Deadlines in the answer found neither in the documents nor in ``known_extra``
    (data returned by the order tool, such as an order's estimated delivery)."""
    known = _CORPUS_CLAIMS | extract_known_claims(" ".join(known_extra or ()))
    return extract_known_claims(answer) - known


def check_output(answer: str, known_extra: set[str] | None = None) -> Verdict:
    """Validate the final answer before it reaches the customer."""
    if not answer or not answer.strip():
        return Verdict(Action.BLOCK, "the generated answer is empty")

    if _output_approves_large_refund(answer):
        return Verdict(Action.ESCALATE, "the answer approves a refund over $500")

    unsupported = unsupported_claims(answer, known_extra)
    if unsupported:
        return Verdict(Action.REFER, f"deadlines not found in any source: {sorted(unsupported)}")

    return Verdict()
