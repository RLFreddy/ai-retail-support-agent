"""Guardrails of the TiendaHogar agent.

Pure, deterministic rules applied at two points:

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

from tiendahogar.domain import Action, Verdict
from tiendahogar.knowledge import DOCUMENTS
from tiendahogar.money import REFUND_LIMIT, largest_amount

_REFUND_RE = re.compile(r"\b(refund(ed|s)?|reimburse(ment)?|money back)\b", re.IGNORECASE)

_TREATMENT_RE = re.compile(
    r"\b(rude|attitude|insult(ed|ing)?|disrespectful|treated|unfriendly)\b", re.IGNORECASE
)
_STAFF_RE = re.compile(r"\b(agent|employee|staff|representative|advisor|cashier)\b", re.IGNORECASE)
#: A contested charge. Asking for an invoice or a receipt is not a dispute.
_BILLING_RE = re.compile(
    r"\b(disputed?|overcharged|charged (me )?(twice|double|wrong(ly)?)|double charged?"
    r"|unauthori[sz]ed charge|billing (error|issue|problem|mistake))\b",
    re.IGNORECASE,
)
_LEGAL_RE = re.compile(r"\b(legal|lawyer|attorney|lawsuit|sue|court)\b", re.IGNORECASE)

_INJECTION_RE = re.compile(
    r"\b(ignore|disregard|forget|override|bypass)\b[^.?!]{0,30}"
    r"\b(instructions?|rules|prompt|guidelines)\b"
    r"|\b(reveal|show|print|display|repeat|output|tell me)\b[^.?!]{0,30}\b(system prompt"
    r"|(hidden|internal|secret|initial|original|your) (instructions|prompt|rules))\b"
    r"|\bsystem prompt\b"
    r"|\b(you are now|pretend (to be|you are)|developer mode|jailbreak)\b",
    re.IGNORECASE,
)
_SENSITIVE_RE = re.compile(
    r"\b(password|passcode|creds|cvv|cvc)\b"
    r"|\b\d(?:[ -]?\d){12,18}\b",  # 13 to 19 digits: a card number
    re.IGNORECASE,
)

#: A deadline: "30 days", "2-3 business days", "5 to 10 business days", "12-month".
_CLAIM_RE = re.compile(
    r"\b(?P<low>\d+)(?:\s*(?:[-‐‑‒–—]|to)\s*(?P<high>\d+))?[\s‐‑‒–—-]*"
    r"(?P<unit>business days?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
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
    if re.search(r"\b(decrypt|exfiltrate|pentest|hacking)\b", message, re.IGNORECASE):
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
    if not re.search(r"\bapprov(e|ed|ing)\b", answer, re.IGNORECASE):
        return False
    amount = largest_amount(answer)
    return bool(_REFUND_RE.search(answer)) and amount is not None and amount > REFUND_LIMIT


def extract_known_claims(text: str) -> set[str]:
    """Deadlines in ``text``, normalized: "2 to 3 business days" → "2-3 business day"."""
    claims = set()
    for match in _CLAIM_RE.finditer(text):
        span = match["low"] + (f"-{match['high']}" if match["high"] else "")
        claims.add(f"{span} {match['unit'].lower().removesuffix('s')}")
    return claims


#: Deadlines stated by the policy documents.
_CORPUS_CLAIMS = frozenset(extract_known_claims(" ".join(d.content for d in DOCUMENTS)))


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
