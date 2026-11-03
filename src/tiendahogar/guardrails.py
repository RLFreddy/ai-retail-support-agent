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
from tiendahogar.knowledge import DOCUMENTS, SUPPORT_EMAIL

#: Refunds above this amount need human approval.
REFUND_THRESHOLD = 500.0

_AMOUNT_RE = re.compile(r"\$\s*([\d.,]+)")

_TREATMENT_RE = re.compile(
    r"\b(rude|attitude|insult(ed|ing)?|disrespectful|treated|unfriendly)\b", re.IGNORECASE
)
_STAFF_RE = re.compile(
    r"\b(agent|employee|staff|representative|advisor|cashier)\b", re.IGNORECASE
)
_BILLING_RE = re.compile(
    r"\b(dispute|charges|billing|invoice|receipt|statement|double charge)\b", re.IGNORECASE
)
_LEGAL_RE = re.compile(r"\b(legal|lawyer|attorney|lawsuit|sue|court)\b", re.IGNORECASE)

#: "(quantity, time unit)" pairs used to ground the output.
_CLAIM_RE = re.compile(r"(\d+\s*(?:business days?|months?|weeks?|years?))", re.IGNORECASE)


def _parse_amount(text: str) -> tuple[bool, float]:
    """Return (found, amount). Uses the LAST amount when there are several."""
    found = False
    amount = 0.0
    for match in _AMOUNT_RE.finditer(text):
        try:
            amount = float(match.group(1).replace(",", ""))
            found = True
        except ValueError:
            continue
    return found, amount


# --------------------------------------------------------------------------- #
# Input rules
# --------------------------------------------------------------------------- #
def _rule_refund_escalation(message: str) -> str | None:
    found, amount = _parse_amount(message)
    if found and amount > REFUND_THRESHOLD:
        return f"refund over ${REFUND_THRESHOLD:,.0f} (${amount:,.0f})"
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
    lowered = message.lower()
    if re.search(r"\b(system prompt|ignore (previous|all)|reveal instructions)\b", lowered):
        return "possible prompt injection"
    if re.search(r"\b(creds|password|cvv|card\s*(is\s+)?\d{4}|\d{15,16})\b", lowered):
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
    found, amount = _parse_amount(answer)
    mentions_refund = re.search(r"\b(refund|return)\b", answer, re.IGNORECASE) is not None
    return found and amount > REFUND_THRESHOLD and mentions_refund


def extract_known_claims(text: str) -> set[str]:
    """Extract normalized "(quantity, unit)" pairs (lowercase, single spaces)."""
    return {re.sub(r"\s+", " ", c).lower() for c in _CLAIM_RE.findall(text)}


def _output_claims_unknown_fact(answer: str, known_extra: set[str] | None = None) -> bool:
    """True when the answer states a deadline found in no authorized source.

    Authorized sources are the policy documents and ``known_extra`` (data
    returned by the order tool, such as an order's estimated delivery).
    """
    if SUPPORT_EMAIL in answer.lower():
        return False
    claims = extract_known_claims(answer)
    if not claims:
        return False
    known = extract_known_claims(" ".join(doc.content for doc in DOCUMENTS))
    known |= known_extra or set()
    return not claims <= known


def check_output(answer: str, known_extra: set[str] | None = None) -> Verdict:
    """Validate the final answer before it reaches the customer."""
    if not answer or not answer.strip():
        return Verdict(Action.BLOCK, "the generated answer is empty")

    if _output_approves_large_refund(answer):
        return Verdict(Action.ESCALATE, "the answer approves a refund over $500")

    if _output_claims_unknown_fact(answer, known_extra=known_extra):
        return Verdict(Action.REFER, "the answer states deadlines without verified grounding")

    return Verdict()
