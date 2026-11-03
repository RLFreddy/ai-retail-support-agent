"""Default promptfoo assertion: grade the agent's action and cited sources.

Test variables:

- ``expected_action``: required action (``answer``, ``refer``, ``escalate``,
  ``block``).
- ``expected_sources``: comma-separated document indices that must be cited
  exactly, ``none`` when nothing may be cited, or unset to skip the check.
- ``includes_sources``: comma-separated indices that must be among the cited
  ones (other documents may be cited too).
- ``forbidden``: comma-separated substrings that must not appear in the answer
  (for example an invented deadline).
"""

from __future__ import annotations

import json


def _split(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def get_assert(output: str, context: dict) -> dict:
    test_vars = context.get("vars", {})
    reply = json.loads(output)
    problems: list[str] = []

    expected_action = test_vars.get("expected_action")
    if expected_action and reply["action"] != expected_action:
        problems.append(f"action {reply['action']!r}, expected {expected_action!r}")

    expected_sources = test_vars.get("expected_sources")
    if expected_sources is not None and expected_sources != "":
        wanted = set() if expected_sources == "none" else {int(i) for i in _split(expected_sources)}
        if set(reply["sources"]) != wanted:
            problems.append(f"sources {sorted(reply['sources'])}, expected {sorted(wanted)}")

    required = {int(i) for i in _split(test_vars.get("includes_sources", ""))}
    if not required <= set(reply["sources"]):
        problems.append(f"sources {sorted(reply['sources'])} miss {sorted(required)}")

    answer = reply["answer"].lower()
    for text in _split(test_vars.get("forbidden", "")):
        if text.lower() in answer:
            problems.append(f"answer contains forbidden text {text!r}")

    return {
        "pass": not problems,
        "score": 0.0 if problems else 1.0,
        "reason": "; ".join(problems) or "action and sources match",
    }
