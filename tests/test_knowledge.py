"""The English and Spanish knowledge bases describe the same policies."""

from tiendahogar.domain import Language
from tiendahogar.knowledge import KNOWLEDGE, default_system_prompt


def test_both_languages_have_the_same_documents_in_the_same_order():
    assert [d.id for d in KNOWLEDGE[Language.EN]] == [d.id for d in KNOWLEDGE[Language.ES]]


def test_the_spanish_prompt_asks_for_spanish_citations():
    prompt = default_system_prompt(Language.ES)
    assert "Responde en español" in prompt
    assert "(Fuente: " in prompt
