"""
Unit tests for Phase 5 / Step 2: QueryTranslator.

Uses an injected `translate_fn` instead of the real Helsinki-NLP model,
so these tests are fast and don't require downloading (or having
network access to download) a ~300MB translation model.
"""

import pytest

from rag_system.retrieval.query_translator import QueryTranslator
from rag_system.utils.exceptions import TranslationError


class TestQueryTranslatorWithFakeFn:
    def test_translates_arabic_text_via_injected_fn(self):
        translator = QueryTranslator(translate_fn=lambda t: "the state general budget")

        result = translator.translate_to_english("الموازنة العامة للدولة")

        assert result == "the state general budget"

    def test_english_text_skips_translation_entirely(self):
        # Should return unchanged WITHOUT calling translate_fn, since the
        # text is already detected as English.
        def boom(_text: str) -> str:
            raise AssertionError("translate_fn should not be called for English text")

        translator = QueryTranslator(translate_fn=boom)

        result = translator.translate_to_english("government budget revenue")

        assert result == "government budget revenue"

    def test_empty_string_returns_unchanged(self):
        translator = QueryTranslator(translate_fn=lambda t: "should not be used")
        assert translator.translate_to_english("") == ""
        assert translator.translate_to_english("   ") == "   "


class TestQueryTranslatorRealPipelineErrors:
    def test_model_load_failure_raises_translation_error(self, monkeypatch):
        """
        Without an injected translate_fn, a non-English query attempts
        to load the real HF tokenizer/model. We simulate that failing
        (e.g. no network / model not cached / bad model name) and
        expect a clean TranslationError, not a raw exception leaking out.
        """
        translator = QueryTranslator(model_name="this-model-does-not-exist")

        def fake_from_pretrained(*args, **kwargs):
            raise OSError("model not found")

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained", fake_from_pretrained
        )

        with pytest.raises(TranslationError):
            translator.translate_to_english("الموازنة العامة للدولة")

def test_cleans_html_entities_and_detached_possessives():
    from rag_system.retrieval.query_translator import _clean_translation

    assert _clean_translation("Egypt &apos; s budget") == "Egypt's budget"


def test_glossary_expands_financial_terms_and_month_ranges():
    from rag_system.retrieval.query_translator import _expand_with_glossary

    expanded = _expand_with_glossary(
        "ما هو الرصيد الأولي خلال يوليو–نوفمبر؟", "What is the initial balance?"
    )

    assert "primary balance" in expanded
    assert "July November" in expanded


def test_glossary_expands_paraphrased_financial_and_social_terms():
    from rag_system.retrieval.query_translator import _expand_with_glossary

    translated = _expand_with_glossary(
        "ما سعر تحويل الدولار؟ وما السقف القانوني لتسهيل السحب على المكشوف؟",
        "What is the price of converting the dollar and the legal ceiling?",
    )

    assert "exchange rate" in translated
    assert "legal limit statutory ceiling" in translated
    assert "overdraft facility" in translated


def test_glossary_expands_alternate_arabic_program_name():
    from rag_system.retrieval.query_translator import _expand_with_glossary

    translated = _expand_with_glossary(
        "في أي برنامج أُدمج معاش الضمان الاجتماعي السابق؟",
        "Which program absorbed the former social security pension?",
    )

    assert "Daman pension" in translated
    assert "social assistance pension" in translated
