"""Unit tests for Phase 5 / Step 2: language detection heuristic."""

from rag_system.retrieval.language_utils import detect_language


class TestDetectLanguage:
    def test_detects_arabic_text(self):
        assert detect_language("الموازنة العامة للدولة") == "ar"

    def test_detects_english_text(self):
        assert detect_language("government budget revenue") == "en"

    def test_defaults_to_english_for_numbers_only(self):
        assert detect_language("12345") == "en"

    def test_defaults_to_english_for_empty_string(self):
        assert detect_language("") == "en"

    def test_mixed_text_picks_majority_script(self):
        # Mostly Arabic with one English word embedded
        assert detect_language("الموازنة العامة للدولة budget") == "ar"
        # Mostly English with one Arabic word embedded
        assert detect_language("the state general budget موازنة") == "en"

    def test_mixed_script_tie_prefers_arabic(self):
        assert detect_language("GDP دين") == "ar"
