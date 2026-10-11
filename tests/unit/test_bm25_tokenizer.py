"""Unit tests for Phase 5 / Step 18: BM25 tokenizer normalization."""

from rag_system.retrieval.bm25_tokenizer import normalize_arabic_word, normalize_tokens


class TestNormalizeArabicWord:
    def test_removes_diacritics(self):
        # "مَدْرَسَة" (with tashkeel) should normalize the same as "مدرسة"
        assert normalize_arabic_word("مَدْرَسَة") == normalize_arabic_word("مدرسة")

    def test_removes_tatweel(self):
        assert normalize_arabic_word("كـــتاب") == normalize_arabic_word("كتاب")

    def test_unifies_alef_variants(self):
        base = normalize_arabic_word("احمد")
        assert normalize_arabic_word("أحمد") == base
        assert normalize_arabic_word("إحمد") == base
        assert normalize_arabic_word("آحمد") == base

    def test_unifies_alef_maksura_and_yeh(self):
        assert normalize_arabic_word("على") == normalize_arabic_word("علي")

    def test_unifies_taa_marbuta_and_haa(self):
        assert normalize_arabic_word("ضريبة") == normalize_arabic_word("ضريبه")

    def test_english_word_passes_through_unchanged(self):
        assert normalize_arabic_word("budget") == "budget"

    def test_does_not_stem_prefixes_or_suffixes(self):
        # Normalization must NOT strip "ال" or possessive suffixes —
        # that would be stemming, which this project deliberately avoids.
        assert normalize_arabic_word("الكتاب") != normalize_arabic_word("كتاب")
        assert normalize_arabic_word("كتابها") != normalize_arabic_word("كتاب")


class TestNormalizeTokens:
    def test_normalizes_each_token_in_list(self):
        tokens = ["أحمد", "budget", "إحمد"]
        result = normalize_tokens(tokens)
        assert result[0] == result[2]  # both alef variants collapse together
        assert result[1] == "budget"

    def test_empty_list_returns_empty_list(self):
        assert normalize_tokens([]) == []

    def test_drops_tokens_that_normalize_to_empty_string(self):
        # A token that is pure diacritics/tatweel normalizes to "".
        tokens = ["كتاب", "\u064B\u064B", "budget"]
        result = normalize_tokens(tokens)
        assert "" not in result
        assert len(result) == 2
