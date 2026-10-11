"""
Cross-lingual query translation for BM25 (Phase 5 / Step 2).

Confirmed real-world scenario for this project: source documents are
mostly English (some Arabic), but questions are asked mostly in
Arabic. Vector search (Phase 4) already bridges this on its own,
because `multilingual-e5-base` embeds Arabic and English text into the
same semantic space. BM25 (Phase 5 / Step 1) cannot: it only matches
exact vocabulary, so an Arabic query has zero lexical overlap with an
English document no matter how relevant that document is.

This module closes that specific gap for BM25 only, by translating a
query to English before *also* searching with it (see
`BM25Retriever.retrieve`, which runs both searches and merges them).
It deliberately does NOT translate documents at indexing time: that
would double storage, cost far more compute, and complicate Phase 14's
incremental indexing.

Uses Helsinki-NLP/opus-mt-ar-en via `transformers` (already a
dependency via sentence-transformers) - a small (~300MB), local model,
consistent with this project's all-local requirement.

Phase 9-B fix: opus-mt output was hurting Arabic retrieval (hit@5 28/35
vs 34/35 for English). Three problems were observed in the logs and are
handled here, on the real-model path only:
  1. HTML artifacts ("Egypt &apos; s") -> `_clean_translation`.
  2. Financial terms mistranslated (budget sector, primary balance,
     foreign portfolio, bound tariff) and acronyms expanded to full
     names while the documents use WTO / AFD -> `_GLOSSARY`.
  3. Date ranges truncated ("July-November" became "July") ->
     `_MONTH_RANGE`.
"""

from __future__ import annotations

import html
import re
from typing import Callable

from rag_system.retrieval.language_utils import detect_language
from rag_system.utils.exceptions import TranslationError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ar-en"

_AR_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def _norm_ar(text: str) -> str:
    """Light Arabic normalization so glossary matching survives spelling variants."""
    text = _AR_DIACRITICS.sub("", text)
    text = re.sub("[أإآ]", "ا", text)
    return text.replace("ى", "ي")


# (Arabic term, English terms appended to the translated query).
# Matching is substring-based on normalized text. Extend freely.
_GLOSSARY_RAW: list[tuple[str, str]] = [
    ("منظمة التجارة العالمية", "WTO World Trade Organization"),
    ("الوكالة الفرنسية للتنمية", "AFD Agence Francaise de Developpement"),
    ("ورقة البحث", "research paper research paper number publication date"),
    ("ورقة بحث", "research paper research paper number publication date"),
    ("معاش الضمان الاجتماعي", "Daman pension social assistance pension"),
    ("سعر تحويل", "exchange rate FX rate currency conversion USD EGP"),
    ("سعر صرف", "exchange rate FX rate currency conversion USD EGP"),
    ("رقم ورقة", "research paper number publication date"),
    ("تاريخ نشر", "publication date publication date"),
    ("أكبر سوق", "largest market major markets export destination"),
    ("السقف القانوني", "legal limit statutory ceiling"),
    ("تسهيل السحب على المكشوف", "overdraft facility overdraft limit"),
    ("غير الزراعية", "non-agricultural non-agricultural products"),
    ("الواردات الثنائية", "bilateral imports trade partner imports"),
    ("جهاز الموازنة", "budget sector"),
    ("الرصيد الأولي", "primary balance"),
    ("محافظ الأجانب", "foreign portfolio investment"),
    ("المقيدة", "bound binding"),
    ("تقييد", "bound binding"),
    ("مؤشر مديري المشتريات", "PMI Purchasing Managers Index"),
    ("صافي الاحتياطيات الدولية", "NIR net international reserves"),
    ("صندوق النقد الدولي", "IMF"),
    ("البنك المركزي", "CBE Central Bank of Egypt"),
    ("السحب على المكشوف", "overdraft"),
    ("الناتج المحلي الإجمالي", "GDP"),
]
_GLOSSARY: list[tuple[str, str]] = [(_norm_ar(ar), en) for ar, en in _GLOSSARY_RAW]

_ENGLISH_GLOSSARY_RAW: list[tuple[str, str]] = [
    ("simple average final bound tariff", "simple average final bound rate"),
    ("average final bound tariff", "simple average final bound rate"),
    ("non-agricultural exports", "non-agricultural products export destination"),
    ("bilateral import value", "bilateral imports trade partner imports major markets"),
    ("destination market", "main export destination major markets"),
    ("largest market", "major markets leading destination"),
    ("top market", "major markets leading destination"),
    ("USD-to-EGP exchange rate", "USD EGP exchange rate FX rate"),
    ("dollar into the pound", "USD EGP exchange rate FX rate"),
    ("research-paper number", "research paper number AFD research papers publication date"),
    ("research paper number", "research paper number AFD research papers publication date"),
    ("publication date", "publication date research paper number publication date"),
    ("statutory ceiling", "legal limit overdraft facility"),
    ("overdraft facility", "overdraft legal limit"),
]

_MONTHS = {
    "يناير": "January",
    "فبراير": "February",
    "مارس": "March",
    "ابريل": "April",
    "مايو": "May",
    "يونيو": "June",
    "يوليو": "July",
    "اغسطس": "August",
    "سبتمبر": "September",
    "اكتوبر": "October",
    "نوفمبر": "November",
    "ديسمبر": "December",
}
_MONTH_RANGE = re.compile(
    r"(%s)\s*[\u2013\u2014\-]\s*(%s)" % ("|".join(_MONTHS), "|".join(_MONTHS))
)


def _clean_translation(text: str) -> str:
    """Fix opus-mt artifacts: HTML entities and detached apostrophes."""
    text = html.unescape(text)
    text = re.sub(r"\s+'\s*s\b", "'s", text)
    return text.strip()


def expand_english_query(query: str) -> str:
    """Add domain synonyms for known English paraphrases without replacing the query."""
    normalized = query.lower()
    extras = [english for phrase, english in _ENGLISH_GLOSSARY_RAW if phrase in normalized]
    return f"{query} {' '.join(extras)}".strip() if extras else query


def _expand_with_glossary(query_ar: str, translated: str) -> str:
    """Append correct English equivalents for known terms found in the Arabic query."""
    norm = _norm_ar(query_ar)
    extras: list[str] = []
    for ar, en in _GLOSSARY:
        if ar in norm:
            extras.append(en)
    for m in _MONTH_RANGE.finditer(norm):
        extras.append(f"{_MONTHS[m.group(1)]} {_MONTHS[m.group(2)]}")
    return f"{translated} {' '.join(extras)}".strip() if extras else translated


class QueryTranslator:
    """
    Translates non-English (in practice: Arabic) queries to English.

    The underlying HuggingFace model is loaded lazily, on first use
    rather than at construction, since it's a real model download
    (~300MB, first run only) that not every query needs -
    `translate_to_english` skips it entirely for queries already
    detected as English.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TRANSLATION_MODEL,
        translate_fn: Callable[[str], str] | None = None,
    ) -> None:
        """
        `translate_fn`, if given, replaces the real model entirely.
        This exists for tests, so they can inject a deterministic fake
        translation function instead of downloading/running a real model.
        """
        self.model_name = model_name
        self._translate_fn = translate_fn
        self._tokenizer = None  # lazy-loaded transformers tokenizer
        self._model = None  # lazy-loaded transformers seq2seq model

    def _load_model(self):
        """
        Load the tokenizer/model directly via AutoTokenizer +
        AutoModelForSeq2SeqLM rather than `transformers.pipeline(...)`,
        whose "translation" task name was rejected on transformers 5.17.0.
        """
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                logger.info(f"Loading translation model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            except Exception as e:
                raise TranslationError(
                    f"Failed to load translation model: {self.model_name}",
                    details={"error": str(e)},
                ) from e
        return self._tokenizer, self._model

    def translate_to_english(self, text: str) -> str:
        """
        Return an English translation of `text`.

        If `text` is already detected as English, it's returned
        unchanged (no wasted model call). Raises `TranslationError` on
        failure - callers that want to degrade gracefully (e.g.
        BM25Retriever) should catch it.
        """
        if not text or not text.strip():
            return text

        if detect_language(text) == "en":
            return text

        if self._translate_fn is not None:
            return self._translate_fn(text)

        tokenizer, model = self._load_model()
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            # max_length=None avoids a warning: MarianMT's generation_config
            # ships a default max_length that otherwise conflicts with the
            # max_new_tokens we set explicitly below.
            output_ids = model.generate(**inputs, max_new_tokens=256, max_length=None)
            raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return _expand_with_glossary(text, _clean_translation(raw))
        except Exception as e:
            raise TranslationError(
                f"Failed to translate query: {text!r}", details={"error": str(e)}
            ) from e