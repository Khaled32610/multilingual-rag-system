"""Deterministic grounding and unit completion for answer numerals."""

from __future__ import annotations

import re
import unicodedata

from rag_system.retrieval.language_utils import detect_language

_CITATIONS = re.compile(r"\[\s*(?:مقتطف|excerpt)\s*[٠-٩0-9]+\s*\]", re.IGNORECASE)
_NUMBERS = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_UNIT_ALIASES = (
    (re.compile(r"\s*(?:EGP|Egyptian pounds?|جنيه مصري|جنيهات مصرية)", re.IGNORECASE), "جنيه مصري", "EGP"),
    (re.compile(r"\s*(?:pounds?|جنيه|جنيهات)", re.IGNORECASE), "جنيه", "pounds"),
    (re.compile(r"\s*(?:US\$|USD|dollars?|دولار(?:اً|ا)?|دولارات)", re.IGNORECASE), "دولار", "USD"),
    (re.compile(r"\s*(?:%|percent(?:age)?|٪|بالمئة|في المئة|نسبة مئوية)", re.IGNORECASE), "٪", "%"),
    (re.compile(r"\s*(?:points?|نقطة|نقاط)", re.IGNORECASE), "نقطة", "points"),
    (re.compile(r"\s*(?:years?|سنة|سنوات|عام|أعوام)", re.IGNORECASE), "سنة", "years"),
    (re.compile(r"\s*(?:months?|شهر|أشهر|شهور)", re.IGNORECASE), "شهر", "months"),
    (re.compile(r"\s*(?:days?|يوم|أيام)", re.IGNORECASE), "يوم", "days"),
    (re.compile(r"\s*(?:billions?|مليار|مليارات)", re.IGNORECASE), "مليار", "billion"),
    (re.compile(r"\s*(?:millions?|مليون|ملايين)", re.IGNORECASE), "مليون", "million"),
    (re.compile(r"\s*(?:thousands?|ألف|آلاف)", re.IGNORECASE), "ألف", "thousand"),
)


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_DIGITS).replace("٬", ",").replace("٫", ".")


def _numbers(text: str) -> set[str]:
    return {match.replace(",", "") for match in _NUMBERS.findall(_normalized(text))}


def _unit_in_context(text: str, match: re.Match[str]) -> tuple[str, str] | None:
    """Return an adjacent source unit and its Arabic/English rendering."""
    before = text[max(0, match.start() - 12):match.start()]
    after = text[match.end():match.end() + 32]
    for pattern, arabic, english in _UNIT_ALIASES:
        if pattern.search(after) or pattern.search(before):
            return arabic, english
    return None


def add_source_unit_if_missing(query: str, answer: str, chunks) -> str:
    """Append a unit only when the exact answer number and unit coexist in evidence."""
    clean_answer = _CITATIONS.sub("", answer)
    answer_numbers = list(_NUMBERS.finditer(_normalized(clean_answer)))
    if not answer_numbers:
        return answer

    existing_units = any(pattern.search(clean_answer) for pattern, _, _ in _UNIT_ALIASES)
    if existing_units:
        return answer

    cited_indices = [int(value.translate(_DIGITS)) for value in re.findall(
        r"\[\s*(?:مقتطف|excerpt)\s*([٠-٩0-9]+)\s*\]", answer, re.IGNORECASE
    )]
    evidence = "\n".join(
        chunks[index - 1].text for index in cited_indices if 1 <= index <= len(chunks)
    )
    normalized_evidence = _normalized(evidence)
    source_unit = None
    answer_number = answer_numbers[0].group().replace(",", "")
    for match in _NUMBERS.finditer(normalized_evidence):
        if match.group().replace(",", "") == answer_number:
            source_unit = _unit_in_context(normalized_evidence, match)
            if source_unit:
                break
    if source_unit is None:
        return answer

    unit = source_unit[0] if detect_language(query) == "ar" else source_unit[1]
    # Insert before a citation so the generated citation remains at sentence end.
    citation = re.search(r"\s*(\[\s*(?:مقتطف|excerpt)\s*[٠-٩0-9]+\s*\])", answer, re.IGNORECASE)
    if citation:
        return answer[:citation.start()].rstrip() + f" {unit}" + answer[citation.start():]
    return answer.rstrip() + f" {unit}"


def unsupported_answer_numbers(result: GeneratedAnswer, citation_numbers: tuple[int, ...]) -> tuple[str, ...]:
    """Return answer numbers absent from the excerpts the answer cited."""
    answer = _CITATIONS.sub("", result.answer)
    answer_numbers = _numbers(answer)
    if not answer_numbers:
        return ()

    evidence = "\n".join(
        result.chunks[index - 1].text
        for index in citation_numbers
        if 1 <= index <= len(result.chunks)
    )
    supported_numbers = _numbers(evidence)
    return tuple(sorted(answer_numbers - supported_numbers))
