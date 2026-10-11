"""Query-focused, extractive views for scoring; original evidence stays untouched."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re

from rag_system.retrieval.models import RetrievedChunk

_STOP = set('what which who when where how was were is are the a an of in on at to for and its by from as with did does this that these those be been it their into about some all much many have has had'.split())


def _terms(text: str) -> set[str]:
    return {word.removesuffix('s') for word in re.findall(r'[^\W_]+', text.lower()) if len(word) > 2 and word not in _STOP}


def evidence_views(query: str, chunks: list[RetrievedChunk], window_words: int = 100) -> list[str]:
    """Pick a contiguous window using query terms and local inverse document frequency.

    Selection uses only the query and retrieved documents, never a reference answer,
    filename preference, question ID, or benchmark-specific vocabulary.
    """
    query_terms = _terms(query)
    frequencies = Counter(term for chunk in chunks for term in _terms(chunk.text))
    weights = {term: math.log(1 + (len(chunks) + 1) / (frequencies[term] + 1)) for term in query_terms}
    views = []
    for chunk in chunks:
        words = chunk.text.split()
        starts = list(range(0, max(1, len(words) - window_words + 1), max(1, window_words // 4)))
        starts.append(max(0, len(words) - window_words))
        def relevance(start: int) -> float:
            return sum(weights[t] for t in _terms(' '.join(words[start:start + window_words])) & query_terms)
        best = max(starts, key=relevance)
        excerpt = ' '.join(words[best:best + window_words])
        title = chunk.metadata.get('document_title')
        if not title:
            title = Path(chunk.metadata.get('source_file', '')).stem.replace('_', ' ')
            title = re.sub(r'^\d+\s+|\b(?:\d{4}|AR|EN)\b', '', title).strip()
            title = ' '.join(title.split())
        views.append(f'Document: {title}\nPassage: {excerpt}' if title else excerpt)
    return views
