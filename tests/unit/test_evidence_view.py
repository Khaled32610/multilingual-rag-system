from rag_system.reranking.evidence_view import evidence_views
from rag_system.retrieval.models import RetrievedChunk


def chunk(text, source='03_Research_Report_2020_EN.pdf'):
    return RetrievedChunk('x',text,{'source_file':source},0.5,1)


def test_focus_finds_evidence_at_end_and_preserves_original():
    original='unrelated material ' * 150 + 'The legal borrowing ceiling is twelve percent of revenue.'
    item=chunk(original)
    view=evidence_views('What is the legal borrowing ceiling?', [item],window_words=30)[0]
    assert 'twelve percent' in view
    assert item.text==original
    assert '2020' not in view


def test_short_passage_kept_without_inventing_evidence():
    text='Authors\nJane Example\nJohn Example'
    view=evidence_views('Who are the authors?', [chunk(text)],window_words=30)[0]
    assert 'Authors Jane Example John Example' in view


def test_no_benchmark_or_gold_metadata_used():
    a=chunk('A neutral evidence passage.')
    b=chunk(a.text)
    b.metadata.update(gold_chunk='injected answer',answer='injected answer',question_id='94')
    assert evidence_views('Question', [a])==evidence_views('Question', [b])
