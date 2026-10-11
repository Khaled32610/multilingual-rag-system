"""Tests that ingestion uses the configured table extraction guards."""

from types import SimpleNamespace

from rag_system.ingestion import pipeline as ingestion_pipeline


def test_pipeline_passes_pdf_config_to_extractor(monkeypatch):
    expected = {
        "enable_table_extraction": False,
        "max_table_text_ratio": 2.5,
        "min_table_density": 0.2,
    }
    monkeypatch.setattr(
        ingestion_pipeline,
        "get_settings",
        lambda: SimpleNamespace(pdf=SimpleNamespace(**expected)),
    )
    captured = {}

    class FakeExtractor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ingestion_pipeline, "PDFExtractor", FakeExtractor)

    ingestion_pipeline.IngestionPipeline()

    assert captured == expected
