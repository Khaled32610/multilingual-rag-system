"""Unit tests for PDF table extraction filtering and formatting."""

from rag_system.ingestion.extractor import PDFExtractor


def test_sparse_chart_like_table_is_rejected():
    extractor = PDFExtractor(min_table_density=0.15)
    table = [["axis", None, None, None], [None, None, None, None]]

    assert extractor._is_dense_enough(table) is False


def test_dense_data_table_is_kept():
    extractor = PDFExtractor(min_table_density=0.15)
    table = [["Country", "Average tariff"], ["Egypt", "36.8"]]

    assert extractor._is_dense_enough(table) is True


def test_table_values_remain_attached_to_column_labels():
    table = [
        ["Country", "Average tariff", "Duty-free"],
        ["Egypt", "36.8", "19.1"],
    ]

    text = PDFExtractor._tables_to_text([table], page_number=3)

    assert "[Table 1, page 3]" in text
    assert "Egypt — Average tariff: 36.8, Duty-free: 19.1" in text
