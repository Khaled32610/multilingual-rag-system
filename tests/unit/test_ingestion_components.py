"""Regression coverage for validation, hashing, cleaning, and ingestion failures."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from rag_system.ingestion.cleaner import TextCleaner
from rag_system.ingestion.hasher import PDFHasher
from rag_system.ingestion.pipeline import IngestionPipeline
from rag_system.ingestion.validator import PDFValidator
from rag_system.utils.exceptions import PDFHashingError


def _pdf(path: Path, text: str = "sample") -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Subject": text})
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def test_validator_accepts_structurally_valid_pdf_and_rejects_corrupt(tmp_path):
    valid = PDFValidator().validate(_pdf(tmp_path / "valid.pdf"))
    corrupt_path = tmp_path / "broken.pdf"
    corrupt_path.write_bytes(b"not a pdf")

    corrupt = PDFValidator().validate(corrupt_path)

    assert valid.is_valid and valid.num_pages == 1
    assert not corrupt.is_valid
    assert corrupt.errors


def test_validator_rejects_missing_directory_and_empty_file(tmp_path):
    validator = PDFValidator()
    assert "does not exist" in validator.validate(tmp_path / "missing.pdf").errors[0]
    empty = tmp_path / "empty.pdf"
    empty.touch()
    result = validator.validate(empty)
    assert not result.is_valid
    assert any("empty" in error.lower() for error in result.errors)


def test_validator_enforces_configured_size_and_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rag_system.ingestion.validator.get_settings",
        lambda: SimpleNamespace(pdf=SimpleNamespace(max_file_size_mb=0, allowed_extensions=[".pdf"])),
    )
    path = _pdf(tmp_path / "too-large.pdf")

    result = PDFValidator().validate(path)

    assert not result.is_valid
    assert any("exceeds max" in error for error in result.errors)
    wrong_extension = tmp_path / "notes.txt"
    wrong_extension.write_text("not pdf")
    assert any("Invalid extension" in error for error in PDFValidator().validate(wrong_extension).errors)


def test_hasher_matches_sha256_and_missing_path_raises(tmp_path):
    path = tmp_path / "content.pdf"
    path.write_bytes(b"identical bytes")
    hasher = PDFHasher()

    assert hasher.compute_hash(path) == hashlib.sha256(b"identical bytes").hexdigest()
    assert hasher.find_duplicates({"a": "same", "b": "same", "c": "other"}) == {"same": ["a", "b"]}
    with pytest.raises(PDFHashingError):
        hasher.compute_hash(tmp_path / "absent.pdf")


def test_cleaner_removes_page_numbers_repairs_hyphenation_and_preserves_arabic():
    text = "Page 12\nThe govern-\nment   budget\n\n\nالموازنة"

    cleaned = TextCleaner().clean(text)

    assert cleaned == "The government budget\n\nالموازنة"


def test_cleaner_keeps_standalone_report_numbers_and_years_inside_page():
    text = "Authors\nResearch papers\nNo\n370\nAUGUST\n2025\nThe Evolution of Social Protection"

    cleaned = TextCleaner().clean(text)

    assert "370" in cleaned
    assert "2025" in cleaned


def test_cleaner_removes_bare_page_numbers_only_at_page_edges():
    text = "12\nA report with a numbered result\n370\nPage 14"

    cleaned = TextCleaner().clean(text)

    assert cleaned == "A report with a numbered result\n370"


def test_pipeline_returns_validation_failure_without_hash_or_extract(tmp_path, monkeypatch):
    pipeline = IngestionPipeline()
    monkeypatch.setattr(pipeline.validator, "validate", lambda path: SimpleNamespace(
        is_valid=False, file_size_mb=1.0, errors=["bad"]
    ))
    monkeypatch.setattr(pipeline.hasher, "compute_hash", lambda path: pytest.fail("must not hash invalid file"))

    result = pipeline.ingest(tmp_path / "bad.pdf")

    assert not result.is_ready_for_chunking
    assert result.validation_errors == ["bad"]


def test_pipeline_cleans_pages_and_returns_ready_metadata(tmp_path, monkeypatch):
    path = _pdf(tmp_path / "ready.pdf")
    pipeline = IngestionPipeline()
    monkeypatch.setattr(pipeline.validator, "validate", lambda source: SimpleNamespace(
        is_valid=True, file_size_mb=0.1, errors=[]
    ))
    monkeypatch.setattr(pipeline.hasher, "compute_hash", lambda source: "a" * 64)
    monkeypatch.setattr(pipeline.extractor, "extract", lambda source: SimpleNamespace(
        total_pages=1, pages=[SimpleNamespace(text="Value-\nadded   tax")],
        is_successful=True, errors=[],
    ))

    result = pipeline.ingest(path)

    assert result.is_ready_for_chunking
    assert result.cleaned_pages == ["Valueadded tax"]
    assert result.content_hash == "a" * 64
