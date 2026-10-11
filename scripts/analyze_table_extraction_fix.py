"""
Diagnostic (Phase 1 follow-up): compares extraction size across the full
corpus with table extraction OFF, with the OLD (unguarded) behavior, and
with the NEW guarded behavior — to confirm the fix actually kills the
chart-as-table explosion without losing real table data (e.g. the WTO
Tariff Profile file, which genuinely needs table extraction).

Run from project root:
    python scripts/analyze_table_extraction_fix.py
"""

from pathlib import Path

from rag_system.ingestion.extractor import PDFExtractor
from rag_system.utils.config import get_settings

RAW_DIR = Path(get_settings().paths.raw_pdf_dir)


def run_one(pdf_path: Path, **extractor_kwargs) -> tuple[int, int, int]:
    """Returns (total_chars, pages_with_tables, tables_discarded)."""
    extractor = PDFExtractor(**extractor_kwargs)
    result = extractor.extract(pdf_path)
    pages_with_tables = sum(1 for p in result.pages if p.has_tables)
    tables_discarded = sum(getattr(p, "tables_discarded", 0) for p in result.pages)
    return result.total_chars, pages_with_tables, tables_discarded


def main() -> None:
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}")
        return

    header = (
        f"{'File':<55} {'no_tables':>10} {'old(unguarded)':>15} "
        f"{'new(guarded)':>13} {'discarded':>10}"
    )
    print(header)
    print("-" * len(header))

    for pdf_path in pdf_files:
        chars_no_tables, _, _ = run_one(pdf_path, enable_table_extraction=False)

        # "old" behavior = guards disabled (ratio/density thresholds set
        # so nothing is ever discarded), to reproduce the pre-fix numbers
        chars_old, pages_old, _ = run_one(
            pdf_path,
            enable_table_extraction=True,
            max_table_text_ratio=float("inf"),
            min_table_density=0.0,
        )

        chars_new, pages_new, discarded = run_one(
            pdf_path,
            enable_table_extraction=True,
        )

        print(
            f"{pdf_path.name:<55} {chars_no_tables:>10} {chars_old:>15} "
            f"{chars_new:>13} {discarded:>10}"
        )


if __name__ == "__main__":
    main()
