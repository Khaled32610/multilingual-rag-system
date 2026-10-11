"""
PDF text extraction (Phase 1 / Step 3).

Extracts text page-by-page from validated PDF files using pdfplumber,
which handles complex layouts (including Arabic/RTL text) better than
lighter-weight libraries. Each page's text is kept separate so later
phases (chunking, citation) can trace content back to its exact page.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from rag_system.ingestion.models import ExtractionResult, PageContent
from rag_system.utils.exceptions import PDFExtractionError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """Extracts text content from PDF files, page by page."""

    def __init__(
        self,
        enable_table_extraction: bool = True,
        max_table_text_ratio: float = 3.0,
        min_table_density: float = 0.15,
    ) -> None:
        """
        Args:
            enable_table_extraction: if True (default), pdfplumber's
                extract_tables() is run on every page alongside
                extract_text(), and any detected tables that pass the
                sanity checks below are appended to that page's text as
                readable "label: value" lines. This exists because
                extract_text() alone flattens multi-column tables into a
                single line of numbers with no reliable link back to
                their row/column labels — but blind table extraction is
                dangerous on chart-heavy report pages (see the two guards
                below), so both are on by default and can be tuned or
                disabled.
            max_table_text_ratio: safety guard. If the text produced from
                a page's detected tables is more than this many times
                longer than the page's plain extract_text() output, the
                tables are treated as spurious (typically pdfplumber
                mis-reading chart/graphic gridlines as table borders on
                dashboard-style report pages) and discarded entirely for
                that page, with a warning logged. A real data table is
                rarely more than 2-3x the surrounding prose; chart-noise
                "tables" commonly come out 10-50x larger.
            min_table_density: safety guard per individual table. If the
                fraction of non-empty cells in a detected table is below
                this threshold, that single table is skipped (very sparse
                grids are almost always decorative lines, not data),
                before it ever contributes to the ratio check above.
        """
        self.enable_table_extraction = enable_table_extraction
        self.max_table_text_ratio = max_table_text_ratio
        self.min_table_density = min_table_density

    def extract(self, file_path: str | Path) -> ExtractionResult:
        """
        Extract text from every page of a PDF file.

        Returns an ExtractionResult containing per-page text plus the
        full concatenated text. Does not raise on partial failures
        (e.g. one unreadable page) — errors are collected and reported
        in the result; call `.raise_if_failed()` if you want a hard stop.
        """
        path = Path(file_path)
        logger.info(f"Extracting text from: {path.name}")

        pages: list[PageContent] = []
        errors: list[str] = []

        try:
            with pdfplumber.open(str(path)) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"{path.name}: {total_pages} page(s) to extract")

                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                        has_tables = False
                        tables_discarded = 0

                        if self.enable_table_extraction:
                            try:
                                raw_tables = page.extract_tables()
                            except Exception as e:
                                # Table extraction is a best-effort enrichment on
                                # top of extract_text(), which already succeeded
                                # above — never let a table-parsing hiccup turn
                                # a working page extraction into a failed one.
                                raw_tables = []
                                logger.warning(
                                    f"{path.name} page {i}: table extraction failed, "
                                    f"keeping plain text only ({e})"
                                )

                            # Guard 1: drop individual tables that are mostly
                            # empty cells (decorative gridlines, not data).
                            dense_tables = []
                            for t in raw_tables:
                                if self._is_dense_enough(t):
                                    dense_tables.append(t)
                                else:
                                    tables_discarded += 1

                            # Guard 2: evaluate each surviving table ON ITS OWN
                            # against the page's plain text, not the combined
                            # total of every table on the page. A page can
                            # legitimately hold one real data table (compact,
                            # size comparable to the surrounding prose) next
                            # to one or more chart/graphic mis-reads (huge,
                            # many multiples of the prose). Judging the whole
                            # page as one blob would throw the real table out
                            # along with the noise; judging table-by-table
                            # keeps the real one and drops only the noise.
                            plain_len = len(text.strip())
                            kept_texts = []
                            for t in dense_tables:
                                t_text = self._tables_to_text([t], page_number=i)
                                if not t_text:
                                    continue
                                if plain_len > 0 and len(t_text) > self.max_table_text_ratio * plain_len:
                                    logger.warning(
                                        f"{path.name} page {i}: discarding one table "
                                        f"({len(t_text)} chars vs {plain_len} plain chars, "
                                        f"ratio {len(t_text) / plain_len:.1f}x > "
                                        f"{self.max_table_text_ratio}x) — likely chart/"
                                        f"graphic misread as a table"
                                    )
                                    tables_discarded += 1
                                else:
                                    kept_texts.append(t_text)

                            if kept_texts:
                                table_text = "\n\n".join(kept_texts)
                                has_tables = True
                                text = f"{text}\n\n{table_text}" if text.strip() else table_text

                        pages.append(
                            PageContent(
                                page_number=i,
                                text=text,
                                has_tables=has_tables,
                                tables_discarded=tables_discarded,
                            )
                        )
                        if not text.strip():
                            logger.warning(f"{path.name} page {i}: no extractable text (possibly scanned/image)")
                    except Exception as e:
                        error_msg = f"Failed to extract page {i}: {e}"
                        errors.append(error_msg)
                        logger.error(f"{path.name}: {error_msg}")
                        pages.append(PageContent(page_number=i, text=""))

        except Exception as e:
            error_msg = f"Failed to open PDF for extraction: {e}"
            errors.append(error_msg)
            logger.error(f"{path.name}: {error_msg}")
            return ExtractionResult(
                file_path=path,
                pages=[],
                full_text="",
                total_pages=0,
                total_chars=0,
                errors=errors,
            )

        full_text = "\n\n".join(p.text for p in pages if p.text.strip())
        total_chars = sum(p.char_count for p in pages)

        result = ExtractionResult(
            file_path=path,
            pages=pages,
            full_text=full_text,
            total_pages=len(pages),
            total_chars=total_chars,
            errors=errors,
        )

        total_discarded = sum(p.tables_discarded for p in pages)
        if result.is_successful:
            logger.info(
                f"Extraction complete: {path.name} — "
                f"{result.total_pages} pages, {result.total_chars} chars"
                + (f", {total_discarded} spurious table(s) discarded" if total_discarded else "")
            )
        else:
            logger.warning(
                f"Extraction finished with issues: {path.name} — "
                f"{len(errors)} error(s), {total_chars} chars extracted"
            )

        return result

    def _is_dense_enough(self, table: list[list[str | None]]) -> bool:
        """
        Returns False if a detected table is mostly empty cells — a
        strong signal it's decorative gridlines (chart axes, borders)
        that pdfplumber mistook for a data table, not real content.
        """
        total_cells = sum(len(row) for row in table if row)
        if total_cells == 0:
            return False
        non_empty = sum(1 for row in table if row for c in row if (c or "").strip())
        density = non_empty / total_cells
        return density >= self.min_table_density

    @staticmethod
    def _tables_to_text(tables: list[list[list[str | None]]], page_number: int) -> str:
        """
        Convert pdfplumber's extract_tables() output into readable
        "row_label — column: value" lines instead of a scrambled grid
        of numbers.

        Assumes a single header row (tables[t][0]) mapped to every
        subsequent row by column position, and treats the first column
        as the row's label (e.g. a country or category name). This
        covers the common one-row-header case cleanly.

        LIMITATION: tables with a multi-row / merged header (a header
        row spanning grouped sub-columns, e.g. "Final bound duties" over
        "AVG | Duty-free | Max") are NOT unwrapped correctly — pdfplumber
        returns each header row as a separate table row, so only the
        row immediately above the data is used as the column names,
        and the higher-level group label is dropped. For those tables
        this still beats the flattened-text baseline (values stay
        attached to their nearest column name and row label) but the
        group label (e.g. which duty regime a column belongs to) is
        lost. If your documents rely heavily on that kind of table,
        this needs a follow-up pass — flag it and we'll extend it.
        """
        blocks: list[str] = []

        for t_idx, table in enumerate(tables, start=1):
            if not table or not table[0]:
                continue

            header = [(c or "").strip() for c in table[0]]
            data_rows = table[1:]
            if not any(header) or not data_rows:
                continue

            lines: list[str] = []
            for row in data_rows:
                if not row or not any(row):
                    continue

                split_cells = [(c or "").split("\n") for c in row]
                other_cells = split_cells[1:]
                max_other_len = max((len(c) for c in other_cells), default=0)
                label_lines = split_cells[0]
                extra = max(0, len(label_lines) - max_other_len) if other_cells else 0
                for heading in label_lines[:extra]:
                    if heading.strip():
                        lines.append(heading.strip())
                aligned_cells = [label_lines[extra:]] + other_cells
                n_sub_rows = max((len(sc) for sc in aligned_cells), default=1)

                for sub_i in range(n_sub_rows):
                    cells = [sc[sub_i].strip() if sub_i < len(sc) else "" for sc in aligned_cells]
                    if not any(cells):
                        continue

                    row_label = cells[0] if cells[0] else None
                    pairs = []
                    start_idx = 1 if row_label else 0
                    for col_name, value in zip(header[start_idx:], cells[start_idx:]):
                        if not value:
                            continue
                        pairs.append(f"{col_name}: {value}" if col_name else value)

                    if not pairs:
                        continue

                    line = f"{row_label} — {', '.join(pairs)}" if row_label else ", ".join(pairs)
                    lines.append(line)

            if lines:
                blocks.append(f"[Table {t_idx}, page {page_number}]\n" + "\n".join(lines))

        return "\n\n".join(blocks)

    def extract_batch(self, file_paths: list[str | Path]) -> list[ExtractionResult]:
        """Extract text from multiple PDF files."""
        logger.info(f"Extracting batch of {len(file_paths)} PDF file(s)")
        return [self.extract(fp) for fp in file_paths]