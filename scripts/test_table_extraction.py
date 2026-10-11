"""Quick before/after check: does the new table extraction actually
produce readable label:value lines for the WTO tariff profile page?"""

from rag_system.ingestion.extractor import PDFExtractor

extractor = PDFExtractor(enable_table_extraction=True)
result = extractor.extract("data/raw/28_WTO_Tariff_Profile_Egypt.pdf")

print(f"Pages: {result.total_pages}, chars: {result.total_chars}")
print(f"Has tables: {[p.has_tables for p in result.pages]}")
print()
print(result.full_text[:2000])
