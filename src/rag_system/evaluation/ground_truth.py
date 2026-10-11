"""
Ground-truth query set for Phase 8 (Retrieval Benchmark).

Manually labeled against the 14 chunks of the real test document
(01_Enacted_Budget_Law_2024_AR.pdf). Each entry maps a query to the
chunk_id(s) considered a correct/relevant answer, used to compute
Recall@k and MRR across retrieval methods.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GroundTruthQuery:
    query: str
    relevant_chunk_ids: list[str]
    language: str  # "ar" or "en"
    note: str = ""


GROUND_TRUTH: list[GroundTruthQuery] = [
    GroundTruthQuery(
        query="وزير المالية يجمع مصاريف إدارية",
        relevant_chunk_ids=["45785c06918cbb7e02ccc2ac"],
        language="ar",
        note="Minister of Finance collecting administrative expenses",
    ),
    GroundTruthQuery(
        query="government budget revenue",
        relevant_chunk_ids=["82b7f35cc380d3a7acb7b70c"],
        language="en",
    ),
    GroundTruthQuery(
        query="الموازنة العامة للدولة قانون رقم 88",
        relevant_chunk_ids=["de5762fbe4e15d0160141a14"],
        language="ar",
        note="Law title itself",
    ),
    GroundTruthQuery(
        query="سقف الدين العام",
        relevant_chunk_ids=["31b111c6b8e606d05b1ac7a6"],
        language="ar",
        note="public debt ceiling",
    ),
    GroundTruthQuery(
        query="Alternative Energy Deposit Account",
        relevant_chunk_ids=["da694e852628349a0a377800"],
        language="en",
    ),
    GroundTruthQuery(
        query="الحد الأقصى لدخل الموظفين",
        relevant_chunk_ids=["ea7cef40f447041dd0938898"],
        language="ar",
        note="max employee income cap",
    ),
    GroundTruthQuery(
        query="research projects funded by grants",
        relevant_chunk_ids=["258209735f1a395c5f552cc5"],
        language="en",
    ),
    GroundTruthQuery(
        query="سداد القروض المحلية والأجنبية",
        relevant_chunk_ids=["5b825a103bce5459bd4a6c0d"],
        language="ar",
        note="domestic and foreign loan repayment",
    ),
    GroundTruthQuery(
        query="Ministry of Finance opinion on financial burdens",
        relevant_chunk_ids=["f727fc4b44c7f3674acf8b14"],
        language="en",
    ),
    GroundTruthQuery(
        query="منح الفصل الثاني",
        relevant_chunk_ids=["2a9a18e0632dbb240e35e440"],
        language="ar",
        note="Chapter Two: Grants",
    ),
]
