# Retrieval Benchmark Results (Phase 8)

**تاريخ:** 26 سبتمبر 2026
**المستند:** 01_Enacted_Budget_Law_2024_AR.pdf (14 chunks)
**Ground truth:** 10 أسئلة (5 عربي / 5 إنجليزي)، labeled يدوياً

## النتائج

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Latency |
|---|---|---|---|---|---|
| vector_only | 0.6 | 0.9 | 1.0 | 0.753 | 58ms |
| bm25_only | 0.6 | 0.9 | 0.9 | 0.750 | 456ms |
| **hybrid_rrf** | **0.7** | **1.0** | **1.0** | **0.850** | **206ms** |
| hybrid_rerank | 0.8 | 1.0 | 1.0 | 0.900 | 18,480ms |

## القرارات المحسومة بالأرقام

### 1. ترجيح RRF حسب اللغة — لا داعي له
مفيش فرق واضح في الأداء مرتبط بلغة السؤال داخل hybrid_rrf (Recall@5=1.0 عبر
كل اللغات). أضعف حالتين (rr=0.5) موزّعين على اللغتين، مش مرتبطين بلغة معينة.
**القرار: RRF بسيط وموحّد بدون ترجيح إضافي (نفس تصميم Phase 6).**

### 2. الـ Reranker — تحسين حقيقي لكن مكلف جداً للاستخدام التفاعلي
- الفايدة: MRR 0.85→0.90, Recall@1 0.7→0.8
- التكلفة: ×90 latency (206ms → 18.5s) — أبطأ بكتير من أي استخدام تفاعلي مقبول
- حتى بعد rerank، سؤالين فضلوا rr=0.5 (مش صلّح كل حاجة)
**القرار: الـ reranker يفضل متاح في الكود، لكن optional (opt-in) في المسارات
الحساسة للوقت. يتقرر تفعيله انتقائياً في Phase 12 (Complete Pipeline).**

## ملاحظات إضافية
- Hybrid RRF لوحده (من غير rerank) بيدي أفضل توازن جودة/سرعة: Recall@5=1.0،
  MRR=0.85، latency=206ms — أنسب افتراضي للاستخدام التفاعلي.
- BM25 لوحده أبطأ من الـ vector لوحده (456ms مقابل 58ms) بسبب overhead
  الترجمة (نموذج Helsinki-NLP يتحمّل مرة واحدة، لكن inference لكل سؤال
  عربي بياخد وقت).
- عينة الأسئلة صغيرة (10 أسئلة، مستند واحد 14 chunk) — النتائج مؤشرات قوية
  مش أرقام إحصائية نهائية. يستأهل إعادة القياس على corpus أكبر في مراحل لاحقة.

---

# Follow-up — clean 8-document corpus and Arabic retrieval diagnosis

The table-extraction correction reduced the corpus to 772 chunks. On the 35-query ground truth, the BM25 retrieval check recorded hit@5 of 29/35 for Arabic and 34/35 for English. The broader retrieval diagnosis found Arabic hit@5 of 23/35 for vector-only and 30/35 for hybrid RRF; English hit@5 was 31/35 for both vector-only and hybrid. Per-query ranks and misses are preserved in `data/benchmarks/phase9b_retrieval_check.md` and `data/benchmarks/phase9b_retrieval_diagnosis.md`.

An Arabic-only union-and-rerank experiment raised hit@5 to 33/35 when combining reranked candidates with fused results, compared with 32/35 for the fused baseline. Both reached 34/35 at hit@8; question 94 is an extraction miss and excluded from the comparable denominator. The reranker averaged 48.9 seconds per query on the measured CPU setup, so this one-question hit@5 gain is not selected as the interactive default. Full ranks are in `data/benchmarks/phase9b_union_rerank.md`.

These are retrieval measurements on fact-seeking questions and should not be generalized to broad semantic or summarization queries without adding those query types to the evaluation set.
