"""Cache candidate retrieval and compare evidence-view reranking with a saved baseline."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['cache','score'],required=True)
    parser.add_argument('--input',default='data/benchmarks/mof35_latest/results.json')
    parser.add_argument('--cache',default='data/evaluation/mof35_candidates.json')
    parser.add_argument('--output',default='data/evaluation/mof35_focused.json')
    parser.add_argument('--ids',default='')
    parser.add_argument('--window',type=int,default=100)
    parser.add_argument('--candidates',type=int,default=15)
    args=parser.parse_args()
    from rag_system.retrieval.models import RetrievedChunk
    from scripts.evaluate_mof_workbook import _gold_match
    if args.stage=='cache':
        from rag_system.retrieval.hybrid_retriever import HybridRetriever
        from rag_system.retrieval.retrieval_pipeline import _prepare_rerank_query
        hybrid=HybridRetriever()
        source=json.loads(Path(args.input).read_text())
        rows=[]
        for number,case in enumerate(source['results'],1):
            candidates=hybrid.retrieve(case['query'],top_k=args.candidates)
            query=_prepare_rerank_query(case['query'],candidates,hybrid.bm25_retriever.query_translator)
            rows.append({'case':{k:v for k,v in case.items() if k not in ('retrieved_chunks','top_sources')},
                'rerank_query':query,'candidates':[c.to_dict() for c in candidates]})
            Path(args.cache).write_text(json.dumps({'candidate_count':args.candidates,'rows':rows},ensure_ascii=False,indent=2))
            print(f'Cached {number}/{len(source["results"])}',flush=True)
        return
    from rag_system.reranking.default import get_default_reranker
    from rag_system.reranking.evidence_view import evidence_views
    reranker=get_default_reranker()
    cache=json.loads(Path(args.cache).read_text())
    wanted=set(args.ids.split(',')) if args.ids else set()
    rows=[r for r in cache['rows'] if not wanted or r['case']['id'] in wanted]
    results=[]
    for number,row in enumerate(rows,1):
        case=row['case'];chunks=[RetrievedChunk(**c) for c in row['candidates']]
        views=evidence_views(row['rerank_query'],chunks,args.window)
        started=time.perf_counter()
        scores=reranker.model.predict([(row['rerank_query'],v) for v in views],batch_size=8)
        ordered=sorted(zip(chunks,scores,views),key=lambda item:float(item[1]),reverse=True)
        retrieved=[]
        for rank,(c,score,view) in enumerate(ordered[:5],1):
            gold=_gold_match(case['gold_chunk'],c.text)
            retrieved.append({'rank':rank,'chunk_id':c.chunk_id,'source_file':c.metadata.get('source_file'),
                'metadata':c.metadata,'text':c.text,'score':float(score),'gold_match':gold,
                'expected_source_match':gold and c.metadata.get('source_file')==case['source_file'],'rerank_view':view})
        hits=[c['rank'] for c in retrieved if c['expected_source_match']]
        any_hits=[c['rank'] for c in retrieved if c['gold_match']]
        result={**case,'retrieved_chunks':retrieved,'baseline_rank':case['hit_rank_expected_source'],
                'hit_rank_expected_source':min(hits) if hits else None,'hit_rank_any_source':min(any_hits) if any_hits else None,
                'rerank_query':row['rerank_query'],'elapsed_seconds':round(time.perf_counter()-started,3)}
        results.append(result)
        summary={'mode':'retrieval','retriever':'reranked','top_k':5,'test_cases_run':len(results),'window_words':args.window}
        for kind in ('any_source','expected_source'):
            for k in (1,5): summary[f'gold_chunk_hit_at_{k}_{kind}']=sum(r[f'hit_rank_{kind}'] is not None and r[f'hit_rank_{kind}']<=k for r in results)/len(results)
        Path(args.output).write_text(json.dumps({'summary':summary,'results':results},ensure_ascii=False,indent=2))
        print(f'[{number}/{len(rows)}] {case["id"]} {case["language"]}: {case["hit_rank_expected_source"]} -> {result["hit_rank_expected_source"]}',flush=True)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
