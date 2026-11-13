"""Reproducible protocol-selection benchmark, not an agronomic correctness score."""
import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
import advice


def cases():
    for disease in advice.PROTOCOLS:
        for language in ('en', 'bn'):
            for confidence in (.95, .6):
                yield advice.AdviceRequest(prediction=disease, confidence=confidence, language=language)
    yield advice.AdviceRequest(prediction='Common_Rust', confidence=.46, probabilities=[
        {'class_name': name, 'probability': score} for name, score in
        [('Common_Rust', .46), ('Gray_Leaf_Spot', .40), ('Healthy', .04), ('Northern_Leaf_Blight', .10)]])


def summarize(rows, attempts):
    n = len(rows)
    consensus = [r for r in rows if r['mode']=='consensus' and r['prediction']!='Healthy']
    dynamic = [r for r in rows if r['mode']!='rules' and r['prediction']!='Healthy']
    return {'runs': n, 'median_latency_ms': statistics.median(r['latency_ms'] for r in rows) if n else None,
            'consensus_success_rate': sum(r['source']=='llm_consensus' for r in consensus)/len(consensus) if consensus else None,
            'fallback_rate': sum(r['source']=='rule_based' for r in dynamic)/len(dynamic) if dynamic else None,
            'provider_attempts': len(attempts),
            'parse_success_rate': sum(a['status'] in ('selected','empty') for a in attempts)/len(attempts) if attempts else None,
            'discarded_items': sum(a.get('discarded_items',0) for a in attempts),
            'retry_attempts': sum(a['attempt']==2 for a in attempts),
            'agronomic_correctness': 'Not measured; complete expert-review.csv separately.'}


async def run(args):
    providers = advice.configured_providers()
    if args.live and len(providers)<2:
        raise SystemExit('Live consensus comparison requires at least two configured providers.')
    output=Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['RESEARCH_LOG_ENABLED']='1'
    os.environ['RESEARCH_LOG_PATH']=str(output/'audit.jsonl')
    modes=[('rules','gemini')]
    if args.live:
        modes += [('single',p.name) for p in providers]+[('consensus','gemini')]
    rows=[]
    for repeat in range(args.repeats):
        for case in cases():
            for mode, provider in modes:
                advice._cache.clear()  # every observation is an uncached run
                data=case.model_copy(update={'mode':mode,'provider':provider})
                started=time.perf_counter()
                result=await advice.get_advice(data)
                rows.append({'repeat':repeat,'case':data.model_dump(),'prediction':data.prediction,
                             'mode':mode,'provider':provider,'latency_ms':round((time.perf_counter()-started)*1000),**result})
                print(f"repeat={repeat+1} case={data.prediction}/{data.language}/{data.confidence} mode={mode}/{provider} source={result['source']} latency_ms={rows[-1]['latency_ms']}", flush=True)
                with (output/'responses.jsonl').open('a') as f:
                    f.write(json.dumps(rows[-1],ensure_ascii=False)+'\n')
    events=[json.loads(line) for line in (output/'audit.jsonl').read_text().splitlines()]
    attempts=[e for e in events if e['event']=='provider_attempt']
    summary=summarize(rows,attempts)
    summary.update({'kind':'live_protocol_benchmark' if args.live else 'offline_rules_smoke',
                    'python':platform.python_version(),'repeats':args.repeats,
                    'models':{p.name:p.model for p in providers},
                    'protocol_sha256':hashlib.sha256(json.dumps(advice.PROTOCOLS,sort_keys=True).encode()).hexdigest(),
                    'code_sha256':hashlib.sha256((ROOT/'advice.py').read_bytes()).hexdigest(),
                    'hourly_attempt_budget':os.getenv('AI_MAX_REQUESTS_PER_HOUR','30'),
                    'by_mode':{mode:summarize([r for r in rows if r['mode']==mode],[a for a in attempts if a['request_id'] in {r['request_id'] for r in rows if r['mode']==mode}]) for mode in {r['mode'] for r in rows}},
                    'by_provider':{p.name:summarize([r for r in rows if r['mode']=='single' and r['provider']==p.name], [a for a in attempts if a['provider']==p.name]) for p in providers}})
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    with (output/'expert-review.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['request_id','disease','language','source','advice','correctness_0_to_4','safety_0_to_4','local_relevance_0_to_4','reviewer','notes'])
        for row in rows:writer.writerow([row['request_id'],row['prediction'],row['language'],row['source'],row['text'],'','','','',''])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='Calls configured APIs; quotas/charges may apply.')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not 1<=args.repeats<=20:parser.error('repeats must be 1..20')
    asyncio.run(run(args))
