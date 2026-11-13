"""Compare all modes on one labeled dataset; enforce exact-hash training exclusions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import infer, MODEL_IDS
from checkpoints import CLASSES


def metrics(rows):
    matrix=[[0]*4 for _ in CLASSES]
    for row in rows:matrix[CLASSES.index(row['label'])][CLASSES.index(row['prediction'])]+=1
    per_class={}
    for i,label in enumerate(CLASSES):
        tp=matrix[i][i]; support=sum(matrix[i]); predicted=sum(row[i] for row in matrix)
        precision=tp/predicted if predicted else 0
        recall=tp/support if support else 0
        per_class[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'support':support}
    return {'n':len(rows),'accuracy':sum(matrix[i][i] for i in range(4))/len(rows),
            'macro_f1':sum(v['f1'] for v in per_class.values())/4,'confusion_matrix':matrix,
            'class_order':CLASSES,'per_class':per_class,
            'median_latency_ms':statistics.median(r['inference_ms'] for r in rows)}


def load_dataset(path, training_path=None, smoke=False):
    manifest=json.loads(Path(path).read_text())
    excluded=set()
    if not smoke:
        if not training_path:raise ValueError('Supply --training-hashes for every model, or use --smoke (not a benchmark).')
        training=json.loads(Path(training_path).read_text())
        if set(training)!=set(MODEL_IDS) or any(not isinstance(v,list) or not v for v in training.values()):
            raise ValueError('Training hashes must contain nonempty lists for all four models')
        excluded={h for values in training.values() for h in values}
    if not isinstance(manifest,list) or not manifest:raise ValueError('Dataset must be a nonempty array')
    seen=set(); records=[]
    for item in manifest:
        file=(Path(path).resolve().parent/item['path']).resolve()
        raw=file.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
        if item['label'] not in CLASSES or digest!=item['sha256']:raise ValueError('Unknown label or image checksum mismatch')
        if digest in seen or digest in excluded:raise ValueError('Duplicate image or exact overlap with training')
        seen.add(digest);records.append((item,raw))
    if not smoke and {item['label'] for item,_ in records}!=set(CLASSES):raise ValueError('Benchmark must cover all four classes')
    return records


def run(args):
    dataset=load_dataset(args.manifest,args.training_hashes,args.smoke)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for item,raw in dataset:
        for name,mode in [(m,'single') for m in MODEL_IDS]+[('ensemble','ensemble'),('cascade','cascade')]:
            result=infer(name if mode=='single' else 'efficientnet_b0',raw,mode,args.threshold)
            row={'configuration':name,'label':item['label'],'image_sha256':item['sha256'],**result}
            rows.append(row)
            with (output/'predictions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    summary={'kind':'smoke_not_accuracy_evidence' if args.smoke else 'common_set_benchmark',
             'dataset_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
             'training_hashes_sha256':hashlib.sha256(Path(args.training_hashes).read_bytes()).hexdigest() if args.training_hashes else None,
             'checkpoints':json.loads((ROOT/'models/manifest.json').read_text()),
             'code_sha256':hashlib.sha256((ROOT/'app.py').read_bytes()).hexdigest(),
             'cascade_threshold':args.threshold,'latency_note':'Sequential execution with one cached model; includes model-load time.',
             'limits':'Hash checks detect exact overlap only; researcher must exclude related images and verify training inventories and preprocessing.',
             'results':{name:metrics([r for r in rows if r['configuration']==name]) for name in [*MODEL_IDS,'ensemble','cascade']}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--training-hashes')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--threshold',type=float,default=.8)
    args=parser.parse_args()
    if not .5<=args.threshold<=.99:parser.error('threshold must be .5..99')
    run(args)
