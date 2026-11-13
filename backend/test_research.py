import asyncio
import json
from pathlib import Path
import pytest
import torch
from checkpoints import normalize_state, validate_state, CheckpointError
from app import classify, MODEL_IDS, app
from fastapi.testclient import TestClient
import advice
import research_logs
from evaluation.classification import metrics, load_dataset


def test_checkpoint_container_prefix_and_shape():
    model=torch.nn.Linear(2,4)
    original=model.state_dict()
    wrapped={'state_dict':{'module.'+k:v for k,v in original.items()}}
    clean=normalize_state(wrapped)
    validate_state(model,clean)
    with pytest.raises(CheckpointError):normalize_state({'classes':['wrong'],'state_dict':original})
    with pytest.raises(CheckpointError):validate_state(model,{'weight':torch.zeros(1000,2),'bias':torch.zeros(1000)})
    with pytest.raises(CheckpointError):validate_state(model,{'weight':original['weight']})
    with pytest.raises(CheckpointError):validate_state(model,{**original,'weight':torch.full((4,2),float('nan'))})


def test_modes_and_cascade_boundary():
    calls=[]
    def confident(name):calls.append(name);return [.8,.1,.05,.05]
    assert classify(confident,'resnet50','cascade',.8)==[.8,.1,.05,.05]
    assert calls==['efficientnet_b0']
    calls.clear()
    def uncertain(name):
        calls.append(name)
        return [.4,.3,.2,.1] if name=='efficientnet_b0' else [.2,.5,.2,.1]
    assert classify(uncertain,'resnet50','cascade',.8)==pytest.approx([.3,.4,.2,.1])
    assert calls==['efficientnet_b0','resnet50']
    calls.clear()
    assert sum(classify(uncertain,'resnet50','ensemble',.8))==pytest.approx(1)
    assert calls==list(MODEL_IDS)


@pytest.mark.parametrize('mode',['ensemble','cascade'])
def test_real_modes(mode):
    raw=(Path(__file__).parents[1]/'frontend/public/samples/healthy.jpg').read_bytes()
    with TestClient(app) as client:
        response=client.post(f'/api/predict/efficientnet_b0?mode={mode}',files={'file':('leaf.jpg',raw,'image/jpeg')})
        assert response.status_code==200,response.text
        result=response.json()
        assert result['model_id']==mode and result['prediction']=='Healthy'
        assert sum(p['probability'] for p in result['probabilities'])==pytest.approx(1)
        assert len(result['models_used'])==(4 if mode=='ensemble' else 1)
        assert client.post('/api/predict/efficientnet_b0?mode=unknown',files={'file':('x.jpg',raw)}).status_code==422


def test_logs_are_correlated_and_do_not_store_keys(tmp_path,monkeypatch):
    monkeypatch.setenv('RESEARCH_LOG_PATH',str(tmp_path/'audit.jsonl'))
    monkeypatch.setenv('RESEARCH_LOG_ENABLED','1')
    monkeypatch.setenv('GEMINI_API_KEY','secret-test-key')
    monkeypatch.setenv('RESEARCH_LOG_RAW','1')
    monkeypatch.setenv('AI_MAX_REQUESTS_PER_HOUR','30')
    advice._calls.clear();advice._cache.clear()
    monkeypatch.setattr(advice,'_generation_lock',asyncio.Lock())
    async def fake(*args):return json.dumps({'selected':{'organic':[advice.PROTOCOLS['Common_Rust']['organic_options'][0],'secret-test-key']}})
    monkeypatch.setattr(advice,'call_provider',fake)
    result=asyncio.run(advice.get_advice(advice.AdviceRequest(prediction='Common_Rust',confidence=.9,mode='single')))
    text=(tmp_path/'audit.jsonl').read_text()
    assert 'secret-test-key' not in text
    events=[json.loads(line) for line in text.splitlines()]
    assert all(e['request_id']==result['request_id'] for e in events)
    attempt=next(e for e in events if e['event']=='provider_attempt')
    assert attempt['discarded_items']==1
    assert next(e for e in events if e['event']=='advisory')['source']=='llm_single'


def test_metrics_known_confusion():
    rows=[{'label':'Healthy','prediction':'Healthy','inference_ms':10},
          {'label':'Common_Rust','prediction':'Healthy','inference_ms':20}]
    result=metrics(rows)
    assert result['accuracy']==.5
    assert result['per_class']['Healthy']['precision']==.5
    assert result['confusion_matrix'][0][2]==1
    assert result['median_latency_ms']==15


def test_dataset_requires_exclusions_and_rejects_overlap(tmp_path):
    import hashlib
    image=tmp_path/'leaf.jpg';image.write_bytes(b'example')
    digest=hashlib.sha256(image.read_bytes()).hexdigest()
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps([{'path':'leaf.jpg','label':'Healthy','sha256':digest}]))
    with pytest.raises(ValueError):load_dataset(manifest)
    assert len(load_dataset(manifest,smoke=True))==1
    training=tmp_path/'training.json';training.write_text(json.dumps({m:[digest] for m in MODEL_IDS}))
    with pytest.raises(ValueError,match='overlap'):load_dataset(manifest,training)


def test_checkpoint_manifest_integrity(tmp_path):
    import hashlib
    from checkpoints import load_checked, CLASSES
    model=torch.nn.Linear(2,4)
    path=tmp_path/'model.pth';torch.save(model.state_dict(),path)
    (tmp_path/'manifest.json').write_text(json.dumps({'files':[{'filename':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}))
    assert load_checked(model,path,CLASSES)
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(CheckpointError,match='checksum'):load_checked(model,path,CLASSES)


def test_no_observations_are_not_reported_as_success():
    from evaluation.llm import summarize
    result=summarize([],[])
    assert result['consensus_success_rate'] is None
    assert result['fallback_rate'] is None
    assert result['parse_success_rate'] is None
