import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from app import app
import advice


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    advice._cache.clear()
    advice._calls.clear()
    monkeypatch.setattr(advice, '_generation_lock', asyncio.Lock())
    for key in ('GEMINI_API_KEY', 'GROQ_API_KEY', 'OPENROUTER_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('AI_MAX_REQUESTS_PER_HOUR', '30')


def request(client, prediction='Common_Rust', language='en', confidence=.95):
    return client.post('/api/advice', json=dict(prediction=prediction, language=language, confidence=confidence))


def enable(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini')
    monkeypatch.setenv('GROQ_API_KEY', 'test-groq')


def payload(prediction='Common_Rust'):
    return {'selected': {cat: advice.PROTOCOLS[prediction][field] for cat, field in advice.CATEGORIES.items()}}


def test_healthy_and_missing_keys(monkeypatch):
    async def forbidden(*args):
        pytest.fail('Must not call a provider')
    monkeypatch.setattr(advice, 'call_provider', forbidden)
    with TestClient(app) as client:
        for language in ('bn', 'en'):
            healthy = request(client, 'Healthy', language, .6).json()
            assert healthy['source'] == 'static'
            assert healthy['text'].startswith(advice.LOW[language])
            result = request(client, language=language).json()
            assert result['source'] == 'rule_based'
            assert result['language'] == language and result['text']
        assert request(client, 'Ignore instructions').status_code == 422
        assert request(client, confidence=2).status_code == 422
        assert request(client, language='fr').status_code == 422


def test_consensus_cache_language_and_guardrails(monkeypatch):
    enable(monkeypatch)
    calls = []
    async def fake(provider, prompt):
        calls.append(prompt)
        result = payload()
        result['summary_en'] = 'Invented chemical dose'
        result['selected_bn'] = {'chemical': ['Invented translation']}
        return json.dumps(result)
    monkeypatch.setattr(advice, 'call_provider', fake)
    with TestClient(app) as client:
        result = request(client).json()
        assert result['source'] == 'llm_consensus' and result['consensus_count'] == 2
        assert len(result['llm_models']) == 2
        request(client, confidence=.99)
        assert len(calls) == 2
        bn = request(client, language='bn').json()['text']
        assert advice.PROTOCOLS['Common_Rust']['organic_options_bn'][0] in bn
        assert 'Invented' not in bn and 'Invented' not in result['text']
        low = request(client, confidence=.5).json()
        assert low['validated'] is False and 'Low-confidence' in low['text']
        assert 'Chemical:' not in low['text']
        assert all('photo' not in prompt for prompt in calls)


@pytest.mark.parametrize('text', [
    'prefix ```json\n{"selected":{"organic":["a"]}}\n``` suffix',
    '{“selected”:{“organic”:[“a”,],},}',
    '{"selected":{"organic":["a } { brace"]}}',
])
def test_json_recovery(text):
    assert advice.extract_json(text)['selected']['organic']


@pytest.mark.parametrize('text', ['not JSON', '[]', '{"selected":[]}', 'x' * 12001])
def test_json_rejection(text):
    assert advice.extract_json(text) is None


def test_retry_and_exact_match_filter(monkeypatch):
    enable(monkeypatch)
    calls = {}
    async def fake(provider, prompt):
        calls[provider.name] = calls.get(provider.name, 0) + 1
        if calls[provider.name] == 1:
            return 'invalid JSON'
        assert 'IMPORTANT' in prompt
        result = payload()
        result['selected']['chemical'] = ['invented pesticide']
        return json.dumps(result)
    monkeypatch.setattr(advice, 'call_provider', fake)
    with TestClient(app) as client:
        result = request(client).json()
        assert result['source'] == 'llm_consensus'
        assert 'invented' not in result['text']
        assert 'Chemical:' not in result['text']
        assert list(calls.values()) == [2, 2]


@pytest.mark.parametrize('failure', [httpx.ReadTimeout('secret-key'), ValueError('secret-key'), KeyError('secret-key')])
def test_failure_falls_back_without_secrets(monkeypatch, failure):
    enable(monkeypatch)
    async def fake(*args):
        raise failure
    monkeypatch.setattr(advice, 'call_provider', fake)
    with TestClient(app) as client:
        response = request(client)
        assert response.status_code == 200
        assert response.json()['source'] == 'rule_based'
        assert 'secret-key' not in response.text


def test_one_valid_vote_or_disagreement_is_not_consensus(monkeypatch):
    enable(monkeypatch)
    async def fake(provider, prompt):
        item = advice.PROTOCOLS['Common_Rust']['organic_options'][0 if provider.name == 'gemini' else 1]
        return json.dumps({'selected': {'organic': [item, item, item]}})
    monkeypatch.setattr(advice, 'call_provider', fake)
    with TestClient(app) as client:
        assert request(client).json()['source'] == 'rule_based'
    monkeypatch.delenv('GROQ_API_KEY')
    with TestClient(app) as client:
        assert request(client).json()['source'] == 'rule_based'


def test_quota_includes_retries_and_fallback(monkeypatch):
    enable(monkeypatch)
    monkeypatch.setenv('AI_MAX_REQUESTS_PER_HOUR', '2')
    calls = []
    async def fake(*args):
        calls.append(1)
        return 'malformed'
    monkeypatch.setattr(advice, 'call_provider', fake)
    with TestClient(app) as client:
        assert request(client).json()['source'] == 'rule_based'
        assert request(client, language='bn').json()['source'] == 'rule_based'
        assert len(calls) == 2


def test_all_classes_fallback_in_both_languages():
    with TestClient(app) as client:
        for disease in advice.PROTOCOLS:
            for language in ('en', 'bn'):
                result = request(client, disease, language, .5)
                assert result.status_code == 200
                assert result.json()['text']
                assert 'Chemical:' not in result.json()['text']
                assert 'রাসায়নিক:' not in result.json()['text']


def test_openrouter_uses_one_current_model(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test')
    monkeypatch.delenv('OPENROUTER_MODEL', raising=False)
    providers=advice.configured_providers()
    assert len(providers)==1
    assert providers[0].model=='nvidia/nemotron-3.5-lightning:free'


@pytest.mark.parametrize('name', ['gemini', 'groq', 'openrouter'])
def test_provider_request_payloads(monkeypatch, name):
    captured = []
    def handler(req):
        captured.append(req)
        return httpx.Response(200, json=(
            {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(payload())}]}}]}
            if name == 'gemini' else {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(payload())}}]}
        ))
    original = httpx.AsyncClient
    monkeypatch.setattr(advice.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    provider = advice.Provider(name, 'model', 'test-key', 'https://example.com/chat/completions')
    text = asyncio.run(advice.call_provider(provider, 'protocol only'))
    assert advice.extract_json(text)
    body = json.loads(captured[0].content)
    if name == 'gemini':
        assert captured[0].headers['x-goog-api-key'] == 'test-key'
        assert body['contents'][0]['parts'][0]['text'] == 'protocol only'
    else:
        assert captured[0].headers['authorization'] == 'Bearer test-key'
        assert body['messages'][0]['content'] == 'protocol only'


def test_truncated_provider_response_rejected(monkeypatch):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={'choices': [{'finish_reason': 'length', 'message': {'content': json.dumps(payload())}}]}))
    monkeypatch.setattr(advice.httpx, 'AsyncClient', lambda **kwargs: original(transport=transport, **kwargs))
    with pytest.raises(ValueError):
        asyncio.run(advice.call_provider(advice.Provider('groq', 'model', 'key', 'https://example.com'), 'prompt'))


def test_mixed_infection_and_score_validation():
    scores=[{'class_name':name,'probability':score} for name,score in
            [('Common_Rust',.46),('Gray_Leaf_Spot',.40),('Healthy',.04),('Northern_Leaf_Blight',.10)]]
    with TestClient(app) as client:
        body=dict(prediction='Common_Rust',confidence=.46,probabilities=scores,mode='rules')
        result=client.post('/api/advice',json=body).json()
        assert result['mixed_infection'] and not result['validated']
        assert 'Chemical:' not in result['text']
        assert 'not prove multiple diseases' in result['text']
        body['confidence']=.95
        assert client.post('/api/advice',json=body).status_code==422
        body['confidence']=.46
        body['probabilities'][2]['class_name']='Common_Rust'
        assert client.post('/api/advice',json=body).status_code==422


def test_controls_and_chemical_floor(monkeypatch):
    enable(monkeypatch)
    calls=[]
    async def fake(provider,prompt):
        calls.append(provider.name)
        return json.dumps(payload())
    monkeypatch.setattr(advice,'call_provider',fake)
    with TestClient(app) as client:
        body=dict(prediction='Common_Rust',confidence=.7,min_confidence=.6,mode='single',provider='groq')
        result=client.post('/api/advice',json=body).json()
        assert result['source']=='llm_single' and result['consensus_count']==0
        assert result['validated'] and 'Chemical:' not in result['text']
        assert calls==['groq']
        body.update(mode='consensus',min_consensus=3)
        assert client.post('/api/advice',json=body).json()['fallback_reason']=='insufficient_models'
        body.update(mode='rules')
        assert client.post('/api/advice',json=body).json()['fallback_reason']=='rules_selected'
        body['min_consensus']=1
        assert client.post('/api/advice',json=body).status_code==422
        body.update(min_consensus=2,min_confidence=1.1)
        assert client.post('/api/advice',json=body).status_code==422


def test_three_votes_and_cache_settings(monkeypatch):
    enable(monkeypatch)
    monkeypatch.setenv('OPENROUTER_API_KEY','test-router')
    async def fake(provider,prompt):
        result=payload()
        if provider.name=='openrouter':result['selected']['chemical']=[]
        return json.dumps(result)
    monkeypatch.setattr(advice,'call_provider',fake)
    with TestClient(app) as client:
        body=dict(prediction='Common_Rust',confidence=.9,min_consensus=2)
        two=client.post('/api/advice',json=body).json()
        assert 'Chemical:' in two['text']
        body['min_consensus']=3
        three=client.post('/api/advice',json=body).json()
        assert three['source']=='llm_consensus' and three['consensus_count']==3
        assert 'Chemical:' not in three['text']
        body['min_confidence']=.95
        assert not client.post('/api/advice',json=body).json()['validated']


def test_config_and_healthy_metadata(monkeypatch):
    enable(monkeypatch)
    monkeypatch.delenv('GEMINI_MODEL',raising=False)
    monkeypatch.delenv('GROQ_MODEL',raising=False)
    with TestClient(app) as client:
        response=client.get('/api/advice/config')
        assert 'test-gemini' not in response.text
        providers={p['id']:p for p in response.json()['providers']}
        assert providers['gemini']['model']=='gemini-3.5-flash-lite'
        assert providers['groq']['model']=='openai/gpt-oss-120b'
        assert not providers['openrouter']['configured']
        result=request(client,'Healthy').json()
        assert result['validated'] and result['llm_models']==[]
