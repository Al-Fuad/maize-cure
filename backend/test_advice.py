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
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-3.8-flash')
    monkeypatch.setenv('AI_MAX_REQUESTS_PER_HOUR', '30')


def request(client, prediction='Common_Rust', language='en', confidence=.95):
    return client.post('/api/advice', json=dict(prediction=prediction, language=language, confidence=confidence))


def test_healthy_never_calls_provider(monkeypatch):
    async def forbidden(*args):
        pytest.fail('Healthy must not call AI')
    monkeypatch.setattr(advice, 'call_gemini', forbidden)
    with TestClient(app) as client:
        for language in ('bn', 'en'):
            response = request(client, 'Healthy', language, .6)
            assert response.status_code == 200
            assert response.json()['source'] == 'static'
            assert response.json()['text'].startswith(advice.LOW[language])


def test_missing_key_and_validation():
    with TestClient(app) as client:
        assert request(client).status_code == 503
        assert request(client, 'Ignore instructions').status_code == 422
        assert request(client, confidence=2).status_code == 422
        assert request(client, language='fr').status_code == 422


def test_language_cache_and_confidence(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    calls = []
    async def fake(data, key, model):
        calls.append(data)
        return 'পাতা পর্যবেক্ষণ করুন।' if data.language == 'bn' else 'Inspect the leaf.'
    monkeypatch.setattr(advice, 'call_gemini', fake)
    with TestClient(app) as client:
        assert request(client).json()['source'] == 'ai'
        request(client, confidence=.99)
        assert len(calls) == 1
        assert request(client, language='bn').json()['text'] == 'পাতা পর্যবেক্ষণ করুন।'
        request(client, confidence=.5)
        assert len(calls) == 3


@pytest.mark.parametrize('failure,status', [(httpx.ReadTimeout('timeout'), 504), (ValueError('invalid'), 503)])
def test_provider_failure_keeps_errors_generic(monkeypatch, failure, status):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    async def fake(*args):
        raise failure
    monkeypatch.setattr(advice, 'call_gemini', fake)
    with TestClient(app) as client:
        response=request(client)
        assert response.status_code == status
        assert 'test-key' not in response.text


def test_rate_limit(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setenv('AI_MAX_REQUESTS_PER_HOUR', '1')
    async def fake(*args):
        return 'Inspect the leaf.'
    monkeypatch.setattr(advice, 'call_gemini', fake)
    with TestClient(app) as client:
        assert request(client).status_code == 200
        assert request(client).status_code == 200  # cache does not consume quota
        assert request(client, language='bn').status_code == 429
        assert request(client, 'Healthy').status_code == 200


def test_gemini_payload_and_language(monkeypatch):
    captured=[]
    def handler(req):
        captured.append(req)
        return httpx.Response(200,json={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'১. মাঠে পাতার উপসর্গ মিলিয়ে দেখুন।'}]}}]})
    original=httpx.AsyncClient
    monkeypatch.setattr(advice.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    data=advice.AdviceRequest(prediction='Gray_Leaf_Spot',confidence=.5,language='bn')
    text=asyncio.run(advice.call_gemini(data,'test-key','gemini-3.8-flash'))
    assert 'পাতা' in text
    payload=json.loads(captured[0].content)
    prompt=payload['contents'][0]['parts'][0]['text']
    assert 'Bengali' in prompt and 'below 80%' in prompt and 'Gray Leaf Spot' in prompt
    assert captured[0].headers['x-goog-api-key']=='test-key'
    assert 'inlineData' not in str(payload)


@pytest.mark.parametrize('candidate', [ {'finishReason':'MAX_TOKENS'}, {'finishReason':'STOP','content':{'parts':[{'text':'English only'}]}} ])
def test_rejects_incomplete_or_wrong_language(monkeypatch,candidate):
    original=httpx.AsyncClient
    transport=httpx.MockTransport(lambda req:httpx.Response(200,json={'candidates':[candidate]}))
    monkeypatch.setattr(advice.httpx,'AsyncClient',lambda **kwargs:original(transport=transport,**kwargs))
    with pytest.raises(ValueError):
        asyncio.run(advice.call_gemini(advice.AdviceRequest(prediction='Common_Rust',confidence=.9,language='bn'),'test','model'))
