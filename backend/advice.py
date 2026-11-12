"""Bounded, localized crop guidance. Images and free-form user prompts are never sent."""
import asyncio
import os
import re
import time
from collections import deque
from typing import Literal

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field

class AdviceRequest(BaseModel):
    prediction: Literal['Healthy', 'Common_Rust', 'Gray_Leaf_Spot', 'Northern_Leaf_Blight']
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    language: Literal['en', 'bn'] = 'en'

HEALTHY = {
    'en': 'The model suggests a healthy leaf. Keep checking leaves regularly, maintain good drainage and balanced nutrition, and avoid unnecessary pesticide use. If you notice spreading spots, wilting, or other symptoms, take a clearer photo and consult a local agricultural extension officer.',
    'bn': 'মডেলের ফল অনুযায়ী পাতাটি সুস্থ মনে হচ্ছে। নিয়মিত পাতা পর্যবেক্ষণ করুন, পানি নিষ্কাশন ভালো রাখুন এবং সুষম পুষ্টি নিশ্চিত করুন। অপ্রয়োজনে কীটনাশক ব্যবহার করবেন না। দাগ ছড়িয়ে পড়া, ঢলে পড়া বা অন্য উপসর্গ দেখা গেলে পরিষ্কার ছবি তুলুন এবং স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।',
}
LOW = {'en': 'The model is uncertain; this result does not confirm that the plant is healthy. Retake a clear leaf photo and check symptoms in the field. ', 'bn': 'মডেল এই ফলে অনিশ্চিত; গাছটি সুস্থ বলে নিশ্চিত হওয়া যায় না। পরিষ্কার ছবি তুলে আবার পরীক্ষা করুন এবং মাঠে উপসর্গ মিলিয়ে দেখুন। '}
_cache: dict[tuple, tuple[float, str]] = {}
_calls: deque[float] = deque()
_generation_lock = asyncio.Lock()

async def call_gemini(data: AdviceRequest, key: str, model: str) -> str:
    language = 'Bengali (বাংলা), using Bengali script throughout' if data.language == 'bn' else 'English'
    prompt = (
        f'A maize leaf image classifier predicted {data.prediction.replace("_", " ")}. '
        f'Classifier confidence is {"below 80% (uncertain)" if data.confidence < .8 else "at least 80%"}; '
        'this is not disease severity or a confirmed diagnosis. What should the grower do next? '
        'Give 4-6 concise practical steps, covering field verification, low-risk immediate care, '
        'monitoring, and when to contact a local agricultural extension officer. '
        'Do not invent location, weather, crop stage, or inspection findings. '
        'Do not prescribe pesticide products, dosages, mixtures, or a spray schedule; '
        'chemical decisions require local diagnosis and label guidance. '
        f'Reply only in {language}. Use plain text, short numbered steps, no Markdown headings. '
        'Keep the answer under 220 words. Explicitly mention uncertainty when confidence is low.'
    )
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
            headers={'x-goog-api-key': key},
            json={'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                  'generationConfig': {'maxOutputTokens': 2048, 'temperature': .3}},
        )
        response.raise_for_status()
        candidates = response.json().get('candidates', [])
        if not candidates or candidates[0].get('finishReason') != 'STOP':
            raise ValueError('Incomplete or blocked response')
        parts = candidates[0].get('content', {}).get('parts', [])
        text = '\n'.join(p['text'] for p in parts if isinstance(p.get('text'), str) and not p.get('thought')).strip()
        if not text or len(text) > 12000:
            raise ValueError('Empty or oversized response')
        if data.language == 'bn' and not re.search(r'[\u0980-\u09ff]', text):
            raise ValueError('Wrong response language')
        return text

async def get_advice(data: AdviceRequest):
    if data.prediction == 'Healthy':
        return {'source': 'static', 'language': data.language, 'text': (LOW[data.language] if data.confidence < .8 else '') + HEALTHY[data.language]}
    key = os.getenv('GEMINI_API_KEY', '').strip()
    model = os.getenv('GEMINI_MODEL', 'gemini-3.8-flash').strip()
    if not key or not re.fullmatch(r'[a-zA-Z0-9._-]+', model):
        raise HTTPException(503, 'ai_unavailable')
    cache_key = (model, data.prediction, data.language, data.confidence < .8)
    async with _generation_lock:
        now = time.monotonic()
        cached = _cache.get(cache_key)
        if cached and now - cached[0] < 3600:
            return {'source': 'ai', 'language': data.language, 'text': cached[1]}
        while _calls and now - _calls[0] >= 3600:
            _calls.popleft()
        try:
            limit = max(1, int(os.getenv('AI_MAX_REQUESTS_PER_HOUR', '30')))
        except ValueError:
            limit = 30
        if len(_calls) >= limit:
            raise HTTPException(429, 'ai_rate_limited')
        _calls.append(now)
        try:
            text = await call_gemini(data, key, model)
        except httpx.TimeoutException:
            raise HTTPException(504, 'ai_timeout') from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise HTTPException(503, 'ai_unavailable') from None
        # At most 12 class/language/confidence entries for the configured model.
        if len(_cache) >= 24:
            _cache.clear()
        _cache[cache_key] = (time.monotonic(), text)
        return {'source': 'ai', 'language': data.language, 'text': text}
