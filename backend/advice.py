"""Bounded, localized crop guidance. Images and free-form user prompts are never sent."""
import asyncio
import uuid
if __package__:
    from .research_logs import record, request_id
else:
    from research_logs import record, request_id
import os
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
import re
import time
from collections import deque
from typing import Literal

import httpx
from pydantic import BaseModel, Field, model_validator

Disease = Literal['Healthy', 'Common_Rust', 'Gray_Leaf_Spot', 'Northern_Leaf_Blight']

class ClassScore(BaseModel):
    class_name: Disease
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)

class AdviceRequest(BaseModel):
    prediction: Literal['Healthy', 'Common_Rust', 'Gray_Leaf_Spot', 'Northern_Leaf_Blight']
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    language: Literal['en', 'bn'] = 'en'
    prediction_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')
    probabilities: list[ClassScore] | None = Field(default=None, min_length=4, max_length=4)
    mode: Literal['consensus', 'single', 'rules'] = 'consensus'
    provider: Literal['gemini', 'groq', 'openrouter'] = 'gemini'
    min_confidence: float = Field(default=.8, ge=.5, le=.99, allow_inf_nan=False)
    min_consensus: int = Field(default=2, ge=2, le=3)

    @model_validator(mode='after')
    def consistent_scores(self):
        if self.probabilities is not None:
            scores = self.probabilities
            if len({p.class_name for p in scores}) != 4 or abs(sum(p.probability for p in scores)-1) > .001:
                raise ValueError('Provide all four class scores summing to one')
            top = max(scores, key=lambda p: p.probability)
            if top.class_name != self.prediction or abs(top.probability-self.confidence) > .0001:
                raise ValueError('Prediction must match the highest class score')
        return self


def mixed_infection(data):
    scores = sorted(data.probabilities or [], key=lambda p: p.probability, reverse=True)
    return (len(scores) >= 2 and all(s.class_name != 'Healthy' for s in scores[:2])
            and scores[1].probability >= .18 and scores[0].probability-scores[1].probability <= .18)


def validated(data):
    return data.confidence >= data.min_confidence and not mixed_infection(data)


HEALTHY = {
    'en': 'The model suggests a healthy leaf. Keep checking leaves regularly, maintain good drainage and balanced nutrition, and avoid unnecessary pesticide use. If you notice spreading spots, wilting, or other symptoms, take a clearer photo and consult a local agricultural extension officer.',
    'bn': 'মডেলের ফল অনুযায়ী পাতাটি সুস্থ মনে হচ্ছে। নিয়মিত পাতা পর্যবেক্ষণ করুন, পানি নিষ্কাশন ভালো রাখুন এবং সুষম পুষ্টি নিশ্চিত করুন। অপ্রয়োজনে কীটনাশক ব্যবহার করবেন না। দাগ ছড়িয়ে পড়া, ঢলে পড়া বা অন্য উপসর্গ দেখা গেলে পরিষ্কার ছবি তুলুন এবং স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।',
}
LOW = {'en': 'The model is uncertain; this result does not confirm that the plant is healthy. Retake a clear leaf photo and check symptoms in the field. ', 'bn': 'মডেল এই ফলে অনিশ্চিত; গাছটি সুস্থ বলে নিশ্চিত হওয়া যায় না। পরিষ্কার ছবি তুলে আবার পরীক্ষা করুন এবং মাঠে উপসর্গ মিলিয়ে দেখুন। '}

# Ported from SAMINSADAF/Test01. Only protocol text reaches the grower;
# provider summaries and provider-authored translations are never displayed.
PROTOCOLS = json.loads((Path(__file__).parent / 'data/treatment_protocols.json').read_text())
CATEGORIES = {'chemical': 'chemical_treatment', 'organic': 'organic_options',
              'prevention': 'prevention', 'safety': 'safety'}
LABELS = {'en': ['Chemical', 'Organic', 'Prevention', 'Safety'],
          'bn': ['রাসায়নিক', 'জৈব', 'প্রতিরোধ', 'সুরক্ষা']}
_cache: dict[tuple, tuple[float, dict]] = {}
_calls: deque[float] = deque()
_generation_lock = asyncio.Lock()


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    key: str
    endpoint: str


def configured_providers():
    specs = [
        ('gemini', 'GEMINI_API_KEY', 'GEMINI_MODEL', 'gemini-3.5-flash-lite', ''),
        ('groq', 'GROQ_API_KEY', 'GROQ_MODEL', 'openai/gpt-oss-120b', 'https://api.groq.com/openai/v1/chat/completions'),
        ('openrouter', 'OPENROUTER_API_KEY', 'OPENROUTER_MODEL', 'nvidia/nemotron-3.5-lightning:free', 'https://openrouter.ai/api/v1/chat/completions'),
    ]
    providers, seen = [], set()
    for name, key_env, model_env, default, endpoint in specs:
        key, model = os.getenv(key_env, '').strip(), os.getenv(model_env, default).strip()
        identity = (endpoint, model)
        pattern = r'[a-zA-Z0-9._-]+' if name == 'gemini' else r'[a-zA-Z0-9._:/-]+'
        if key and re.fullmatch(pattern, model) and identity not in seen:
            providers.append(Provider(name, model, key, endpoint))
            seen.add(identity)
    return providers


def allowed_items(data):
    protocol = PROTOCOLS[data.prediction]
    return {category: (list(protocol[field]) if category != 'chemical' or
                      (validated(data) and data.confidence >= .8 and data.prediction != 'Healthy') else [])
            for category, field in CATEGORIES.items()}


def build_prompt(data, allowed):
    return (f'Disease classifier prediction: {data.prediction}. '
            f'Confidence validation: {"passed" if validated(data) else "uncertain"}. '
            'This is not a confirmed field diagnosis or disease severity. '
            'Select up to two exact strings per category from the supplied protocol only. '
            'Do not add treatments or paraphrase. Empty allowed lists must stay empty. '
            'Return JSON only: {"selected":{"chemical":[],"organic":[],"prevention":[],"safety":[]}}. '
            'Allowed protocol: ' + json.dumps(allowed, ensure_ascii=False))


def extract_json(text):
    """Recover fenced/prefixed JSON, smart quotes and trailing commas, without eval."""
    if not isinstance(text, str) or len(text) > 12000:
        return None
    normalized = text.translate(str.maketrans({'“': '"', '”': '"'}))
    for candidate in (text, normalized, re.sub(r',\s*([}\]])', r'\1', normalized)):
        for match in re.finditer(r'\{', candidate):
            try:
                value, _ = json.JSONDecoder().raw_decode(candidate[match.start():])
                if isinstance(value, dict) and isinstance(value.get('selected'), dict):
                    return value
            except json.JSONDecodeError:
                continue
    return None


def sanitize(payload, allowed):
    selected = payload.get('selected', {})
    clean = {}
    for category, items in allowed.items():
        raw = selected.get(category, [])
        # One vote per model per item, regardless of duplicates in its response.
        clean[category] = [item for item in items if isinstance(raw, list) and item in raw][:2]
    return clean


async def call_provider(provider, prompt):
    async with httpx.AsyncClient(timeout=18) as client:
        if provider.name == 'gemini':
            response = await client.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{provider.model}:generateContent',
                headers={'x-goog-api-key': provider.key},
                json={'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                      'generationConfig': {'maxOutputTokens': 2048, 'temperature': .2}},
            )
            response.raise_for_status()
            candidate = response.json()['candidates'][0]
            if candidate.get('finishReason') != 'STOP':
                raise ValueError('Incomplete response')
            text = '\n'.join(part['text'] for part in candidate.get('content', {}).get('parts', [])
                             if isinstance(part.get('text'), str) and not part.get('thought'))
        else:
            response = await client.post(provider.endpoint,
                headers={'Authorization': f'Bearer {provider.key}'},
                json={'model': provider.model, 'messages': [{'role': 'user', 'content': prompt}],
                      'temperature': .2, 'max_tokens': 2048})
            response.raise_for_status()
            choice = response.json()['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Incomplete response')
            text = choice['message']['content']
        if not isinstance(text, str) or not text.strip() or len(text) > 12000:
            raise ValueError('Invalid response')
        return text


def reserve_call():
    now = time.monotonic()
    while _calls and now - _calls[0] >= 3600:
        _calls.popleft()
    try:
        limit = max(0, int(os.getenv('AI_MAX_REQUESTS_PER_HOUR', '30')))
    except ValueError:
        limit = 30
    if len(_calls) >= limit:
        return False
    _calls.append(now)
    return True


async def provider_selection(provider, prompt, allowed):
    for attempt in range(2):
        started = time.perf_counter()
        if not reserve_call():
            record('provider_attempt', provider=provider.name, model=provider.model, attempt=attempt+1, status='quota')
            return None
        try:
            text = await call_provider(provider, prompt + ('\nIMPORTANT: Return only valid JSON with a selected object.' if attempt else ''))
            if os.getenv('RESEARCH_LOG_RAW', '0') == '1':
                redacted = text
                for client in configured_providers():
                    redacted = redacted.replace(client.key, '[REDACTED]')
                redacted = re.sub(r'(?:AIza|gsk_|sk-)[A-Za-z0-9_-]+', '[REDACTED]', redacted)
                record('provider_raw', provider=provider.name, model=provider.model, response=redacted[:12000])
            payload = extract_json(text)
            if payload is not None:
                selected = sanitize(payload, allowed)
                raw = payload['selected']
                discarded = sum(sum(not isinstance(item, str) or item not in allowed[category] for item in raw.get(category, []))
                                for category in allowed if isinstance(raw.get(category), list))
                record('provider_attempt', provider=provider.name, model=provider.model, attempt=attempt+1,
                       status='selected' if any(selected.values()) else 'empty', selected=selected,
                       discarded_items=discarded, latency_ms=round((time.perf_counter()-started)*1000),
                       prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                return (provider, selected) if any(selected.values()) else None
            record('provider_attempt', provider=provider.name, model=provider.model, attempt=attempt+1,
                   status='invalid_json', latency_ms=round((time.perf_counter()-started)*1000))
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            record('provider_attempt', provider=provider.name, model=provider.model, attempt=attempt+1,
                   status='error', error_type=type(error).__name__, latency_ms=round((time.perf_counter()-started)*1000))
            return None
    return None


def render_advice(data, selected, source, models=(), votes=None):
    protocol = PROTOCOLS[data.prediction]
    bn = data.language == 'bn'
    intro = ('একাধিক মডেলের সম্মতিতে প্রোটোকলভিত্তিক পরামর্শ।' if bn else 'Protocol guidance agreed by multiple models.') if source == 'llm_consensus' else ('একটি মডেলের প্রোটোকল নির্বাচন; এটি consensus নয়।' if bn else 'Single-model protocol selection; this is not consensus.') if source == 'llm_single' else ('নিয়মভিত্তিক প্রোটোকল থেকে পরামর্শ।' if bn else 'Guidance from the rule-based protocol.')
    warning = ('পূর্বাভাসে আত্মবিশ্বাস কম। পরিষ্কার ছবি তুলুন এবং রাসায়নিক ব্যবহারের আগে রোগ নিশ্চিত করুন।' if bn else 'Low-confidence prediction. Retake a clear photo and confirm the disease before chemical treatment.') if data.confidence < data.min_confidence else ''
    if mixed_infection(data):
        warning += (' সম্ভাব্য মিশ্র সংক্রমণ বা কাছাকাছি রোগের স্কোর। এটি একাধিক রোগের প্রমাণ নয়; কৃষি কর্মকর্তার মাধ্যমে নিশ্চিত করুন।' if bn else ' Possible mixed infection or ambiguous disease scores. This does not prove multiple diseases; seek field confirmation.')
    confirmation = 'মাঠে উপসর্গ মিলিয়ে স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।' if bn else 'Confirm symptoms in the field with a local agricultural extension officer.'
    bullets = []
    for index, (category, field) in enumerate(CATEGORIES.items()):
        for item in selected[category]:
            translated = protocol[field + '_bn'][protocol[field].index(item)] if bn else item
            bullets.append(f'{LABELS[data.language][index]}: {translated}')
    local = protocol['local_relevance_bn' if bn else 'local_relevance']
    text = '\n\n'.join(part for part in [intro, warning, confirmation, '\n'.join(f'{str(i+1).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")) if bn else i+1}. {s}' for i, s in enumerate(bullets)), local] if part)
    return {'source': source, 'language': data.language, 'text': text,
            'llm_models': list(models), 'consensus_count': data.min_consensus if source == 'llm_consensus' else 0,
            'votes': votes or {}, 'validated': validated(data), 'mixed_infection': mixed_infection(data),
            'mixed_infection_checked': data.probabilities is not None,
            'min_confidence': data.min_confidence, 'min_consensus': data.min_consensus, 'mode': data.mode}


async def _get_advice(data: AdviceRequest):
    if data.prediction == 'Healthy':
        return {'source': 'static', 'language': data.language,
                'text': (LOW[data.language] if data.confidence < data.min_confidence else '') + HEALTHY[data.language],
                'validated': validated(data), 'mixed_infection': False,
                'mixed_infection_checked': data.probabilities is not None,
                'llm_models': [], 'consensus_count': 0, 'votes': {}, 'mode': data.mode}
    allowed = allowed_items(data)
    fallback = render_advice(data, allowed, 'rule_based')
    if data.mode == 'rules':
        return {**fallback, 'fallback_reason': 'rules_selected'}
    providers = configured_providers()
    if data.mode == 'single':
        providers = [p for p in providers if p.name == data.provider]
    required = data.min_consensus if data.mode == 'consensus' else 1
    if len(providers) < required:
        return {**fallback, 'fallback_reason': 'insufficient_models'}
    # Fingerprints invalidate cached votes when credentials change without retaining keys.
    cache_key = (tuple((p.name, p.model, hashlib.sha256(p.key.encode()).hexdigest()) for p in providers),
                 data.prediction, data.language, data.confidence < data.min_confidence, data.confidence < .8,
                 mixed_infection(data), data.probabilities is not None, data.min_confidence, data.min_consensus, data.mode)
    async with _generation_lock:
        cached = _cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 3600:
            return {**cached[1], 'cache_hit': True}
        try:
            async with asyncio.timeout(40):
                results = await asyncio.gather(*(provider_selection(p, build_prompt(data, allowed), allowed) for p in providers))
        except TimeoutError:
            return {**fallback, 'fallback_reason': 'timeout'}
        responses = [r for r in results if r is not None]
        votes = {category: {item: sum(item in selection[category] for _, selection in responses)
                            for item in items} for category, items in allowed.items()}
        consensus = {category: [item for item, count in counts.items() if count >= required]
                     for category, counts in votes.items()}
        if not any(consensus.values()):
            return {**fallback, 'fallback_reason': 'no_agreement_or_unavailable'}
        result = render_advice(data, consensus, 'llm_consensus' if data.mode == 'consensus' else 'llm_single',
                               [f'{p.name}:{p.model}' for p, _ in responses], votes)
        if len(_cache) >= 24:
            _cache.clear()
        _cache[cache_key] = (time.monotonic(), result)
        return result


async def get_advice(data: AdviceRequest):
    trace = uuid.uuid4().hex
    token = request_id.set(trace)
    started = time.perf_counter()
    try:
        result = await _get_advice(data)
        result = {**result, 'request_id': trace, 'cache_hit': result.get('cache_hit', False)}
        record('advisory', prediction_id=data.prediction_id, prediction=data.prediction,
               confidence=data.confidence, language=data.language, mode=data.mode,
               min_confidence=data.min_confidence, min_consensus=data.min_consensus,
               protocol_sha256=hashlib.sha256(json.dumps(PROTOCOLS, sort_keys=True).encode()).hexdigest(),
               latency_ms=round((time.perf_counter()-started)*1000), **result_without_text(result))
        return result
    finally:
        request_id.reset(token)


def result_without_text(result):
    return {key: result.get(key) for key in ('source', 'llm_models', 'consensus_count', 'votes',
            'validated', 'mixed_infection', 'fallback_reason', 'cache_hit')}
