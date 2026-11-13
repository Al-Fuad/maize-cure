# Maize Cure

FastAPI + Next.js maize leaf research application. Upload an image, choose one of four timm models, inspect class probabilities, try sample leaves, and explore the original evaluation reports.

## Run locally

Python 3.12 and Node.js 20.9+ required.

```sh
python3.12 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/python backend/download_models.py
cd backend
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

In another terminal:

```sh
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

API documentation: http://localhost:8000/docs

## Model assets

The original `*_best.pth` checkpoints belong in a GitHub release tagged `models-v1`, not in Git history. `backend/models/manifest.json` records exact sizes and SHA-256 hashes. `download_models.py` verifies every download. Set `MODEL_REPOSITORY=owner/repo` for another release repository. Models are loaded strictly with `weights_only=True` and `pretrained=False`; one model is cached at a time.

## Data provenance and limits

Source: https://drive.google.com/drive/folders/1l8QjkxI501vtjfQ3lVE_iqquQbJOlxl7

Reports are from `results/epochs_100_separate_v1`; checkpoints are from the corresponding `models` directory. The four sample images come from `dataset/processed/test`. Original comparison PNGs are preserved in `frontend/public/reports`.

Class order comes from each run's `result.json`: Common Rust, Gray Leaf Spot, Healthy, Northern Leaf Blight. The older dataset_info.json lists five classes, but Southern Leaf Blight is **not supported** by these four-class models. That older file is retained only as source material.

Input processing: EXIF orientation correction, RGB, bilinear resize to 224 × 224, ImageNet mean/std. **The original training transform was not provided; normalization and resize strategy are assumptions.** Validate against the original training code before claiming equivalent evaluation performance. Samples are smoke tests, not a new accuracy evaluation.

The runs have different split IDs; reported metrics are not a controlled model ranking. Confidence is a softmax score, not a calibrated probability of correct diagnosis. Out-of-domain images can receive confident predictions. Disease guidance uses the original app’s constrained protocol and multi-model consensus; it is not a confirmed field diagnosis.

## Deployment

Backend: create a Render Python web service from this GitHub repository; root directory `backend`, Python 3.12, build command `pip install -r requirements.txt && python download_models.py`, start command `uvicorn app:app --host 0.0.0.0 --port $PORT`, health path `/health`. Set `CORS_ORIGINS` to the exact frontend origin. For Linux CPU-only PyTorch installation, use `pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu` before installing requirements. Provision enough RAM for PyTorch plus the largest loaded model; verify runtime memory rather than assuming a free instance is sufficient.

Frontend: import this GitHub repo into Vercel with root directory `frontend`, framework Next.js, and `NEXT_PUBLIC_API_URL` set to the Render HTTPS service URL. This variable is bundled at build time; redeploy after changes.

Uploaded images are processed in memory and not retained by application code. API accepts JPEG, PNG, WebP up to 10 MB and 20 megapixels. CORS is not authentication; use deployment-level rate limits if exposing a high-traffic public API.

## Validation

```sh
backend/.venv/bin/pip install pytest httpx
cd backend && .venv/bin/pytest -q
cd ../frontend && npm run build
```

Tests exercise real weights for all models, report structure, class order, and invalid upload handling.

## Bengali / English and backend AI guidance

The language switch translates navigation, analysis, model information, numeric values,
interactive charts, errors, and care guidance, and remembers the selected language in
this browser. Original research PNGs remain unmodified with their English labels.

Healthy predictions show localized static care advice without an AI request. For the
three disease classes, FastAPI `POST /api/advice` now follows the original
[SAMINSADAF/Test01 advisory engine](https://huggingface.co/spaces/SAMINSADAF/Test01/blob/main/src/advisory_pipeline.py):

- Gemini, Groq GPT-OSS, and OpenRouter Nemotron select exact items from a local protocol.
- Each item needs agreement from the requested two or three distinct configured models. Duplicate selections do not add votes.
- Fenced/prefixed JSON, smart quotes, and trailing commas are recovered. Unparseable output gets one stricter retry.
- Unknown advice is discarded. Displayed Bengali text comes from the matching protocol translation, never unchecked model summaries or translations.
- Chemical-category recommendations are suppressed below 80% classifier confidence, below the selected confidence threshold, or when mixed-infection ambiguity is flagged. This threshold does not establish a field diagnosis.
- Missing credentials, provider errors, exhausted quota, or no consensus return deterministic rule-based guidance. Healthy care stays static.

The bilingual protocol in `backend/data/treatment_protocols.json` is copied from the
[original app](https://huggingface.co/spaces/SAMINSADAF/Test01/blob/main/ground_truth/treatment_protocols.json).
It is inherited research material, not a newly validated agronomic protocol.
The existing frontend contract (`source`, `language`, `text`) is preserved. Additional
fields expose `llm_models`, `consensus_count`, per-item `votes`, and `validated`
(confidence threshold passed and no mixed-infection flag; not a field diagnosis). Sources are `static`, `rule_based`, `llm_single`, or `llm_consensus`.
The frontend sends all four class probabilities. The API validates their sum, unique labels, and consistency with the top prediction. Two disease scores trigger a possible mixed-infection warning when the second score is at least 0.18 and their gap is at most 0.18. This heuristic indicates ambiguity, not confirmed co-infection. No photos are sent to LLM providers.

### Environment

Copy `backend/.env.example` to `backend/.env`. Configure two or three providers:

- `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.5-flash-lite`
- `GROQ_API_KEY`, `GROQ_MODEL=openai/gpt-oss-120b`
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free`

These defaults were checked against the providers' public free-tier/model catalogs on
2026-11-13; free quotas and availability can change. Old `GROQ_LLAMA_MODEL`,
`OPENROUTER_DEEPSEEK_MODEL`, and `OPENROUTER_GLM_MODEL` settings are no longer read.
Update deployed environment variables too; an explicitly set `GEMINI_MODEL` overrides
the new default. Live quality/latency on maize advice has not yet been benchmarked.

### Research controls and report

After a prediction, expand Research controls. Choose consensus (2 or 3 votes), a
single provider for research, or rules only. Set the confidence threshold from 50%
to 99% and press Update guidance. Single-model output is explicitly labeled as such,
never as consensus. Missing providers or failed consensus return local protocol advice.
`GET /api/advice/config` exposes configured provider names/model IDs, never keys.
The UI displays the advice source, validation state, fallback explanation and model votes.

Create advisory report generates a preview and downloadable PNG in the selected
language, with prediction, scores, model, settings, advice source, participating models,
validation/warnings and advice. Browser canvas shapes Bengali using Noto Sans Bengali;
font loading must succeed before a Bengali report is exported. Reports stay in the browser.
Changing the prediction, language or applied controls hides stale report/advice content.

FastAPI loads `backend/.env`; process environment values take priority. Restart after
changes. For production, set these variables on the Render backend and redeploy.
No frontend AI secrets are required. Never put keys in `NEXT_PUBLIC_` variables or Git.

`AI_MAX_REQUESTS_PER_HOUR=30` bounds actual provider attempts, including retries,
per backend process. Set it to `0` for rule-only operation. This is not a distributed
billing cap. Successful consensus is cached for an hour by provider configuration,
class, language, and confidence category. Fallback responses are not cached so a
recovered provider can be retried. Provider work is concurrent with a 40-second
overall deadline after acquiring the generation lock. Raw responses and credentials
are not logged or returned.

### Validation

```sh
backend/.venv/bin/python -m pytest backend/test_advice.py -q
```

Tests cover protocol fallback, exact-match filtering, two-model voting, malformed JSON,
retries, quotas, bilingual output, confidence guardrails, caching, and mocked provider
HTTP contracts. Live provider checks require valid credentials and model access.

## Classification modes, checkpoint validation and research

The website supports single model, four-model soft-voting ensemble, and hybrid cascade
(EfficientNet-B0 first; below the selected threshold, average it with ResNet50).
`POST /api/predict/{model_id}?mode=ensemble` and `?mode=cascade&cascade_threshold=0.8`
return mode, actual models used and a prediction ID as well as existing scores.
Ensemble/cascade accuracy is shown as not evaluated; original single-model reports remain
separate. One model is kept in memory, so switching and multi-model modes include reload cost.

Checkpoint loading verifies the release SHA-256/size, class order, key set, tensor shapes
and finite values. Known `state_dict`/`model_state_dict` containers and uniform `module.` /
`_orig_mod.` prefixes are supported. No mismatched weights or classifier heads are discarded.
A bare state dictionary has no embedded class labels: its verified manifest and run report
provide that mapping. Invalid weights produce an explicit service error rather than random outputs.

Local rotating research logs and reproducible LLM/classification evaluation commands are
in [backend/evaluation/README.md](backend/evaluation/README.md). Real provider results need
API access; credible classification comparisons need a common held-out set and complete
training-image inventories for all four models. The shipped samples only support smoke tests.
