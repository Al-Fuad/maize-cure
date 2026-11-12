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

The runs have different split IDs; reported metrics are not a controlled model ranking. Confidence is a softmax score, not a calibrated probability of correct diagnosis. Out-of-domain images can receive confident predictions. No disease treatment recommendations are generated.

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

Healthy predictions show localized static care advice without any AI request. For the
three disease classes, the browser calls FastAPI `POST /api/advice`; FastAPI calls the
Gemini REST API directly (no Vercel AI SDK). Changing language reloads guidance in that
language. The prediction remains visible if guidance fails, with a retry button.
Only the class and a confidence category (below / at least 80%) reach Gemini; photos
are not sent. Advice is general crop care, not a confirmed field diagnosis.

### Local environment

Copy `backend/.env.example` to `backend/.env` and set:

```dotenv
GEMINI_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-3.8-flash
AI_MAX_REQUESTS_PER_HOUR=30
```

Create the key at https://aistudio.google.com/apikey. Only `GEMINI_API_KEY` is required;
the model and limit shown above are defaults. FastAPI automatically loads `backend/.env`;
process environment values take priority. Restart the backend after editing it.
Keep the key in the backend only, never in a `NEXT_PUBLIC_` variable or in Git.
The existing frontend `.env.local` still only needs `NEXT_PUBLIC_API_URL`.

### Production

In Render → maize-cure-api → Environment, add `GEMINI_API_KEY` and optionally the two
settings above, then save/redeploy. Vercel needs no AI secret; its existing
`NEXT_PUBLIC_API_URL=https://maize-cure-api.onrender.com` remains sufficient.
Without a key, prediction and Healthy advice work; disease guidance reports unavailable.

Guidance is cached for one hour by model, class, language, and confidence category.
The generation quota is shared by all visitors within each backend process and resets
on restart; cache hits and Healthy responses do not consume it. This is not a distributed
billing cap. Set provider-side quotas for your account as needed. Provider calls time out
and return generic errors without exposing credentials. Tests mock the provider: a live
Gemini smoke test requires your valid API key and model access.

```sh
cd backend
.venv/bin/python -m pytest -q
```
