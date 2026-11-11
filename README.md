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
