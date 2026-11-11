"""Maize Cure: inference using the supplied timm checkpoints."""
import io
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).parent
MODEL_IDS = ('mobilenetv2_100', 'efficientnet_b0', 'densenet121', 'resnet50')
NAMES = dict(zip(MODEL_IDS, ('MobileNetV2', 'EfficientNet-B0', 'DenseNet121', 'ResNet50')))
MAX_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
app = FastAPI(title='Maize Cure API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(','), allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
@app.middleware('http')
async def reject_large_requests(request, call_next):
    if request.method == 'POST':
        length = request.headers.get('content-length')
        if length:
            try:
                if int(length) > MAX_BYTES + 65536:
                    return JSONResponse({'detail': 'Maximum file size is 10 MB.'}, status_code=413)
            except ValueError:
                return JSONResponse({'detail': 'Invalid Content-Length.'}, status_code=400)
    return await call_next(request)


lock = threading.Lock()
_cached = (None, None)


def read_report(model_id):
    if model_id not in MODEL_IDS:
        raise HTTPException(404, 'Unknown model')
    folder = ROOT / 'data' / model_id
    result = json.loads((folder / 'result.json').read_text())
    return {**result, 'id': model_id, 'name': NAMES[model_id],
            'history': json.loads((folder / 'history.json').read_text()),
            'class_report': json.loads((folder / f'{model_id}_class_report.json').read_text()),
            'class_balance': json.loads((folder / 'class_balance.json').read_text()),
            'weights_available': (ROOT / 'models' / f'{model_id}_best.pth').is_file(),
            'preprocessing': {'size': 224, 'resize': 'bilinear stretch', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'status': 'Assumed ImageNet normalization; original training transform not supplied.'}}


@app.get('/health')
def health():
    available = [m for m in MODEL_IDS if (ROOT / 'models' / f'{m}_best.pth').is_file()]
    return JSONResponse({'status': 'ok' if len(available) == len(MODEL_IDS) else 'not_ready', 'models_available': available}, status_code=200 if len(available) == len(MODEL_IDS) else 503)


@app.get('/api/models')
def models():
    return {'models': [read_report(m) for m in MODEL_IDS], 'comparison_note': 'Reported evaluation scores use different split IDs. Compare cautiously; these are not field accuracy guarantees.'}


@app.get('/api/models/{model_id}')
def model_info(model_id: str):
    return read_report(model_id)


def infer(model_id, raw):
    global _cached
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ('JPEG', 'PNG', 'WEBP'):
                raise HTTPException(415, 'Use a JPEG, PNG, or WebP image.')
            if source.width * source.height > 20_000_000:
                raise HTTPException(413, 'Image exceeds 20 megapixels.')
            image = ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, 'The uploaded file is not a valid image.')
    except Image.DecompressionBombError:
        raise HTTPException(413, 'Image dimensions are too large.')
    report = read_report(model_id)
    path = ROOT / 'models' / f'{model_id}_best.pth'
    if not path.is_file():
        raise HTTPException(503, 'Model weights are not installed yet.')
    import torch
    import timm
    from torchvision import transforms
    torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS', '2')))
    tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])(image).unsqueeze(0)
    # Serialize loading/inference; retain one model to bound memory on small instances.
    with lock:
        started = time.perf_counter()
        if _cached[0] != model_id:
            _cached = (None, None)
            model = timm.create_model(model_id, pretrained=False, num_classes=4)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            model.eval()
            _cached = (model_id, model)
        with torch.inference_mode():
            probs = torch.softmax(_cached[1](tensor), dim=1)[0].tolist()
        elapsed = round((time.perf_counter() - started) * 1000)
    predictions = sorted([{'class_name': label, 'probability': p} for label, p in zip(report['config']['classes'], probs)], key=lambda x: x['probability'], reverse=True)
    return {'model_id': model_id, 'prediction': predictions[0]['class_name'], 'confidence': predictions[0]['probability'], 'probabilities': predictions, 'inference_ms': elapsed, 'preprocessing_status': report['preprocessing']['status'], 'note': 'Scores are model confidence, not certainty. Only the four supported maize leaf classes can be predicted.'}


@app.post('/api/predict/{model_id}')
async def predict(model_id: str, file: UploadFile = File(...)):
    read_report(model_id)
    raw = await file.read(MAX_BYTES + 1)
    await file.close()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, 'Maximum file size is 10 MB.')
    if not raw:
        raise HTTPException(400, 'Choose an image first.')
    return await run_in_threadpool(infer, model_id, raw)
