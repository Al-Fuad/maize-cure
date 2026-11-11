import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app import app, MODEL_IDS, MAX_BYTES

client = TestClient(app)

def test_reports_and_class_order():
    data = client.get('/api/models').json()
    assert len(data['models']) == 4
    for model in data['models']:
        assert model['config']['classes'] == ['Common_Rust', 'Gray_Leaf_Spot', 'Healthy', 'Northern_Leaf_Blight']
        assert len(model['history']['train_loss']) == 100

def test_invalid_uploads():
    assert client.post('/api/predict/unknown', files={'file': ('x.jpg', b'bad')}).status_code == 404
    assert client.post('/api/predict/mobilenetv2_100', files={'file': ('x.jpg', b'bad')}).status_code == 400
    assert client.post('/api/predict/mobilenetv2_100', files={'file': ('x.jpg', b'x'*(MAX_BYTES+1))}).status_code == 413
    image = io.BytesIO()
    Image.new('RGB', (20,20)).save(image, format='GIF')
    assert client.post('/api/predict/mobilenetv2_100', files={'file': ('x.gif', image.getvalue())}).status_code == 415

@pytest.mark.parametrize('model', MODEL_IDS)
def test_real_inference(model):
    path = Path(__file__).parent.parent / 'frontend/public/samples/healthy.jpg'
    response = client.post(f'/api/predict/{model}', files={'file': ('healthy.jpg', path.read_bytes(), 'image/jpeg')})
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data['probabilities']) == 4
    assert sum(p['probability'] for p in data['probabilities']) == pytest.approx(1, abs=1e-5)
    assert data['prediction'] == 'Healthy'
