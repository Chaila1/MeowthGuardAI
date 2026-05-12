import pytest
import io 
from PIL import Image
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_PokeScanEndpoint(client):
    image  = Image.new('RGB', (244,244), color='blue')
    imgBytes = io.BytesIO()
    image.save(imgBytes, format='JPEG')
    imgBytes.seek(0)

    data = {
        'file': (imgBytes, 'dummy_card.jpg')
    }

    respon = client.post('/pokeScan/', data=data, content_type='multipart/form-data')

    assert respon.status_code == 200

    Json = respon.get_json()

    assert 'prediction' in Json
    assert 'confidenceScore' in Json

    assert Json['prediction'] in ['real', 'fake']