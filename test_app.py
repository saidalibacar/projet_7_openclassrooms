import json
import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_id(client):
    # Test GET request
    response = client.get('/predict_id?id=1&threshold=0.065')
    assert response.status_code == 200
    data = json.loads(response.data.decode('utf-8'))
    assert 'True Prediction' in data
    assert 'Binary Prediction' in data
    assert 'Class' in data

    # Test POST request
    data = {'id': '1', 'threshold': 0.065}
    response = client.post('/predict_id', json=data)
    assert response.status_code == 200
    data = json.loads(response.data.decode('utf-8'))
    assert 'True Prediction' in data
    assert 'Binary Prediction' in data
    assert 'Class' in data

def test_predict_id_invalid_input(client):
    # Test GET request with invalid ID
    response = client.get('/predict_id?id=invalid_id')
    assert response.status_code == 200
    data = json.loads(response.data.decode('utf-8'))
    assert 'error' in data

    # Test POST request with missing ID
    data = {'threshold': 0.065}
    response = client.post('/predict_id', json=data)
    assert response.status_code == 200
    data = json.loads(response.data.decode('utf-8'))
    assert 'error' in data

    # to run test_app 
    # open the terminal and go on file directory where is located app.py and test_app.py 
    # then tap pytest
