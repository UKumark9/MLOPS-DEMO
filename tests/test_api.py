from fastapi.testclient import TestClient
import sys; sys.path.insert(0, ".")
from src.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_valid():
    payload = {"sepal_length":5.1,"sepal_width":3.5,
               "petal_length":1.4,"petal_width":0.2}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()
    assert "label" in r.json()

def test_predict_invalid():
    r = client.post("/predict", json={"sepal_length": -1})
    assert r.status_code == 422