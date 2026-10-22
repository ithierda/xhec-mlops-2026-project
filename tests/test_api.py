from fastapi.testclient import TestClient

from src.web_service.main import app

client = TestClient(app)

VALID_ABALONE = {
    "sex": "M",
    "length": 0.455,
    "diameter": 0.365,
    "height": 0.095,
    "whole_weight": 0.514,
    "shucked_weight": 0.2245,
    "viscera_weight": 0.101,
    "shell_weight": 0.15,
}


def test_health_check() -> None:
    """The app starts and the home endpoint answers."""
    response = client.get("/")
    assert response.status_code == 200


def test_predict_valid_abalone() -> None:
    """A valid abalone gets a positive number of rings."""
    response = client.post("/predict", json=VALID_ABALONE)
    assert response.status_code == 201
    assert response.json()["rings_prediction"] > 0


def test_predict_rejects_unknown_sex() -> None:
    """A sex other than M, F or I is rejected by the Pydantic validation."""
    response = client.post("/predict", json={**VALID_ABALONE, "sex": "X"})
    assert response.status_code == 422


def test_predict_rejects_negative_measurement() -> None:
    """A negative measurement is rejected by the Pydantic validation."""
    response = client.post("/predict", json={**VALID_ABALONE, "shell_weight": -0.1})
    assert response.status_code == 422
