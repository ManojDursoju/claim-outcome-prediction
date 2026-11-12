
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_CLAIM = {
    "payer": "Aetna",
    "provider": "Dr.Rao",
    "cpt_code": "99213",
    "diagnosis_code": "M54.5",
    "billed_amount": 220.00,
    "patient_age": 45
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    response = client.post("/predict", json=VALID_CLAIM)

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_status"] in [
        "Paid",
        "Partially_Paid",
        "Denied"
    ]

    assert set(result["probabilities"]) == {
        "Paid",
        "Partially_Paid",
        "Denied"
    }

    assert abs(sum(result["probabilities"].values()) - 1) < 0.001


def test_invalid_billed_amount():
    claim = VALID_CLAIM.copy()
    claim["billed_amount"] = "invalid"

    response = client.post("/predict", json=claim)

    assert response.status_code == 400


def test_missing_required_field():
    claim = VALID_CLAIM.copy()
    del claim["payer"]

    response = client.post("/predict", json=claim)

    assert response.status_code == 400
