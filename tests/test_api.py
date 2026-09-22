"""
Tests de la API.

Todos marcados `integration`: la API necesita PostgreSQL y el modelo
entrenado. Se usa `fastapi.testclient.TestClient`, que no levanta un
servidor real (llama la app directamente en proceso).
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.main import app
from src.data.database import get_engine

pytestmark = pytest.mark.integration

client = TestClient(app)


@pytest.fixture(scope="module")
def existing_customer_id():
    engine = get_engine()
    with engine.connect() as conn:
        return conn.execute(text("SELECT customer_id FROM customers LIMIT 1")).scalar()


@pytest.fixture(autouse=True)
def _clean_predictions_table():
    """Evita que las predicciones de test se acumulen en la tabla real
    entre ejecuciones de la suite."""
    yield
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM predictions WHERE model_version = 'xgboost_v1'"))


VALID_PAYLOAD = {
    "credit_score": 650,
    "geography": "Germany",
    "gender": "Male",
    "age": 55,
    "tenure": 2,
    "balance": 150000.0,
    "num_of_products": 3,
    "has_cr_card": True,
    "is_active_member": False,
    "estimated_salary": 100000.0,
}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_reports_database_and_model_ok(self):
        resp = client.get("/health")
        body = resp.json()
        assert body["database"] is True
        assert body["model_loaded"] is True
        assert body["status"] == "ok"


class TestGetCustomerEndpoint:
    def test_existing_customer_returns_200(self, existing_customer_id):
        resp = client.get(f"/customers/{existing_customer_id}")
        assert resp.status_code == 200
        assert resp.json()["customer_id"] == existing_customer_id

    def test_existing_customer_has_churn_type(self, existing_customer_id):
        resp = client.get(f"/customers/{existing_customer_id}")
        body = resp.json()
        assert body["churn_type"] in {"softchurn", "midchurn", "hardchurn"}

    def test_unknown_customer_returns_404(self):
        resp = client.get("/customers/999999999")
        assert resp.status_code == 404

    def test_non_integer_customer_id_returns_422(self):
        resp = client.get("/customers/not-a-number")
        assert resp.status_code == 422


class TestGetCustomerPredictionEndpoint:
    def test_returns_200_with_valid_probability(self, existing_customer_id):
        resp = client.get(f"/customers/{existing_customer_id}/prediction")
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["customer_id"] == existing_customer_id

    def test_response_has_all_expected_fields(self, existing_customer_id):
        resp = client.get(f"/customers/{existing_customer_id}/prediction")
        body = resp.json()
        expected_fields = {
            "customer_id", "churn_probability", "churn_prediction", "threshold_used",
            "churn_type", "economic_loss", "campaign", "expected_avoided_loss",
            "campaign_cost", "expected_net_profit",
        }
        assert expected_fields.issubset(body.keys())

    def test_unknown_customer_returns_404(self):
        resp = client.get("/customers/999999999/prediction")
        assert resp.status_code == 404

    def test_persists_prediction_to_database(self, existing_customer_id):
        engine = get_engine()
        with engine.connect() as conn:
            count_before = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE customer_id = :cid"),
                {"cid": existing_customer_id},
            ).scalar()

        client.get(f"/customers/{existing_customer_id}/prediction")

        with engine.connect() as conn:
            count_after = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE customer_id = :cid"),
                {"cid": existing_customer_id},
            ).scalar()
        assert count_after == count_before + 1


class TestPredictEndpoint:
    def test_valid_payload_returns_200(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_probability_in_valid_range(self):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        body = resp.json()
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_hypothetical_customer_has_null_customer_id_in_response(self):
        """Sin customer_id en el payload, la respuesta debe reflejarlo
        como None (no inventarse uno)."""
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["customer_id"] is None

    def test_hypothetical_customer_is_not_persisted(self):
        """Sin customer_id, no debe intentar guardar en `predictions`
        (violaria la FK) ni fallar por ello."""
        engine = get_engine()
        with engine.connect() as conn:
            count_before = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()

        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

        with engine.connect() as conn:
            count_after = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
        assert count_after == count_before

    def test_missing_required_field_returns_422(self):
        incomplete_payload = dict(VALID_PAYLOAD)
        del incomplete_payload["credit_score"]
        resp = client.post("/predict", json=incomplete_payload)
        assert resp.status_code == 422

    def test_age_out_of_range_returns_422(self):
        bad_payload = dict(VALID_PAYLOAD, age=200)
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_negative_balance_returns_422(self):
        bad_payload = dict(VALID_PAYLOAD, balance=-100.0)
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422

    def test_high_risk_customer_gets_campaign_true(self):
        """Cliente con perfil de alto riesgo (inactivo, muchos
        productos, mayor) y ChurnType caro (hardchurn por balance/
        salario altos) deberia recibir campaign=True."""
        high_risk_payload = dict(
            VALID_PAYLOAD, age=60, num_of_products=3, is_active_member=False,
            balance=180000.0, estimated_salary=150000.0,
        )
        resp = client.post("/predict", json=high_risk_payload)
        body = resp.json()
        assert body["churn_probability"] > 0.5
        assert body["campaign"] is True

    def test_predict_with_valid_existing_customer_id_persists(self, existing_customer_id):
        payload_with_id = dict(VALID_PAYLOAD, customer_id=existing_customer_id)
        engine = get_engine()
        with engine.connect() as conn:
            count_before = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE customer_id = :cid"),
                {"cid": existing_customer_id},
            ).scalar()

        resp = client.post("/predict", json=payload_with_id)
        assert resp.status_code == 200
        assert resp.json()["customer_id"] == existing_customer_id

        with engine.connect() as conn:
            count_after = conn.execute(
                text("SELECT COUNT(*) FROM predictions WHERE customer_id = :cid"),
                {"cid": existing_customer_id},
            ).scalar()
        assert count_after == count_before + 1