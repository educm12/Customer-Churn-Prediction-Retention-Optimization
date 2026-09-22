"""
Tests de src/business/prediction_service.py.

Marcados `integration` porque `score_customer` necesita el modelo
entrenado (aunque no necesita PostgreSQL: opera sobre un diccionario
de features en memoria, no consulta la base de datos).
"""

import pytest

from src.business.prediction_service import score_customer
from src.config import CAMPAIGN_COST, RETENTION_SUCCESS_PROB

pytestmark = pytest.mark.integration

BASE_FEATURES = {
    "credit_score": 650,
    "geography": "France",
    "gender": "Female",
    "age": 40,
    "tenure": 5,
    "balance": 75000.0,
    "num_of_products": 2,
    "has_cr_card": True,
    "is_active_member": True,
    "estimated_salary": 80000.0,
}


class TestScoreCustomer:
    def test_returns_all_expected_keys(self):
        result = score_customer(BASE_FEATURES)
        expected_keys = {
            "customer_id", "churn_probability", "churn_prediction", "threshold_used",
            "churn_type", "economic_loss", "campaign", "expected_avoided_loss",
            "campaign_cost", "expected_net_profit",
        }
        assert expected_keys == set(result.keys())

    def test_churn_probability_in_valid_range(self):
        result = score_customer(BASE_FEATURES)
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_customer_id_is_passed_through(self):
        result = score_customer(BASE_FEATURES, customer_id=12345)
        assert result["customer_id"] == 12345

    def test_customer_id_defaults_to_none(self):
        result = score_customer(BASE_FEATURES)
        assert result["customer_id"] is None

    def test_churn_type_is_one_of_three_categories(self):
        result = score_customer(BASE_FEATURES)
        assert result["churn_type"] in {"softchurn", "midchurn", "hardchurn"}

    def test_economic_loss_matches_churn_type(self):
        from src.config import ECONOMIC_LOSS

        result = score_customer(BASE_FEATURES)
        assert result["economic_loss"] == ECONOMIC_LOSS[result["churn_type"]]

    def test_campaign_cost_matches_config(self):
        result = score_customer(BASE_FEATURES)
        assert result["campaign_cost"] == CAMPAIGN_COST

    def test_expected_net_profit_formula_is_consistent(self):
        """expected_net_profit debe ser exactamente
        avoided_loss - campaign_cost, con avoided_loss =
        churn_probability * RETENTION_SUCCESS_PROB * economic_loss."""
        result = score_customer(BASE_FEATURES)
        expected_avoided_loss = round(
            result["churn_probability"] * RETENTION_SUCCESS_PROB * result["economic_loss"], 2
        )
        expected_net_profit = round(expected_avoided_loss - CAMPAIGN_COST, 2)
        assert result["expected_avoided_loss"] == pytest.approx(expected_avoided_loss, abs=0.02)
        assert result["expected_net_profit"] == pytest.approx(expected_net_profit, abs=0.02)

    def test_campaign_true_iff_expected_net_profit_positive(self):
        result = score_customer(BASE_FEATURES)
        assert result["campaign"] == (result["expected_net_profit"] > 0)

    def test_churn_prediction_uses_threshold_used(self):
        result = score_customer(BASE_FEATURES)
        expected = int(result["churn_probability"] >= result["threshold_used"])
        assert result["churn_prediction"] == expected

    def test_custom_threshold_is_respected(self):
        result_low = score_customer(BASE_FEATURES, threshold=0.01)
        result_high = score_customer(BASE_FEATURES, threshold=0.99)
        assert result_low["churn_prediction"] == 1
        assert result_high["churn_prediction"] == 0

    def test_high_value_customer_gets_hardchurn(self):
        """Cliente con balance/salario/productos altos y tenure alta ->
        value_score alto -> hardchurn (regla de negocio)."""
        high_value_features = dict(
            BASE_FEATURES, balance=200000.0, estimated_salary=200000.0,
            num_of_products=4, tenure=10,
        )
        result = score_customer(high_value_features)
        assert result["churn_type"] == "hardchurn"

    def test_low_value_customer_gets_softchurn(self):
        low_value_features = dict(
            BASE_FEATURES, balance=0.0, estimated_salary=0.0,
            num_of_products=1, tenure=0,
        )
        result = score_customer(low_value_features)
        assert result["churn_type"] == "softchurn"

    def test_does_not_mutate_input_features_dict(self):
        features_copy = dict(BASE_FEATURES)
        score_customer(features_copy)
        assert features_copy == BASE_FEATURES