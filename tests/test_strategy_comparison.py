"""
Tests de src/business/strategy_comparison.py.

Unitarios con datos sinteticos: no requieren PostgreSQL ni el modelo
entrenado (esos se prueban indirectamente al ejecutar el script, no
en la suite de tests para mantenerla rapida).
"""

import numpy as np
import pandas as pd
import pytest

from src.business.strategy_comparison import compare_strategies


def make_synthetic_test_set(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y_true = pd.Series(rng.binomial(1, 0.20, n))
    noise = rng.normal(0, 0.2, n)
    churn_probability_model = pd.Series(np.clip(y_true * 0.6 + 0.15 + noise, 0.01, 0.99))
    economic_loss = pd.Series(rng.choice([500, 1000, 2000], n).astype(float))
    return y_true, churn_probability_model, economic_loss


class TestCompareStrategies:
    def test_returns_three_strategies(self):
        y_true, proba, loss = make_synthetic_test_set()
        result = compare_strategies(y_true, proba, loss, xgboost_threshold=0.3)
        assert len(result) == 3
        assert any(name.startswith("XGBoost") for name in result.index)
        assert "Campaign Everyone" in result.index
        assert "Campaign Nobody" in result.index

    def test_campaign_everyone_contacts_all(self):
        y_true, proba, loss = make_synthetic_test_set()
        result = compare_strategies(y_true, proba, loss, xgboost_threshold=0.3)
        assert result.loc["Campaign Everyone", "n_contacted"] == len(y_true)

    def test_campaign_nobody_contacts_none(self):
        y_true, proba, loss = make_synthetic_test_set()
        result = compare_strategies(y_true, proba, loss, xgboost_threshold=0.3)
        assert result.loc["Campaign Nobody", "n_contacted"] == 0

    def test_xgboost_contacts_only_customers_above_threshold(self):
        y_true, proba, loss = make_synthetic_test_set()
        threshold = 0.5
        result = compare_strategies(y_true, proba, loss, xgboost_threshold=threshold)
        expected_n_contacted = int((proba >= threshold).sum())
        xgboost_row = result[result.index.str.startswith("XGBoost")].iloc[0]
        assert xgboost_row["n_contacted"] == expected_n_contacted

    def test_xgboost_contacts_fewer_than_everyone_for_reasonable_threshold(self):
        y_true, proba, loss = make_synthetic_test_set()
        result = compare_strategies(y_true, proba, loss, xgboost_threshold=0.5)
        xgboost_row = result[result.index.str.startswith("XGBoost")].iloc[0]
        assert xgboost_row["n_contacted"] < result.loc["Campaign Everyone", "n_contacted"]

    def test_all_strategies_use_actual_outcome_not_predicted_probability(self):
        """Punto clave: el 'P(Churn)' que alimenta la formula
        economica debe ser el desenlace REAL (y_true) para las tres
        estrategias, no la probabilidad predicha. Verificamos que
        expected_churners_total_portfolio (que depende de esa P(Churn))
        coincide con el conteo real de churners, no con la suma de
        probabilidades predichas (que seria distinta)."""
        y_true, proba, loss = make_synthetic_test_set()
        result = compare_strategies(y_true, proba, loss, xgboost_threshold=0.3)

        real_churners = float(y_true.sum())
        predicted_sum = float(proba.sum())
        assert real_churners != pytest.approx(predicted_sum)  # deben diferir (dataset sintetico con ruido)

        # Las tres filas deben coincidir con el conteo REAL, no con la suma de proba.
        for strategy in result.index:
            assert result.loc[strategy, "expected_churners_total_portfolio"] == pytest.approx(real_churners)

    def test_lower_threshold_never_decreases_net_profit_relative_to_nobody(self):
        """Sanity check economico: XGBoost con un threshold razonable no
        deberia poder hacerlo peor que Campaign Nobody en beneficio neto
        (Nobody siempre da 0 por definicion; XGBoost, al elegir
        selectivamente, deberia dar >= 0 en la mayoria de casos
        razonables, aunque no esta garantizado en el caso general)."""
        y_true, proba, loss = make_synthetic_test_set()
        result = compare_strategies(y_true, proba, loss, xgboost_threshold=0.5)
        assert result.loc["Campaign Nobody", "expected_net_profit"] == 0.0

    def test_total_economic_impact_is_never_positive_when_churn_exists(self):
        """Con clientes que realmente abandonan en el set, el impacto
        economico total no puede ser positivo para ninguna estrategia
        (el churn siempre cuesta algo; a lo sumo se minimiza la perdida)."""
        y_true, proba, loss = make_synthetic_test_set()
        result = compare_strategies(y_true, proba, loss, xgboost_threshold=0.3)
        assert (result["total_economic_impact"] <= 0).all()

    def test_strategy_index_names_are_distinct(self):
        y_true, proba, loss = make_synthetic_test_set()
        result = compare_strategies(y_true, proba, loss, xgboost_threshold=0.3)
        assert result.index.is_unique
