"""
Tests de src/business/strategy_comparison.py.

Unitarios con datos sintéticos: no requieren PostgreSQL ni el modelo
entrenado. Se prueban las cuatro estrategias:

    - Campaign Everyone
    - Campaign Nobody
    - XGBoost (business threshold)
    - XGBoost (F1 threshold)
"""

import numpy as np
import pandas as pd
import pytest

from src.business.strategy_comparison import compare_strategies


def make_synthetic_test_set(n=300, seed=0):
    rng = np.random.default_rng(seed)

    y_true = pd.Series(rng.binomial(1, 0.20, n))

    noise = rng.normal(0, 0.2, n)
    churn_probability_model = pd.Series(
        np.clip(
            y_true * 0.6 + 0.15 + noise,
            0.01,
            0.99,
        )
    )

    economic_loss = pd.Series(
        rng.choice([500, 1000, 2000], n).astype(float)
    )

    return y_true, churn_probability_model, economic_loss


def run_comparison(
    y_true,
    proba,
    loss,
    business_threshold=0.3,
    f1_threshold=0.5,
):
    """Helper para ejecutar compare_strategies con ambos thresholds."""
    return compare_strategies(
        y_true,
        proba,
        loss,
        xgboost_threshold_business=business_threshold,
        xgboost_threshold_f1=f1_threshold,
    )


class TestCompareStrategies:

    def test_returns_four_strategies(self):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        assert len(result) == 4

        assert "Campaign Everyone" in result.index
        assert "Campaign Nobody" in result.index

        assert any(
            name.startswith("XGBoost (threshold=")
            for name in result.index
        )

    def test_campaign_everyone_contacts_all(self):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        assert (
            result.loc["Campaign Everyone", "n_contacted"]
            == len(y_true)
        )

    def test_campaign_nobody_contacts_none(self):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        assert (
            result.loc["Campaign Nobody", "n_contacted"]
            == 0
        )

    def test_business_threshold_contacts_correct_customers(self):
        y_true, proba, loss = make_synthetic_test_set()

        threshold = 0.5

        result = run_comparison(
            y_true,
            proba,
            loss,
            business_threshold=threshold,
            f1_threshold=0.7,
        )

        expected_n_contacted = int((proba >= threshold).sum())

        business_rows = [
            name
            for name in result.index
            if f"threshold={threshold:.2f}" in name
        ]

        assert len(business_rows) == 1

        business_row = result.loc[business_rows[0]]

        assert (
            business_row["n_contacted"]
            == expected_n_contacted
        )

    def test_f1_threshold_contacts_correct_customers(self):
        y_true, proba, loss = make_synthetic_test_set()

        threshold = 0.7

        result = run_comparison(
            y_true,
            proba,
            loss,
            business_threshold=0.3,
            f1_threshold=threshold,
        )

        expected_n_contacted = int((proba >= threshold).sum())

        f1_rows = [
            name
            for name in result.index
            if f"threshold={threshold:.2f}" in name
        ]

        assert len(f1_rows) == 1

        f1_row = result.loc[f1_rows[0]]

        assert (
            f1_row["n_contacted"]
            == expected_n_contacted
        )

    def test_higher_threshold_does_not_contact_more_customers(self):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(
            y_true,
            proba,
            loss,
            business_threshold=0.3,
            f1_threshold=0.7,
        )

        business_row = result[
            result.index.str.contains("threshold=0.30")
        ].iloc[0]

        f1_row = result[
            result.index.str.contains("threshold=0.70")
        ].iloc[0]

        assert (
            f1_row["n_contacted"]
            <= business_row["n_contacted"]
        )

    def test_both_xgboost_strategies_contact_fewer_than_everyone(
        self,
    ):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(
            y_true,
            proba,
            loss,
            business_threshold=0.3,
            f1_threshold=0.5,
        )

        everyone_contacted = result.loc[
            "Campaign Everyone",
            "n_contacted",
        ]

        xgboost_rows = result[
            result.index.str.startswith("XGBoost")
        ]

        assert (
            xgboost_rows["n_contacted"]
            < everyone_contacted
        ).all()

    def test_all_strategies_use_actual_outcome_not_predicted_probability(
        self,
    ):
        """
        Punto clave: el valor utilizado como P(Churn) en la fórmula
        económica es el desenlace real (y_true) para las cuatro
        estrategias, no la probabilidad predicha.

        Por tanto, expected_churners_total_portfolio debe coincidir
        con el número real de churners independientemente de la
        estrategia.
        """
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        real_churners = float(y_true.sum())
        predicted_sum = float(proba.sum())

        assert real_churners != pytest.approx(
            predicted_sum
        )

        for strategy in result.index:
            assert (
                result.loc[
                    strategy,
                    "expected_churners_total_portfolio",
                ]
                == pytest.approx(real_churners)
            )

    def test_campaign_nobody_has_zero_net_profit(self):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        assert (
            result.loc[
                "Campaign Nobody",
                "expected_net_profit",
            ]
            == 0.0
        )

    def test_campaign_nobody_has_zero_campaign_cost(self):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        assert (
            result.loc[
                "Campaign Nobody",
                "total_campaign_cost",
            ]
            == 0.0
        )

    def test_campaign_everyone_contacts_all_and_nobody_contacts_none(
        self,
    ):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        assert (
            result.loc[
                "Campaign Everyone",
                "n_contacted",
            ]
            == len(y_true)
        )

        assert (
            result.loc[
                "Campaign Nobody",
                "n_contacted",
            ]
            == 0
        )

    def test_total_economic_impact_is_never_positive_when_churn_exists(
        self,
    ):
        """
        Con clientes que realmente abandonan en el set, el impacto
        económico total no puede ser positivo para ninguna estrategia.
        """
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        assert (
            result["total_economic_impact"] <= 0
        ).all()

    def test_strategy_index_names_are_distinct(self):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        assert result.index.is_unique

    def test_business_and_f1_strategies_are_distinct_when_thresholds_differ(
        self,
    ):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(
            y_true,
            proba,
            loss,
            business_threshold=0.3,
            f1_threshold=0.7,
        )

        xgboost_rows = result[
            result.index.str.startswith("XGBoost")
        ]

        assert len(xgboost_rows) == 2
        assert xgboost_rows.index[0] != xgboost_rows.index[1]

    def test_result_contains_all_expected_metrics(self):
        y_true, proba, loss = make_synthetic_test_set()

        result = run_comparison(y_true, proba, loss)

        expected_columns = {
            "n_total_customers",
            "n_contacted",
            "pct_contacted",
            "total_campaign_cost",
            "expected_churners_total_portfolio",
            "expected_loss_total_portfolio",
            "expected_avoided_loss",
            "expected_realized_loss",
            "expected_retained_customers",
            "expected_net_profit",
            "total_economic_impact",
            "roi_pct",
            "cost_per_retained_customer",
        }

        assert expected_columns.issubset(
            set(result.columns)
        )