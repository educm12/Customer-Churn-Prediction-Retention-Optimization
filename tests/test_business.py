"""
Tests unitarios de la regla de negocio de ChurnType.

No requieren PostgreSQL (a diferencia de tests/test_database.py):
operan sobre DataFrames en memoria, para poder correr en cualquier
entorno (`pytest -m "not integration"`).
"""

import numpy as np
import pandas as pd
import pytest

from src.business.campaign import (
    CAMPAIGN_COST,
    RETENTION_SUCCESS_PROB,
    campaign_everyone,
    campaign_nobody,
    compare_baselines,
    compute_expected_avoided_loss,
    compute_expected_net_profit,
    simulate_campaign_strategy,
)
from src.business.churn_impact import (
    ECONOMIC_LOSS,
    add_churn_type_columns,
    classify_churn_type,
    compute_value_score,
)


def make_customer(**overrides) -> pd.DataFrame:
    """Crea un DataFrame de un unico cliente 'neutro' (banda 0.50 en
    todo), sobreescribiendo los campos indicados. Util para aislar el
    efecto de una sola variable en cada test."""
    base = {
        "Balance": 75000.0,
        "EstimatedSalary": 75000.0,
        "NumOfProducts": 2,
        "Tenure": 3,
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestClassifyChurnType:
    def test_below_soft_threshold(self):
        assert classify_churn_type(0.39) == "softchurn"

    def test_at_soft_threshold_boundary_is_midchurn(self):
        # value_score < 0.4 -> softchurn; == 0.4 ya cae en midchurn.
        assert classify_churn_type(0.40) == "midchurn"

    def test_inside_mid_band(self):
        assert classify_churn_type(0.50) == "midchurn"

    def test_at_hard_threshold_boundary_is_midchurn(self):
        # <= 0.6 -> midchurn; el corte a hardchurn es estrictamente > 0.6.
        assert classify_churn_type(0.60) == "midchurn"

    def test_above_hard_threshold(self):
        assert classify_churn_type(0.61) == "hardchurn"


class TestComputeValueScore:
    def test_lowest_band_in_every_variable_gives_minimum_score(self):
        df = make_customer(Balance=0, EstimatedSalary=0, NumOfProducts=1, Tenure=0)
        score = compute_value_score(df)[0]
        # Todas las variables en banda 0.25 -> value_score = 0.25 exacto.
        assert score == pytest.approx(0.25)

    def test_highest_band_in_every_variable_gives_maximum_score(self):
        df = make_customer(Balance=200000, EstimatedSalary=200000, NumOfProducts=4, Tenure=10)
        score = compute_value_score(df)[0]
        # Todas las variables en banda 1.0 -> value_score = 1.0 exacto.
        assert score == pytest.approx(1.0)

    def test_balance_band_thresholds(self):
        # Solo Balance cambia; el resto se mantiene en banda 0.25 (Tenure=3
        # cae en banda 0.50 de Tenure segun la regla <=5, asi que fijamos
        # explicitamente valores minimos para aislar el efecto de Balance).
        low = make_customer(Balance=49999, EstimatedSalary=0, NumOfProducts=1, Tenure=0)
        mid = make_customer(Balance=50000, EstimatedSalary=0, NumOfProducts=1, Tenure=0)
        assert compute_value_score(low)[0] < compute_value_score(mid)[0]

    def test_does_not_require_exited_column(self):
        """compute_value_score no debe fallar ni usar `Exited`, aunque
        la columna exista en el DataFrame (no debe leerla)."""
        df = make_customer()
        df["Exited"] = 1
        score_with_exited = compute_value_score(df)[0]

        df_no_exited = make_customer()
        score_without_exited = compute_value_score(df_no_exited)[0]

        assert score_with_exited == score_without_exited


class TestAddChurnTypeColumns:
    def test_adds_expected_columns(self):
        df = make_customer()
        result = add_churn_type_columns(df)
        assert {"value_score", "ChurnType", "EconomicLoss"}.issubset(result.columns)

    def test_does_not_mutate_input_dataframe(self):
        df = make_customer()
        original_columns = list(df.columns)
        add_churn_type_columns(df)
        assert list(df.columns) == original_columns

    def test_economic_loss_matches_churn_type_mapping(self):
        df = pd.concat(
            [
                make_customer(Balance=0, EstimatedSalary=0, NumOfProducts=1, Tenure=0),
                make_customer(Balance=200000, EstimatedSalary=200000, NumOfProducts=4, Tenure=10),
            ],
            ignore_index=True,
        )
        result = add_churn_type_columns(df)
        expected = result["ChurnType"].map(ECONOMIC_LOSS)
        assert (result["EconomicLoss"] == expected).all()

    def test_churn_type_never_derived_from_exited(self):
        """Dos clientes identicos salvo por Exited deben recibir el
        mismo ChurnType: la columna Exited debe ser irrelevante."""
        churner = make_customer()
        churner["Exited"] = 1
        stayer = make_customer()
        stayer["Exited"] = 0

        result_churner = add_churn_type_columns(churner)
        result_stayer = add_churn_type_columns(stayer)

        assert result_churner["ChurnType"].iloc[0] == result_stayer["ChurnType"].iloc[0]

    def test_realistic_customer_distribution_is_reasonable(self):
        """Con un mix variado de clientes, las tres categorias deben
        poder aparecer (smoke test, no exhaustivo)."""
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame(
            {
                "Balance": rng.uniform(0, 250000, n),
                "EstimatedSalary": rng.uniform(0, 200000, n),
                "NumOfProducts": rng.integers(1, 5, n),
                "Tenure": rng.integers(0, 11, n),
            }
        )
        result = add_churn_type_columns(df)
        assert set(result["ChurnType"].unique()) <= {"softchurn", "midchurn", "hardchurn"}
        assert result["ChurnType"].nunique() > 1


# ============================================================
# Tests de src/business/campaign.py (Baselines)
# ============================================================
class TestExpectedValueFormula:
    def test_expected_avoided_loss_formula(self):
        # P(Churn)=1.0, retention=0.40, L=1000 -> avoided loss = 400
        result = compute_expected_avoided_loss(
            pd.Series([1.0]), pd.Series([1000.0]), retention_prob=0.40
        )
        assert result.iloc[0] == pytest.approx(400.0)

    def test_expected_avoided_loss_is_zero_if_churn_probability_is_zero(self):
        result = compute_expected_avoided_loss(pd.Series([0.0]), pd.Series([2000.0]))
        assert result.iloc[0] == 0.0

    def test_expected_net_profit_subtracts_campaign_cost(self):
        avoided_loss = pd.Series([400.0])
        result = compute_expected_net_profit(avoided_loss, campaign_cost=50.0)
        assert result.iloc[0] == pytest.approx(350.0)

    def test_low_churn_probability_yields_negative_net_profit(self):
        """Propiedad clave de la formula: un cliente con riesgo muy bajo
        debe dar beneficio esperado negativo (-campaign_cost aprox),
        penalizando contactarlo sin reglas adicionales."""
        avoided_loss = compute_expected_avoided_loss(
            pd.Series([0.01]), pd.Series([2000.0]), retention_prob=0.40
        )
        net_profit = compute_expected_net_profit(avoided_loss, campaign_cost=CAMPAIGN_COST)
        assert net_profit.iloc[0] < 0

    def test_high_churn_probability_and_high_loss_yields_positive_net_profit(self):
        avoided_loss = compute_expected_avoided_loss(
            pd.Series([0.9]), pd.Series([2000.0]), retention_prob=RETENTION_SUCCESS_PROB
        )
        net_profit = compute_expected_net_profit(avoided_loss, campaign_cost=CAMPAIGN_COST)
        assert net_profit.iloc[0] > 0


class TestBaselineStrategies:
    def test_campaign_everyone_contacts_all(self):
        decision = campaign_everyone(100)
        assert decision.sum() == 100
        assert decision.all()

    def test_campaign_nobody_contacts_none(self):
        decision = campaign_nobody(100)
        assert decision.sum() == 0
        assert not decision.any()


class TestSimulateCampaignStrategy:
    def test_campaign_nobody_has_zero_cost_and_zero_profit(self):
        churn_prob = pd.Series([0.8, 0.1, 0.5])
        loss = pd.Series([2000.0, 500.0, 1000.0])
        decision = campaign_nobody(3)

        result = simulate_campaign_strategy(churn_prob, decision, loss)
        assert result["n_contacted"] == 0
        assert result["total_campaign_cost"] == 0.0
        assert result["expected_avoided_loss"] == 0.0
        assert result["expected_net_profit"] == 0.0
        assert result["roi_pct"] == 0.0

    def test_campaign_nobody_realized_loss_equals_full_portfolio_loss(self):
        """Si no se contacta a nadie, TODA la perdida esperada de la
        cartera se materializa: expected_realized_loss debe coincidir
        con expected_loss_total_portfolio, y no con 0."""
        churn_prob = pd.Series([0.8, 0.1, 0.5])
        loss = pd.Series([2000.0, 500.0, 1000.0])
        decision = campaign_nobody(3)

        result = simulate_campaign_strategy(churn_prob, decision, loss)
        assert result["expected_realized_loss"] == pytest.approx(result["expected_loss_total_portfolio"])

    def test_campaign_nobody_total_economic_impact_is_not_zero(self):
        """'No hacer nada' NO es gratis: el banco pierde dinero por los
        clientes que abandonan igualmente. total_economic_impact debe
        ser fuertemente negativo, no 0."""
        churn_prob = pd.Series([0.8, 0.1, 0.5])
        loss = pd.Series([2000.0, 500.0, 1000.0])
        decision = campaign_nobody(3)

        result = simulate_campaign_strategy(churn_prob, decision, loss)
        expected_loss = float((churn_prob * loss).sum())
        assert result["total_economic_impact"] == pytest.approx(-expected_loss)
        assert result["total_economic_impact"] < 0

    def test_campaign_everyone_cost_matches_n_times_campaign_cost(self):
        churn_prob = pd.Series([0.8, 0.1, 0.5])
        loss = pd.Series([2000.0, 500.0, 1000.0])
        decision = campaign_everyone(3)

        result = simulate_campaign_strategy(churn_prob, decision, loss, campaign_cost=50.0)
        assert result["total_campaign_cost"] == pytest.approx(150.0)

    def test_realized_loss_plus_avoided_loss_equals_portfolio_loss(self):
        """Relacion exacta: lo que se evita + lo que se pierde = la
        perdida total de la cartera sin intervencion."""
        churn_prob = pd.Series([0.8, 0.1, 0.5])
        loss = pd.Series([2000.0, 500.0, 1000.0])
        decision = campaign_everyone(3)

        result = simulate_campaign_strategy(churn_prob, decision, loss)
        assert result["expected_avoided_loss"] + result["expected_realized_loss"] == pytest.approx(
            result["expected_loss_total_portfolio"]
        )

    def test_total_economic_impact_difference_matches_net_profit(self):
        """La diferencia de total_economic_impact entre dos estrategias
        sobre la MISMA cartera debe coincidir con la diferencia de sus
        expected_net_profit (ambas metricas deben ser consistentes)."""
        churn_prob = pd.Series([0.8, 0.1, 0.5, 0.9])
        loss = pd.Series([2000.0, 500.0, 1000.0, 2000.0])

        everyone = simulate_campaign_strategy(churn_prob, campaign_everyone(4), loss)
        nobody = simulate_campaign_strategy(churn_prob, campaign_nobody(4), loss)

        impact_diff = everyone["total_economic_impact"] - nobody["total_economic_impact"]
        profit_diff = everyone["expected_net_profit"] - nobody["expected_net_profit"]
        assert impact_diff == pytest.approx(profit_diff)

    def test_uncontacted_customers_do_not_contribute_to_avoided_loss(self):
        """Aunque un cliente tenga alta probabilidad de churn, si
        `campaign_decision` es False, no debe contar en avoided loss."""
        churn_prob = pd.Series([0.95])
        loss = pd.Series([2000.0])
        decision = pd.Series([False])

        result = simulate_campaign_strategy(churn_prob, decision, loss)
        assert result["expected_avoided_loss"] == 0.0

    def test_expected_churners_total_is_independent_of_campaign_decision(self):
        """`expected_churners_total_portfolio` describe la cartera
        completa, no cambia segun a quien se contacte."""
        churn_prob = pd.Series([0.8, 0.1, 0.5])
        loss = pd.Series([2000.0, 500.0, 1000.0])

        result_everyone = simulate_campaign_strategy(churn_prob, campaign_everyone(3), loss)
        result_nobody = simulate_campaign_strategy(churn_prob, campaign_nobody(3), loss)

        assert (
            result_everyone["expected_churners_total_portfolio"]
            == result_nobody["expected_churners_total_portfolio"]
        )

    def test_cost_per_retained_customer_is_none_when_no_one_retained(self):
        churn_prob = pd.Series([0.0, 0.0])
        loss = pd.Series([1000.0, 1000.0])
        decision = campaign_everyone(2)

        result = simulate_campaign_strategy(churn_prob, decision, loss)
        assert result["expected_retained_customers"] == 0.0
        assert result["cost_per_retained_customer"] is None


class TestCompareBaselines:
    def test_compare_baselines_returns_both_strategies(self):
        df = pd.DataFrame(
            {
                "exited": [1, 0, 1, 0, 1],
                "economic_loss": [2000.0, 500.0, 1000.0, 500.0, 2000.0],
            }
        )
        result = compare_baselines(df)
        assert set(result.index) == {"Campaign Everyone", "Campaign Nobody"}

    def test_compare_baselines_everyone_contacts_all_rows(self):
        df = pd.DataFrame(
            {
                "exited": [1, 0, 1, 0, 1],
                "economic_loss": [2000.0, 500.0, 1000.0, 500.0, 2000.0],
            }
        )
        result = compare_baselines(df)
        assert result.loc["Campaign Everyone", "n_contacted"] == 5
        assert result.loc["Campaign Nobody", "n_contacted"] == 0

    def test_compare_baselines_expected_churners_matches_actual_exited_count(self):
        df = pd.DataFrame(
            {
                "exited": [1, 0, 1, 0, 1],
                "economic_loss": [2000.0, 500.0, 1000.0, 500.0, 2000.0],
            }
        )
        result = compare_baselines(df)
        assert result.loc["Campaign Everyone", "expected_churners_total_portfolio"] == 3
