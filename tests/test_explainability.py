"""
Tests de src/models/explainability.py.

Todos marcados como `integration`: SHAP y feature importance necesitan
el modelo ya entrenado (models/xgboost_model.pkl). No hay una
version "unitaria sin modelo" razonable para este modulo, ya que su
proposito entero es explicar ESE modelo concreto.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
import shap

from src.models.explainability import (
    build_shap_explainer,
    compute_shap_explanation,
    get_feature_importance,
    plot_feature_importance,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.models.predict import load_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def sample_customers(model):
    """Un puñado de clientes sinteticos (con las columnas que el
    pipeline espera, sin churn_type porque el modelo final se entreno
    sin ella) para probar explainability sin depender de PostgreSQL."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame(
        {
            "credit_score": rng.integers(400, 850, n),
            "age": rng.integers(18, 80, n),
            "tenure": rng.integers(0, 10, n),
            "balance": rng.uniform(0, 200000, n),
            "num_of_products": rng.integers(1, 4, n),
            "has_cr_card": rng.integers(0, 2, n).astype(bool),
            "is_active_member": rng.integers(0, 2, n).astype(bool),
            "estimated_salary": rng.uniform(0, 200000, n),
            "geography": rng.choice(["France", "Germany", "Spain"], n),
            "gender": rng.choice(["Male", "Female"], n),
        }
    )


class TestGetFeatureImportance:
    def test_returns_dataframe_with_expected_columns(self, model):
        result = get_feature_importance(model)
        assert {"feature", "importance"}.issubset(result.columns)

    def test_is_sorted_descending_by_importance(self, model):
        result = get_feature_importance(model)
        assert list(result["importance"]) == sorted(result["importance"], reverse=True)

    def test_does_not_include_churn_type(self, model):
        """El modelo final se entreno sin churn_type: no debe
        aparecer en la feature importance."""
        result = get_feature_importance(model)
        assert not result["feature"].str.contains("churn_type").any()

    def test_all_importances_are_non_negative(self, model):
        result = get_feature_importance(model)
        assert (result["importance"] >= 0).all()


class TestShapExplanation:
    def test_build_shap_explainer_returns_tree_explainer(self, model):
        explainer = build_shap_explainer(model)
        assert isinstance(explainer, shap.TreeExplainer)

    def test_compute_shap_explanation_has_one_row_per_customer(self, model, sample_customers):
        explanation = compute_shap_explanation(sample_customers, model=model)
        assert len(explanation.values) == len(sample_customers)

    def test_compute_shap_explanation_has_feature_names(self, model, sample_customers):
        explanation = compute_shap_explanation(sample_customers, model=model)
        assert explanation.feature_names is not None
        assert len(explanation.feature_names) == explanation.values.shape[1]

    def test_shap_values_sum_close_to_prediction_minus_base(self, model, sample_customers):
        """Propiedad fundamental de SHAP: base_value + suma(shap_values)
        debe aproximar la salida cruda (margen) del modelo para cada
        cliente."""
        import numpy as np

        explanation = compute_shap_explanation(sample_customers, model=model)
        X_transformed = model.named_steps["preprocessing"].transform(sample_customers)
        raw_margin = model.named_steps["model"].predict(X_transformed, output_margin=True)

        reconstructed = explanation.base_values + explanation.values.sum(axis=1)
        assert np.allclose(reconstructed, raw_margin, atol=1e-3)


class TestExplainabilityPlots:
    def test_plot_feature_importance_creates_file(self, model, tmp_path):
        importance_df = get_feature_importance(model)
        path = tmp_path / "feature_importance.png"
        plot_feature_importance(importance_df, str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_plot_shap_summary_creates_file(self, model, sample_customers, tmp_path):
        explanation = compute_shap_explanation(sample_customers, model=model)
        path = tmp_path / "shap_summary.png"
        plot_shap_summary(explanation, str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_plot_shap_waterfall_creates_file(self, model, sample_customers, tmp_path):
        explanation = compute_shap_explanation(sample_customers, model=model)
        path = tmp_path / "shap_waterfall.png"
        plot_shap_waterfall(explanation, index=0, path=str(path))
        assert path.exists()
        assert path.stat().st_size > 0
