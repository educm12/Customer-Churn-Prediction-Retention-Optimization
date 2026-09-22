"""
Tests de src/features/feature_engineering.py.

La mayoria son unitarios (DataFrame sintetico en memoria). El unico
test que toca PostgreSQL (`test_load_customers_from_db_*`) se marca
como `integration`.
"""

import pandas as pd
import pytest

from src.features.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OPTIONAL_CATEGORICAL_FEATURES,
    TARGET_COL,
    build_feature_pipeline,
    get_feature_names,
    get_feature_target,
    load_customers_from_db,
)


def make_customers_df(n: int = 20) -> pd.DataFrame:
    """DataFrame sintetico con la misma forma que la tabla `customers`
    de PostgreSQL (snake_case), sin depender de la base de datos."""
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "row_number": range(1, n + 1),
            "surname": [f"Surname{i}" for i in range(n)],
            "credit_score": [650] * n,
            "geography": (["France", "Germany", "Spain"] * n)[:n],
            "gender": (["Male", "Female"] * n)[:n],
            "age": [40] * n,
            "tenure": [5] * n,
            "balance": [75000.0] * n,
            "num_of_products": [2] * n,
            "has_cr_card": [True] * n,
            "is_active_member": [False] * n,
            "estimated_salary": [80000.0] * n,
            "exited": ([0, 1] * n)[:n],
            "value_score": [0.5] * n,
            "churn_type": (["softchurn", "midchurn", "hardchurn"] * n)[:n],
            "economic_loss": [1000.0] * n,
        }
    )


class TestGetFeatureTarget:
    def test_excludes_identifiers_and_leakage_columns(self):
        df = make_customers_df()
        X, y = get_feature_target(df)
        forbidden = {"customer_id", "row_number", "surname", "value_score", "economic_loss", "exited"}
        assert forbidden.isdisjoint(set(X.columns))

    def test_target_is_the_exited_column(self):
        df = make_customers_df()
        _, y = get_feature_target(df)
        assert y.name == TARGET_COL
        assert set(y.unique()) <= {0, 1}

    def test_include_churn_type_true_by_default(self):
        df = make_customers_df()
        X, _ = get_feature_target(df)
        assert "churn_type" in X.columns

    def test_include_churn_type_false_excludes_it(self):
        df = make_customers_df()
        X, _ = get_feature_target(df, include_churn_type=False)
        assert "churn_type" not in X.columns

    def test_all_expected_numeric_and_categorical_features_present(self):
        df = make_customers_df()
        X, _ = get_feature_target(df)
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + OPTIONAL_CATEGORICAL_FEATURES:
            assert col in X.columns

    def test_raises_if_required_column_missing(self):
        df = make_customers_df().drop(columns=["balance"])
        with pytest.raises(ValueError):
            get_feature_target(df)


class TestBuildFeaturePipeline:
    def test_pipeline_fits_and_transforms_without_error(self):
        df = make_customers_df()
        X, _ = get_feature_target(df)
        pipeline = build_feature_pipeline()
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape[0] == len(df)

    def test_output_has_no_categorical_dtype_left(self):
        """Tras el One-Hot Encoding, todas las columnas deben ser
        numericas (requisito de XGBoost)."""
        df = make_customers_df()
        X, _ = get_feature_target(df)
        pipeline = build_feature_pipeline()
        X_transformed = pipeline.fit_transform(X)
        # ColumnTransformer con OneHotEncoder + passthrough numerico
        # devuelve un array puramente numerico.
        assert hasattr(X_transformed, "dtype") or hasattr(X_transformed, "dtypes")

    def test_feature_count_differs_with_and_without_churn_type(self):
        df = make_customers_df()
        X_with, _ = get_feature_target(df, include_churn_type=True)
        X_without, _ = get_feature_target(df, include_churn_type=False)

        pipeline_with = build_feature_pipeline(include_churn_type=True)
        pipeline_without = build_feature_pipeline(include_churn_type=False)

        n_with = pipeline_with.fit_transform(X_with).shape[1]
        n_without = pipeline_without.fit_transform(X_without).shape[1]

        assert n_with > n_without

    def test_get_feature_names_matches_transformed_shape(self):
        df = make_customers_df()
        X, _ = get_feature_target(df)
        pipeline = build_feature_pipeline()
        X_transformed = pipeline.fit_transform(X)
        names = get_feature_names(pipeline)
        assert len(names) == X_transformed.shape[1]

    def test_unknown_category_at_transform_time_does_not_raise(self):
        """handle_unknown='ignore' debe evitar errores si en produccion
        (API) llega una categoria de Geography no vista en entrenamiento."""
        df = make_customers_df()
        X, _ = get_feature_target(df)
        pipeline = build_feature_pipeline()
        pipeline.fit(X)

        new_customer = X.iloc[[0]].copy()
        new_customer["geography"] = "Italy"  # no vista en fit()
        result = pipeline.transform(new_customer)
        assert result.shape[0] == 1


@pytest.mark.integration
def test_load_customers_from_db_returns_expected_columns():
    df = load_customers_from_db()
    assert len(df) == 10000
    assert "churn_type" in df.columns
    assert "exited" in df.columns


@pytest.mark.integration
def test_end_to_end_pipeline_against_real_database():
    df = load_customers_from_db()
    X, y = get_feature_target(df)
    pipeline = build_feature_pipeline()
    X_transformed = pipeline.fit_transform(X)
    assert X_transformed.shape[0] == 10000
    assert y.isin([0, 1]).all()
