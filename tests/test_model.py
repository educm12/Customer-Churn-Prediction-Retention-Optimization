"""
Tests de src/models/train.py, evaluate.py, predict.py.

Los tests que entrenan un XGBoost completo con Optuna (muchos trials x
5 folds)
serian demasiado lentos para una suite normal, asi que:
    - Las funciones de evaluate.py se testean con datos sinteticos
      (rapido, sin dependencias).
    - build_full_pipeline / compute_scale_pos_weight se testean de
      forma aislada (no requieren entrenar).
    - Solo los tests marcados `integration` cargan el modelo YA
      entrenado (models/xgboost_model.pkl) y PostgreSQL, sin re-entrenar.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import compute_probability_metrics, compute_threshold_metrics, threshold_sweep, to_clean_arrays
from src.models.train import build_full_pipeline, compute_scale_pos_weight


# ============================================================
# src/models/evaluate.py (unitarios, sin dependencias externas)
# ============================================================
class TestComputeProbabilityMetrics:
    def test_perfect_predictions_give_roc_auc_one(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_proba = np.array([0.01, 0.02, 0.98, 0.99])
        metrics = compute_probability_metrics(y_true, y_proba)
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_random_predictions_give_roc_auc_near_half(self):
        rng = np.random.default_rng(0)
        y_true = pd.Series(rng.integers(0, 2, 2000))
        y_proba = rng.uniform(0, 1, 2000)
        metrics = compute_probability_metrics(y_true, y_proba)
        assert 0.4 < metrics["roc_auc"] < 0.6

    def test_returns_all_three_metrics(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7])
        metrics = compute_probability_metrics(y_true, y_proba)
        assert set(metrics.keys()) == {"roc_auc", "pr_auc", "log_loss"}


class TestComputeThresholdMetrics:
    def test_threshold_zero_predicts_everyone_as_churn(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = compute_threshold_metrics(y_true, y_proba, threshold=0.0)
        assert metrics["n_predicted_churn"] == 4
        assert metrics["recall"] == 1.0  # todos los positivos reales se capturan

    def test_threshold_above_max_proba_predicts_nobody_as_churn(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = compute_threshold_metrics(y_true, y_proba, threshold=1.01)
        assert metrics["n_predicted_churn"] == 0
        assert metrics["recall"] == 0.0

    def test_confusion_matrix_values_sum_to_total(self):
        y_true = pd.Series([0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.6, 0.2])
        metrics = compute_threshold_metrics(y_true, y_proba, threshold=0.5)
        assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == 5

    def test_recall_equals_tpr(self):
        y_true = pd.Series([0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.6, 0.2])
        metrics = compute_threshold_metrics(y_true, y_proba, threshold=0.5)
        assert metrics["recall"] == pytest.approx(metrics["tpr"])

    def test_higher_threshold_never_increases_recall(self):
        y_true = pd.Series([0, 1, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.6, 0.55, 0.3, 0.7])
        low = compute_threshold_metrics(y_true, y_proba, threshold=0.2)
        high = compute_threshold_metrics(y_true, y_proba, threshold=0.8)
        assert high["recall"] <= low["recall"]


class TestThresholdSweep:
    def test_returns_one_row_per_threshold(self):
        y_true = pd.Series([0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.6, 0.2])
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = threshold_sweep(y_true, y_proba, thresholds)
        assert len(result) == len(thresholds)
        assert list(result["threshold"]) == thresholds

    def test_n_predicted_churn_decreases_as_threshold_increases(self):
        y_true = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.4, 0.6, 0.55, 0.3, 0.7, 0.2])
        result = threshold_sweep(y_true, y_proba, [0.1, 0.5, 0.9])
        n_churn = result["n_predicted_churn"].tolist()
        assert n_churn == sorted(n_churn, reverse=True)


class TestToCleanArraysDtypeRobustness:
    """Regresion: en algunos entornos (Windows, ciertas combinaciones
    de version de pandas/SQLAlchemy leyendo de PostgreSQL), las
    columnas booleanas/enteras llegan con dtypes 'extension' de pandas
    (`Int64`, `boolean`, con mayuscula) en vez de los numpy estandar
    (`int64`, `bool`). sklearn (`roc_curve`, `type_of_target`, etc.) no
    siempre los reconoce y lanza `ValueError: unknown format is not
    supported` aunque los datos sean perfectamente binarios. Estos
    tests reproducen esos dtypes explicitamente para evitar que el
    problema vuelva a colarse sin que se note aqui (en Linux, con las
    versiones de este entorno, el bug original no se reproduce, pero
    el fix debe seguir siendo correcto igualmente)."""

    def test_handles_pandas_nullable_int_dtype(self):
        y_true = pd.Series([0, 1, 0, 1], dtype="Int64")  # nullable, no numpy int64
        y_proba = np.array([0.1, 0.9, 0.4, 0.6])
        y_true_arr, y_proba_arr = to_clean_arrays(y_true, y_proba)
        assert y_true_arr.dtype == np.int64
        assert y_proba_arr.dtype == np.float64

    def test_handles_pandas_nullable_boolean_dtype(self):
        y_true = pd.Series([False, True, False, True], dtype="boolean")  # nullable
        y_proba = np.array([0.1, 0.9, 0.4, 0.6])
        y_true_arr, _ = to_clean_arrays(y_true, y_proba)
        assert y_true_arr.dtype == np.int64
        assert list(y_true_arr) == [0, 1, 0, 1]

    def test_handles_pandas_nullable_float_dtype_for_proba(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_proba = pd.Series([0.1, 0.9, 0.4, 0.6], dtype="Float64")  # nullable
        _, y_proba_arr = to_clean_arrays(y_true, y_proba)
        assert y_proba_arr.dtype == np.float64

    def test_compute_probability_metrics_works_with_nullable_dtypes(self):
        """El caso real que fallaba: roc_curve (via
        compute_probability_metrics) con dtypes nullable."""
        y_true = pd.Series([0, 1, 0, 1, 1, 0], dtype="Int64")
        y_proba = pd.Series([0.1, 0.9, 0.3, 0.6, 0.8, 0.2], dtype="Float64")
        metrics = compute_probability_metrics(y_true, y_proba)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_compute_threshold_metrics_works_with_nullable_dtypes(self):
        y_true = pd.Series([0, 1, 0, 1, 1, 0], dtype="boolean")
        y_proba = pd.Series([0.1, 0.9, 0.3, 0.6, 0.8, 0.2], dtype="Float64")
        metrics = compute_threshold_metrics(y_true, y_proba, threshold=0.5)
        assert metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] == 6

    def test_returns_plain_numpy_not_pandas_extension_array(self):
        """El resultado debe ser numpy 'de verdad', no un array de
        extension de pandas envuelto — para que cualquier libreria de
        terceros (sklearn, matplotlib, shap) lo acepte sin sorpresas."""
        y_true = pd.Series([0, 1], dtype="Int64")
        y_true_arr = to_clean_arrays(y_true)
        assert isinstance(y_true_arr, np.ndarray)
        assert not hasattr(y_true_arr, "mask")  # los MaskedArray/IntegerArray de pandas si lo tienen


# ============================================================
# src/models/train.py (funciones auxiliares, sin entrenar)
# ============================================================
class TestComputeScalePosWeight:
    def test_balanced_classes_give_ratio_one(self):
        y = pd.Series([0, 1, 0, 1])
        assert compute_scale_pos_weight(y) == pytest.approx(1.0)

    def test_imbalanced_classes_give_ratio_above_one(self):
        # 80% clase 0, 20% clase 1 (como el dataset real)
        y = pd.Series([0] * 80 + [1] * 20)
        assert compute_scale_pos_weight(y) == pytest.approx(4.0)


class TestBuildFullPipeline:
    def test_pipeline_has_preprocessing_and_model_steps(self):
        pipeline = build_full_pipeline(include_churn_type=True, scale_pos_weight=4.0)
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessing" in step_names
        assert "model" in step_names

    def test_model_step_receives_scale_pos_weight(self):
        pipeline = build_full_pipeline(include_churn_type=False, scale_pos_weight=3.9)
        assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(3.9)

    def test_pipeline_fits_and_predicts_on_synthetic_data(self):
        """Smoke test rapido: un pipeline SIN tuning (hiperparametros por
        defecto, pocos arboles) debe poder entrenar y predecir sobre un
        DataFrame pequeño sin errores. No sustituye al entrenamiento
        real, solo verifica que el pipeline esta bien construido."""
        n = 100
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
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
        y = pd.Series(rng.integers(0, 2, n))

        pipeline = build_full_pipeline(include_churn_type=False, scale_pos_weight=1.0)
        pipeline.set_params(model__n_estimators=10, model__max_depth=2)
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)[:, 1]
        assert len(proba) == n
        assert ((proba >= 0) & (proba <= 1)).all()


class TestTuneHyperparametersOptuna:
    def _make_synthetic_data(self, n=150, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
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
        y = pd.Series(rng.integers(0, 2, n))
        return X, y

    def test_tune_hyperparameters_runs_and_returns_study_with_best_params(self):
        """Con muy pocos trials y datos sinteticos pequeños (rapido, sin
        DB), verifica que la integracion con Optuna funciona de punta a
        punta: crea un study, explora, y devuelve mejores hiperparametros
        dentro del espacio de busqueda esperado."""
        from sklearn.model_selection import StratifiedKFold

        from src.models.train import tune_hyperparameters

        X, y = self._make_synthetic_data()
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        study = tune_hyperparameters(X, y, scale_pos_weight=1.0, include_churn_type=False, cv=cv, n_trials=3)

        assert len(study.trials) == 3
        assert 0.0 <= study.best_value <= 1.0
        assert set(study.best_params.keys()) == {
            "n_estimators", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_weight", "gamma",
        }
        assert 100 <= study.best_params["n_estimators"] <= 500
        assert 3 <= study.best_params["max_depth"] <= 8

    def test_tune_hyperparameters_is_reproducible_with_same_seed(self):
        """El sampler TPE se fija con RANDOM_STATE: dos llamadas con los
        mismos datos deben explorar los mismos puntos y llegar al mismo
        mejor resultado."""
        from sklearn.model_selection import StratifiedKFold

        from src.models.train import tune_hyperparameters

        X, y = self._make_synthetic_data()
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        study_1 = tune_hyperparameters(X, y, scale_pos_weight=1.0, include_churn_type=False, cv=cv, n_trials=3)
        study_2 = tune_hyperparameters(X, y, scale_pos_weight=1.0, include_churn_type=False, cv=cv, n_trials=3)

        assert study_1.best_params == study_2.best_params
        assert study_1.best_value == pytest.approx(study_2.best_value)


# ============================================================
# Integracion: modelo YA entrenado (no reentrena, solo carga)
# ============================================================
@pytest.mark.integration
def test_trained_model_file_exists_and_loads():
    from src.models.predict import load_model

    model = load_model()
    assert model is not None
    assert "model" in dict(model.steps)


@pytest.mark.integration
def test_trained_model_predicts_valid_probabilities_on_real_data():
    from src.features.feature_engineering import get_feature_target, load_customers_from_db
    from src.models.predict import predict_churn_probability

    df = load_customers_from_db()
    X, _ = get_feature_target(df, include_churn_type=False)
    proba = predict_churn_probability(X.head(50))
    assert len(proba) == 50
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.integration
def test_trained_model_meets_minimum_roc_auc_on_test_set():
    """Test de regresion: el modelo guardado debe mantener un ROC-AUC
    razonable sobre el test set guardado durante el entrenamiento. Si un reentrenamiento
    futuro baja mucho de esto, este test debe fallar."""
    import pandas as pd

    from src.models.evaluate import compute_probability_metrics

    test_predictions = pd.read_csv("data/processed/test_predictions.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")["exited"]

    metrics = compute_probability_metrics(y_test, test_predictions["churn_probability"])
    assert metrics["roc_auc"] > 0.80