"""
Entrenamiento de XGBoost.

Flujo:
    PostgreSQL (customers)
        -> train/test split estratificado 80/20 (test se aparta y NO se
           vuelve a tocar hasta la evaluacion final)
        -> ablation study: con/sin `churn_type` como feature, evaluado
           SOLO con cross-validation sobre el train set
        -> hyperparameter tuning con Optuna (TPE sampler) +
           StratifiedKFold sobre la configuracion elegida
        -> entrenamiento final sobre todo el train set
        -> evaluacion UNICA sobre el test set
        -> serializacion del pipeline completo (preprocessing + modelo)

Ejecucion:
    python -m src.models.train
"""

import json
import os
import time

import joblib
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODEL_METADATA_PATH, MODEL_PATH, N_CV_FOLDS, N_OPTUNA_TRIALS, OPTUNA_TRIALS_PATH, RANDOM_STATE, TEST_SIZE
from src.features.feature_engineering import build_feature_pipeline, get_feature_target, load_customers_from_db
from src.models.evaluate import compute_probability_metrics

# RANDOM_STATE, TEST_SIZE, N_CV_FOLDS, N_OPTUNA_TRIALS y las rutas se
# definen en src/config.py (unica fuente de verdad); se re-exponen
# aqui por compatibilidad con el resto del modulo.
METADATA_PATH = MODEL_METADATA_PATH

optuna.logging.set_verbosity(optuna.logging.WARNING)  # el progreso ya lo imprime run_training_pipeline


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Ratio negativos/positivos en el set de entrenamiento, para
    compensar el desbalanceo (~80/20) sin necesidad de
    resampling (undersampling/oversampling)."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / n_pos


def build_full_pipeline(include_churn_type: bool, scale_pos_weight: float) -> Pipeline:
    """Pipeline completo: preprocesado + XGBoost, como una
    unica unidad serializable."""
    preprocessing_pipeline = build_feature_pipeline(include_churn_type=include_churn_type)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )
    return Pipeline(steps=[*preprocessing_pipeline.steps, ("model", model)])


def run_ablation_study(
    X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float, cv
) -> dict:
    """Compara include_churn_type=True vs False, evaluado SOLO con CV
    sobre el train set (el test set no se toca en esta decision, para
    no contaminar la evaluacion final).

    `churn_type` no aporta leakage (no depende de
    `exited`), pero podria ser redundante con balance/salary/products/
    tenure. Esto decide, con evidencia, si merece la pena mantenerla.
    """
    results = {}
    for include in [True, False]:
        pipeline = build_full_pipeline(include_churn_type=include, scale_pos_weight=scale_pos_weight)
        scores = cross_validate(
            pipeline,
            X_train if include else X_train.drop(columns=["churn_type"]),
            y_train,
            cv=cv,
            scoring=["roc_auc", "average_precision", "neg_log_loss"],
            n_jobs=-1,
        )
        results[str(include)] = {
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "pr_auc_mean": float(scores["test_average_precision"].mean()),
            "log_loss_mean": float(-scores["test_neg_log_loss"].mean()),
        }
    return results


def _suggest_xgb_params(trial: optuna.Trial) -> dict:
    """Espacio de busqueda de hiperparametros,
    expresado como distribuciones de Optuna en vez de listas discretas:
    permite que el sampler TPE explore el espacio continuo, no solo los
    valores que se nos hubieran ocurrido a mano."""
    return {
        "model__n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "model__max_depth": trial.suggest_int("max_depth", 3, 8),
        "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "model__min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "model__gamma": trial.suggest_float("gamma", 0.0, 0.5),
    }


def _make_objective(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float, include_churn_type: bool, cv):
    """Cierra sobre los datos de train y devuelve la funcion objetivo
    que Optuna maximiza: ROC-AUC medio en `cv` (StratifiedKFold) sobre
    el train set. El test set nunca entra aqui."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial)
        pipeline = build_full_pipeline(include_churn_type=include_churn_type, scale_pos_weight=scale_pos_weight)
        pipeline.set_params(**params)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    return objective


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    include_churn_type: bool,
    cv,
    n_trials: int = N_OPTUNA_TRIALS,
) -> optuna.Study:
    """Optimiza hiperparametros con Optuna (sampler TPE: Tree-structured
    Parzen Estimator), que a diferencia de RandomizedSearchCV elige cada
    combinacion en funcion de los resultados de las anteriores (busqueda
    bayesiana), en vez de muestrear el espacio a ciegas. Con el mismo
    presupuesto de intentos, suele converger a mejores hiperparametros."""
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgboost_churn_tuning")
    objective = _make_objective(X_train, y_train, scale_pos_weight, include_churn_type, cv)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def run_training_pipeline() -> tuple[Pipeline, dict]:
    print("Cargando datos desde PostgreSQL...")
    df = load_customers_from_db()

    # Cargamos siempre con churn_type incluido; el ablation study decide
    # despues si se usa o no, filtrando columnas del mismo split.
    X, y = get_feature_target(df, include_churn_type=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} filas | Test: {len(X_test)} filas")
    print(f"Tasa de churn en train: {y_train.mean():.4f} | en test: {y_test.mean():.4f}")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight (neg/pos en train): {scale_pos_weight:.3f}")

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("\n=== Ablation study: include_churn_type True vs False (CV sobre train) ===")
    ablation_results = run_ablation_study(X_train, y_train, scale_pos_weight, cv)
    for include, metrics in ablation_results.items():
        print(f"  include_churn_type={include}: {metrics}")

    best_include_churn_type = ablation_results["True"]["roc_auc_mean"] >= ablation_results["False"]["roc_auc_mean"]
    print(f"\n-> Configuracion elegida: include_churn_type={best_include_churn_type}")

    X_train_final = X_train if best_include_churn_type else X_train.drop(columns=["churn_type"])
    X_test_final = X_test if best_include_churn_type else X_test.drop(columns=["churn_type"])

    print("\n=== Hyperparameter tuning (Optuna, TPE sampler) ===")
    study = tune_hyperparameters(
        X_train_final, y_train, scale_pos_weight, best_include_churn_type, cv, n_trials=N_OPTUNA_TRIALS
    )
    print(f"Mejor ROC-AUC (CV en train): {study.best_value:.4f}")
    print(f"Mejores hiperparametros: {study.best_params}")

    best_pipeline = build_full_pipeline(include_churn_type=best_include_churn_type, scale_pos_weight=scale_pos_weight)
    best_pipeline.set_params(**{f"model__{k}": v for k, v in study.best_params.items()})
    best_pipeline.fit(X_train_final, y_train)

    print("\n=== Evaluacion final sobre TEST set (no usado hasta ahora) ===")
    y_proba_test = best_pipeline.predict_proba(X_test_final)[:, 1]
    test_metrics = compute_probability_metrics(y_test, y_proba_test)
    for name, value in test_metrics.items():
        print(f"  Test {name}: {value:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    # Historial completo de trials de Optuna (util para graficar
    # convergencia de la busqueda).
    os.makedirs("reports", exist_ok=True)
    study.trials_dataframe().to_csv(OPTUNA_TRIALS_PATH, index=False)

    metadata = {
        "include_churn_type": best_include_churn_type,
        "scale_pos_weight": scale_pos_weight,
        "optimization_method": "optuna_tpe",
        "best_params": study.best_params,
        "cv_roc_auc": float(study.best_value),
        "test_metrics": test_metrics,
        "ablation_results": ablation_results,
        "n_train": len(X_train_final),
        "n_test": len(X_test_final),
        "test_size": TEST_SIZE,
        "n_cv_folds": N_CV_FOLDS,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "random_state": RANDOM_STATE,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    print(f"\nModelo guardado en {MODEL_PATH}")
    print(f"Metadata guardada en {METADATA_PATH}")
    print(f"Historial de trials de Optuna guardado en {OPTUNA_TRIALS_PATH}")

    # Se guardan train/test (features + target + probabilidades) para
    # reutilizarlos en el threshold analysis sin tener que repetir
    # el entrenamiento.
    os.makedirs("data/processed", exist_ok=True)
    X_test_final.assign(churn_probability=y_proba_test).to_csv("data/processed/test_predictions.csv", index=False)
    y_test.to_frame(name="exited").to_csv("data/processed/y_test.csv", index=False)

    return best_pipeline, metadata


if __name__ == "__main__":
    run_training_pipeline()
