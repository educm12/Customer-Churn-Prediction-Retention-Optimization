"""
Metricas de evaluacion reutilizables.

Se separan en dos grupos:
    - Metricas de PROBABILIDAD (no dependen de un threshold): ROC-AUC,
      PR-AUC, Log Loss. Se usan durante el entrenamiento/tuning
      y para comparar modelos entre si.
    - Metricas de THRESHOLD (dependen de convertir probabilidad -> clase
      con `probability >= threshold`): Accuracy, Precision, Recall, F1,
      Specificity, FPR/TPR, matriz de confusion. Se usan en el analisis
      de thresholds.

Ambos grupos se usan tanto en la evaluacion del modelo como en
el barrido de thresholds y el dashboard de evaluacion.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def to_clean_arrays(y_true, y_proba=None):
    """Convierte y_true/y_proba a arrays de numpy con dtype estandar
    (int64/float64), sin dtypes 'extension' de pandas (Int64, boolean,
    Float64 con capital, propios de columnas leidas desde PostgreSQL
    en ciertas versiones de pandas/SQLAlchemy).

    Por que hace falta: sklearn (`roc_curve`, `precision_recall_curve`,
    `type_of_target`, etc.) no siempre reconoce los dtypes 'nullable'
    de pandas y puede lanzar `ValueError: unknown format is not
    supported` aunque los datos sean perfectamente binarios. Esto es
    dependiente de la version de pandas/sklearn instalada, asi que
    puede no reproducirse en todos los entornos — forzar aqui un dtype
    numpy estandar lo hace robusto en cualquier maquina.
    """
    y_true_arr = np.asarray(y_true).astype(np.int64)
    if y_proba is None:
        return y_true_arr
    y_proba_arr = np.asarray(y_proba).astype(np.float64)
    return y_true_arr, y_proba_arr


def compute_probability_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict:
    """Metricas que no dependen de un threshold de clasificacion."""
    y_true, y_proba = to_clean_arrays(y_true, y_proba)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }


def compute_threshold_metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    """Metricas de clasificacion para un threshold concreto.

    `y_pred = 1` si `y_proba >= threshold`.
    """
    y_true, y_proba = to_clean_arrays(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # equivalente a recall

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "fpr": fpr,
        "tpr": tpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_predicted_churn": int(y_pred.sum()),
        "n_predicted_no_churn": int(len(y_pred) - y_pred.sum()),
    }


def threshold_sweep(y_true: pd.Series, y_proba: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    """Aplica `compute_threshold_metrics` para una lista de thresholds y
    devuelve una tabla (una fila por threshold). Base para el analisis
    completo de los thresholds."""
    rows = [compute_threshold_metrics(y_true, y_proba, t) for t in thresholds]
    return pd.DataFrame(rows)