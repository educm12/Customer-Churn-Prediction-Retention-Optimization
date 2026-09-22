"""
Carga del modelo entrenado y prediccion de churn probability.

Usado por: threshold analysis, SHAP y API.
"""

import joblib
import pandas as pd

from src.config import MODEL_PATH

_model_cache = None


def load_model(path: str = MODEL_PATH):
    """Carga el pipeline completo (preprocesado + XGBoost) serializado
    por `src/models/train.py`. Cacheado en memoria: solo se lee de
    disco una vez por proceso (importante para la API, que
    no debe recargar el modelo en cada request)."""
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(path)
    return _model_cache


def predict_churn_probability(X: pd.DataFrame, model=None) -> pd.Series:
    """Devuelve P(Exited=1) para cada fila de X, usando el pipeline
    completo (aplica el mismo preprocesado que en entrenamiento)."""
    model = model if model is not None else load_model()
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="churn_probability")


def predict_churn_label(X: pd.DataFrame, threshold: float, model=None) -> pd.Series:
    """Convierte la probabilidad en clase (0/1) segun un threshold
    dado. El threshold NUNCA se fija a 0.5 por defecto: debe
    especificarse explicitamente."""
    proba = predict_churn_probability(X, model=model)
    return (proba >= threshold).astype(int).rename("churn_prediction")
