"""
Servicio de prediccion: une el riesgo de abandono (XGBoost) con el 
impacto economico (ChurnType) y la decision de campaña en un unico punto de entrada.

Se disena independiente de FastAPI a proposito: es pura logica de
negocio sobre un diccionario de features, para que tanto la API 
como el dashboard de Streamlit puedan reutilizarla sin
duplicar codigo ni depender el uno del otro.

============================================================
Sobre la decision de campaña ("campaign") en este servicio
============================================================
Aqui se recomienda campaña si el beneficio esperado de ESE cliente concreto es positivo
(`expected_net_profit > 0`), calculado con su propio `economic_loss`
segun su `ChurnType`. Esto es mas granular que el threshold GLOBAL
optimizado para maximizar 'F1' (`OPERATIVE_THRESHOLD`, un unico corte de
probabilidad para toda la cartera): el punto de equilibrio real de un
cliente depende de cuanto se perderia si se va, no solo de su
probabilidad de abandono. Por ejemplo, con la economia actual
(`RETENTION_SUCCESS_PROB=0.30`, `CAMPAIGN_COST=50€`), el breakeven en
probabilidad es ~8.3% para un `hardchurn` pero ~33.3% para un
`softchurn` — un cliente de bajo valor necesita mucha mas certeza de
abandono para que compense contactarlo.

`churn_prediction` (0/1), en cambio, SI usa el threshold operativo
global (`OPERATIVE_THRESHOLD`) para mantener coherencia con las
metricas de clasificacion reportadas (Precision,mRecall, etc., 
que se calcularon con ese unico corte). Por eso`churn_prediction` y 
`campaign` pueden diferir para un mismo cliente:
son dos preguntas distintas ("¿lo clasificamos como churner segun el
corte operativo estandar?" vs "¿compensa economicamente
contactarLE A EL en concreto?").
"""

from typing import Optional

import pandas as pd

from src.business.churn_impact import add_churn_type_columns
from src.config import OPERATIVE_THRESHOLD, CAMPAIGN_COST, RETENTION_SUCCESS_PROB
from src.models.predict import load_model, predict_churn_probability

# Columnas (snake_case) que el modelo final espera, en el mismo orden
# que src/features/feature_engineering.py (sin churn_type).
MODEL_FEATURE_COLUMNS = [
    "credit_score", "age", "tenure", "balance", "num_of_products",
    "has_cr_card", "is_active_member", "estimated_salary", "geography", "gender",
]


def _features_to_model_input(features: dict) -> pd.DataFrame:
    return pd.DataFrame([{col: features[col] for col in MODEL_FEATURE_COLUMNS}])


def _features_to_business_input(features: dict) -> pd.DataFrame:
    """Traduce a los nombres PascalCase que usa `churn_impact.py`
    (definidos sobre las columnas originales del CSV)."""
    return pd.DataFrame([{
        "Balance": features["balance"],
        "EstimatedSalary": features["estimated_salary"],
        "NumOfProducts": features["num_of_products"],
        "Tenure": features["tenure"],
    }])


def score_customer(
    features: dict,
    customer_id: Optional[int] = None,
    model=None,
    threshold: float = OPERATIVE_THRESHOLD,
) -> dict:
    """Calcula la prediccion completa (riesgo + impacto economico +
    decision de campaña) para UN cliente.

    Parameters
    ----------
    features:
        Diccionario con las claves de `MODEL_FEATURE_COLUMNS`
        (credit_score, age, tenure, balance, num_of_products,
        has_cr_card, is_active_member, estimated_salary, geography,
        gender), en snake_case.
    customer_id:
        Opcional; se incluye en la respuesta y se usa como referencia
        si la prediccion se persiste despues (ver
        `src/data/predictions_repository.py`).
    threshold:
        Threshold operativo para `churn_prediction`. Por defecto, el
        threshold de negocio global que maximiza el beneficio neto de la cartera
        (`OPERATIVE_THRESHOLD` en `src/config.py`).
    """
    model = model if model is not None else load_model()

    X_model = _features_to_model_input(features)
    churn_probability = float(predict_churn_probability(X_model, model=model).iloc[0])

    X_business = _features_to_business_input(features)
    business_row = add_churn_type_columns(X_business).iloc[0]
    churn_type = str(business_row["ChurnType"])
    economic_loss = float(business_row["EconomicLoss"])

    churn_prediction = int(churn_probability >= threshold)

    expected_avoided_loss = churn_probability * RETENTION_SUCCESS_PROB * economic_loss
    expected_net_profit = expected_avoided_loss - CAMPAIGN_COST
    campaign = expected_net_profit > 0  # decision economica individual 

    return {
        "customer_id": customer_id,
        "churn_probability": round(churn_probability, 4),
        "churn_prediction": churn_prediction,
        "threshold_used": threshold,
        "churn_type": churn_type,
        "economic_loss": economic_loss,
        "campaign": campaign,
        "expected_avoided_loss": round(expected_avoided_loss, 2),
        "campaign_cost": CAMPAIGN_COST,
        "expected_net_profit": round(expected_net_profit, 2),
    }