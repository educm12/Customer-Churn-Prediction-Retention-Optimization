"""
Script standalone para puntuar TODOS los clientes de la tabla `customers` 
e insertar sus predicciones en la tabla `predictions` en bloque 
(bulk insert con `to_sql`).

Reutiliza los mismos bloques que `prediction_service.score_customer`
(prediccion del modelo + impacto economico), pero vectorizados sobre
todo el DataFrame de una vez.

Evita duplicados: la query solo trae los `customer_id` que TODAVIA no
tienen fila en `predictions`, asi que el script se puede volver a
ejecutar sin generar entradas repetidas para el mismo cliente.

Ejecucion:
    python -m src.data.ingest_predictions
"""

import pandas as pd
from sqlalchemy import text

from src.business.churn_impact import add_churn_type_columns
from src.business.prediction_service import MODEL_FEATURE_COLUMNS
from src.config import CAMPAIGN_COST, OPERATIVE_THRESHOLD, RETENTION_SUCCESS_PROB
from src.data.database import get_engine
from src.models.predict import load_model, predict_churn_probability

_CUSTOMERS_QUERY = text(
    """
    SELECT c.customer_id, c.credit_score, c.geography, c.gender, c.age,
           c.tenure, c.balance, c.num_of_products, c.has_cr_card,
           c.is_active_member, c.estimated_salary
    FROM customers c
    WHERE NOT EXISTS (
        SELECT 1 FROM predictions p WHERE p.customer_id = c.customer_id
    )
    """
)

# Nombres PascalCase que espera `churn_impact.add_churn_type_columns`
# (definidos sobre las columnas originales del CSV).
_BUSINESS_COLUMN_MAP = {
    "balance": "Balance",
    "estimated_salary": "EstimatedSalary",
    "num_of_products": "NumOfProducts",
    "tenure": "Tenure",
}


def _fetch_customers_df() -> pd.DataFrame:
    """Clientes que TODAVIA no tienen prediccion guardada (evita
    customer_id duplicados en `predictions`)."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(_CUSTOMERS_QUERY, conn)


def ingest_predictions(
    model_version: str = "xgboost_v1",
    threshold: float = OPERATIVE_THRESHOLD,
) -> int:
    """Puntua a todos los clientes pendientes y guarda sus
    predicciones en un unico bulk insert.

    Returns
    -------
    Numero de predicciones insertadas.
    """
    df = _fetch_customers_df()
    if df.empty:
        print("No hay clientes pendientes de puntuar (todos tienen ya prediccion).")
        return 0

    model = load_model()  

    # --- Riesgo de abandono (vectorizado sobre todo el DataFrame) ---
    X_model = df[MODEL_FEATURE_COLUMNS]
    churn_probability = predict_churn_probability(X_model, model=model)

    # --- Impacto economico (vectorizado) ---
    X_business = df.rename(columns=_BUSINESS_COLUMN_MAP)[list(_BUSINESS_COLUMN_MAP.values())]
    business = add_churn_type_columns(X_business)

    expected_avoided_loss = churn_probability * RETENTION_SUCCESS_PROB * business["EconomicLoss"]
    expected_net_profit = expected_avoided_loss - CAMPAIGN_COST

    predictions_df = pd.DataFrame({
        "customer_id": df["customer_id"],
        "churn_probability": churn_probability.round(4),
        "threshold_used": threshold,
        "churn_prediction": churn_probability >= threshold,
        "churn_type": business["ChurnType"].values,
        "economic_loss": business["EconomicLoss"].values,
        "campaign_cost": CAMPAIGN_COST,
        "expected_avoided_loss": expected_avoided_loss.round(2),
        "expected_net_profit": expected_net_profit.round(2),
        "campaign_recommendation": expected_net_profit > 0,
        "model_version": model_version,
    })

    engine = get_engine()
    predictions_df.to_sql(
        "predictions", engine, if_exists="append", index=False, method="multi", chunksize=1000
    )

    print(f"Predicciones insertadas: {len(predictions_df)} clientes procesados.")
    return len(predictions_df)


if __name__ == "__main__":
    ingest_predictions()