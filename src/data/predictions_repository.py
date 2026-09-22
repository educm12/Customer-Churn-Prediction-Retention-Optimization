"""
Persistencia de predicciones en la tabla `predictions` (esquema
definido en `sql/init_schema.sql`).

Separado de `src/business/prediction_service.py` para mantener la
logica de negocio (calcular la prediccion) independiente del acceso a
datos (guardarla) — la API usa ambos; el dashboard puede reutilizar 
`prediction_service` sin depender de la base de datos si quisiera.
"""

from sqlalchemy import text

from src.data.database import get_engine

_INSERT_PREDICTION_SQL = text(
    """
    INSERT INTO predictions (
        customer_id, churn_probability, threshold_used, churn_prediction,
        churn_type, economic_loss, campaign_cost, expected_avoided_loss,
        expected_net_profit, campaign_recommendation, model_version
    ) VALUES (
        :customer_id, :churn_probability, :threshold_used, :churn_prediction,
        :churn_type, :economic_loss, :campaign_cost, :expected_avoided_loss,
        :expected_net_profit, :campaign_recommendation, :model_version
    )
    """
)


def save_prediction(prediction: dict, model_version: str = "xgboost_v1") -> bool:
    """Inserta una prediccion en la tabla `predictions`.

    Requiere que `prediction['customer_id']` exista en la tabla
    `customers` (clave foranea). Si `customer_id` es None (cliente 
    hipotetico puntuado via `POST /predict` sin id, no dado de alta 
    en la base de datos), no seguarda nada y se devuelve False sin 
    lanzar error: es un caso valido, no un fallo.

    Returns
    -------
    True si se guardo, False si se omitio (customer_id ausente).
    """
    if prediction.get("customer_id") is None:
        return False

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            _INSERT_PREDICTION_SQL,
            {
                "customer_id": prediction["customer_id"],
                "churn_probability": prediction["churn_probability"],
                "threshold_used": prediction["threshold_used"],
                "churn_prediction": bool(prediction["churn_prediction"]),
                "churn_type": prediction["churn_type"],
                "economic_loss": prediction["economic_loss"],
                "campaign_cost": prediction["campaign_cost"],
                "expected_avoided_loss": prediction["expected_avoided_loss"],
                "expected_net_profit": prediction["expected_net_profit"],
                "campaign_recommendation": bool(prediction["campaign"]),
                "model_version": model_version,
            },
        )
    return True