"""GET /customers/{customer_id} y GET /customers/{customer_id}/prediction."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from api.schemas import CustomerResponse, PredictionResponse
from src.data.database import get_engine

router = APIRouter(prefix="/customers", tags=["customers"])

_CUSTOMER_QUERY = text(
    """
    SELECT customer_id, surname, credit_score, geography, gender, age, tenure,
           balance, num_of_products, has_cr_card, is_active_member,
           estimated_salary, exited, churn_type, economic_loss
    FROM customers
    WHERE customer_id = :customer_id
    """
)

_LATEST_PREDICTION_QUERY = text(
    """
    SELECT customer_id, churn_probability, threshold_used, churn_prediction,
           churn_type, economic_loss, campaign_cost, expected_avoided_loss,
           expected_net_profit, campaign_recommendation
    FROM predictions
    WHERE customer_id = :customer_id
    ORDER BY prediction_id DESC
    LIMIT 1
    """
)


def _fetch_customer_row(customer_id: int) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(_CUSTOMER_QUERY, {"customer_id": customer_id}).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Cliente {customer_id} no encontrado")
    return dict(row)


def _fetch_latest_prediction(customer_id: int) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(_LATEST_PREDICTION_QUERY, {"customer_id": customer_id}).mappings().first()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No hay prediccion guardada para el cliente {customer_id}",
        )
    prediction = dict(row)
    prediction["campaign"] = prediction.pop("campaign_recommendation")
    return prediction


@router.get("/{customer_id}", response_model=CustomerResponse)
def get_customer(customer_id: int) -> CustomerResponse:
    row = _fetch_customer_row(customer_id)
    row["balance"] = float(row["balance"])
    row["estimated_salary"] = float(row["estimated_salary"])
    if row["economic_loss"] is not None:
        row["economic_loss"] = float(row["economic_loss"])
    return CustomerResponse(**row)


@router.get("/{customer_id}/prediction", response_model=PredictionResponse)
def get_customer_prediction(customer_id: int) -> PredictionResponse:
    _fetch_customer_row(customer_id)  # 404 si el cliente no existe
    prediction = _fetch_latest_prediction(customer_id)  # 404 si no tiene prediccion guardada
    return PredictionResponse(**prediction)