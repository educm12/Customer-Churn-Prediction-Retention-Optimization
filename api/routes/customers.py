"""GET /customers/{customer_id} y GET /customers/{customer_id}/prediction."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from api.schemas import CustomerResponse, PredictionResponse
from src.business.prediction_service import score_customer
from src.data.database import get_engine
from src.data.predictions_repository import save_prediction

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


def _fetch_customer_row(customer_id: int) -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(_CUSTOMER_QUERY, {"customer_id": customer_id}).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Cliente {customer_id} no encontrado")
    return dict(row)


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
    row = _fetch_customer_row(customer_id)
    features = {
        "credit_score": row["credit_score"],
        "geography": row["geography"],
        "gender": row["gender"],
        "age": row["age"],
        "tenure": row["tenure"],
        "balance": float(row["balance"]),
        "num_of_products": row["num_of_products"],
        "has_cr_card": row["has_cr_card"],
        "is_active_member": row["is_active_member"],
        "estimated_salary": float(row["estimated_salary"]),
    }
    prediction = score_customer(features, customer_id=customer_id)
    save_prediction(prediction)  # historico en la tabla `predictions`
    return PredictionResponse(**prediction)