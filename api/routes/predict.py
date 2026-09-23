"""POST /predict — puntua un cliente (existente o hipotetico) a partir
de sus features, sin necesidad de que este ya en la base de datos."""

from fastapi import APIRouter

from api.schemas import PredictRequest, PredictionResponse
from src.business.prediction_service import score_customer

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> PredictionResponse:
    features = request.model_dump(exclude={"customer_id"})
    prediction = score_customer(features, customer_id=request.customer_id)
    return PredictionResponse(**prediction)