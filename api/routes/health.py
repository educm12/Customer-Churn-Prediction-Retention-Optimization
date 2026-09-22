"""GET /health — comprueba conexion a PostgreSQL y modelo cargado."""

from fastapi import APIRouter

from api.schemas import HealthResponse
from src.data.database import check_connection
from src.models.predict import load_model

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    database_ok = check_connection()
    try:
        load_model()
        model_ok = True
    except Exception:
        model_ok = False

    status = "ok" if (database_ok and model_ok) else "degraded"
    return HealthResponse(status=status, database=database_ok, model_loaded=model_ok)