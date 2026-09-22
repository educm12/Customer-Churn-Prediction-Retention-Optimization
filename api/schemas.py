"""
Esquemas Pydantic para validar inputs y outputs de la API.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Features de un cliente en el formato que espera el modelo. 
    Usada como base de `PredictRequest`."""

    credit_score: int = Field(..., ge=300, le=900, description="Credit score del cliente")
    geography: str = Field(..., description="Pais del cliente: France, Germany o Spain")
    gender: str = Field(..., description="Genero: Male o Female")
    age: int = Field(..., ge=18, le=100)
    tenure: int = Field(..., ge=0, le=20, description="Años como cliente del banco")
    balance: float = Field(..., ge=0, description="Saldo en cuenta (EUR)")
    num_of_products: int = Field(..., ge=1, le=4)
    has_cr_card: bool
    is_active_member: bool
    estimated_salary: float = Field(..., ge=0)


class PredictRequest(CustomerFeatures):
    customer_id: Optional[int] = Field(
        default=None,
        description="Opcional. Si se indica y el cliente existe en la base de datos, "
        "la prediccion se persiste en la tabla `predictions`.",
    )


class PredictionResponse(BaseModel):
    customer_id: Optional[int] = None
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: int = Field(..., ge=0, le=1)
    threshold_used: float
    churn_type: str
    economic_loss: float
    campaign: bool
    expected_avoided_loss: float
    campaign_cost: float
    expected_net_profit: float


class CustomerResponse(BaseModel):
    customer_id: int
    surname: Optional[str] = None
    credit_score: int
    geography: str
    gender: str
    age: int
    tenure: int
    balance: float
    num_of_products: int
    has_cr_card: bool
    is_active_member: bool
    estimated_salary: float
    exited: Optional[bool] = None
    churn_type: Optional[str] = None
    economic_loss: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    status: str
    database: bool
    model_loaded: bool