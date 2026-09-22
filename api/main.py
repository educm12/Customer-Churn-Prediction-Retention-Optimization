"""
Punto de entrada de la API.

Ejecucion local:
    uvicorn api.main:app --reload --port 8000

Endpoints (ver README para ejemplos):
    GET  /health
    GET  /customers/{customer_id}
    GET  /customers/{customer_id}/prediction
    POST /predict
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import customers, health, predict
from src.models.predict import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga el modelo una vez al arrancar el servidor (queda cacheado
    # en memoria por `src.models.predict.load_model`), para que la
    # primera peticion real no pague el coste de leer el .pkl de disco.
    load_model()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Prediccion de churn (XGBoost) + impacto economico (ChurnType) + "
        "decision de campaña de retencion, sobre datos de clientes de un banco."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(health.router)
app.include_router(customers.router)
app.include_router(predict.router)