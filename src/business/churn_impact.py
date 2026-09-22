"""
Regla de negocio de ChurnType (impacto economico del abandono).

Definida y validada en notebooks/02_business_rules.ipynb con datos
reales del CSV. Se centraliza aqui para que tanto el ETL (src/data/ingestion.py)
como el feature engineering (src/features/feature_engineering.py)
y, mas adelante, la API usen exactamente la misma logica.

IMPORTANTE: estas funciones NUNCA deben recibir ni usar `Exited`.
`ChurnType` representa el impacto economico del abandono, no la probabilidad
de que ocurra (eso lo predice XGBoost).
"""

import numpy as np
import pandas as pd

from src.config import CHURN_TYPE_HARD_THRESHOLD, CHURN_TYPE_SOFT_THRESHOLD, CHURN_TYPE_WEIGHTS, ECONOMIC_LOSS

# Alias locales por compatibilidad con el resto del codigo/tests, que
# referencian estos nombres dentro de este modulo. La UNICA fuente de
# verdad de los valores sigue siendo src/config.py.
WEIGHTS = CHURN_TYPE_WEIGHTS
SOFT_THRESHOLD = CHURN_TYPE_SOFT_THRESHOLD
HARD_THRESHOLD = CHURN_TYPE_HARD_THRESHOLD


def compute_value_score(df: pd.DataFrame) -> np.ndarray:
    """Calcula el indice de valor economico del cliente a partir de
    Balance, EstimatedSalary, NumOfProducts y Tenure (columnas en su
    nomenclatura original del CSV, PascalCase).

    Cada variable se banda en {0.25, 0.50, 0.75, 1.0} segun cortes de
    negocio fijos, y se combina con los pesos de WEIGHTS.
    """
    balance_score = np.select(
        [df["Balance"] < 50000, df["Balance"] < 100000, df["Balance"] < 150000],
        [0.25, 0.50, 0.75],
        default=1.0,
    )
    salary_score = np.select(
        [
            df["EstimatedSalary"] < 50000,
            df["EstimatedSalary"] < 100000,
            df["EstimatedSalary"] < 150000,
        ],
        [0.25, 0.50, 0.75],
        default=1.0,
    )
    products_score = np.select(
        [df["NumOfProducts"] <= 1, df["NumOfProducts"] == 2, df["NumOfProducts"] == 3],
        [0.25, 0.50, 0.75],
        default=1.0,
    )
    tenure_score = np.select(
        [df["Tenure"] <= 2, df["Tenure"] <= 5, df["Tenure"] <= 7],
        [0.25, 0.50, 0.75],
        default=1.0,
    )
    return (
        WEIGHTS["balance"] * balance_score
        + WEIGHTS["salary"] * salary_score
        + WEIGHTS["products"] * products_score
        + WEIGHTS["tenure"] * tenure_score
    )


def classify_churn_type(value_score: float) -> str:
    """Clasifica un value_score individual en softchurn/midchurn/hardchurn."""
    if value_score < SOFT_THRESHOLD:
        return "softchurn"
    elif value_score <= HARD_THRESHOLD:
        return "midchurn"
    else:
        return "hardchurn"


def add_churn_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Añade value_score, ChurnType y EconomicLoss a un DataFrame que
    tenga las columnas originales del CSV (Balance, EstimatedSalary,
    NumOfProducts, Tenure). No modifica el DataFrame de entrada."""
    df = df.copy()
    df["value_score"] = compute_value_score(df)
    df["ChurnType"] = df["value_score"].apply(classify_churn_type)
    df["EconomicLoss"] = df["ChurnType"].map(ECONOMIC_LOSS)
    return df
