"""
Feature engineering para el modelo XGBoost.

Fuente de datos: PostgreSQL, tabla `customers`: tras la migracion, 
PostgreSQL es la fuente principal, no el CSV.

Decisiones de diseño y por que (evitar data leakage, ver notebooks/01_eda.ipynb):

    EXCLUIDAS SIEMPRE:
    - `exited`: es el target, nunca una feature.
    - `customer_id`, `row_number`, `surname`: identificadores sin poder
      predictivo genuino.
    - `value_score`: es una combinacion lineal determinista de
      balance/estimated_salary/num_of_products/tenure, que YA estan
      presentes como features individuales. Incluirla duplicaria esa
      señal casi por completo (alta redundancia / multicolinealidad
      innecesaria) sin aportar informacion nueva.
    - `economic_loss`: mapeo 1 a 1 de `churn_type` a EUR. Incluir ambas
      seria 100% redundante.
    - `created_at`, `updated_at`: metadatos tecnicos, no de negocio.

    INCLUIDA POR DEFECTO (configurable):
    - `churn_type`: a diferencia de `value_score`, es una variable
      categorica (no una combinacion lineal explicita) y SI podria
      aportar una frontera de decision no lineal distinta a la que
      capturan las variables numericas por separado. Se dispone del
      parametro `include_churn_type` para poder comparar metricas y 
      feature importance con y sin esta variable antes de decidir 
      si se queda en el modelo final.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.database import get_engine

TARGET_COL = "exited"

NUMERIC_FEATURES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "num_of_products",
    "has_cr_card",
    "is_active_member",
    "estimated_salary",
]
CATEGORICAL_FEATURES = ["geography", "gender"]
OPTIONAL_CATEGORICAL_FEATURES = ["churn_type"]

EXCLUDED_COLUMNS = [
    "customer_id",
    "row_number",
    "surname",
    "value_score",
    "economic_loss",
    "created_at",
    "updated_at",
]


def load_customers_from_db() -> pd.DataFrame:
    """Carga la tabla `customers` completa desde PostgreSQL."""
    engine = get_engine()
    return pd.read_sql("SELECT * FROM customers", engine)


def get_feature_target(df: pd.DataFrame, include_churn_type: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Separa un DataFrame de `customers` en (X, y), quedandose solo con
    las columnas que se usaran como features.

    Parameters
    ----------
    df:
        DataFrame con las columnas de la tabla `customers` (snake_case).
    include_churn_type:
        Si True (por defecto), incluye `churn_type` como feature
        categorica. 
    """
    categorical = CATEGORICAL_FEATURES + (OPTIONAL_CATEGORICAL_FEATURES if include_churn_type else [])
    feature_cols = NUMERIC_FEATURES + categorical

    missing = [c for c in feature_cols + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas esperadas ausentes en el DataFrame: {missing}")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


def build_preprocessing_pipeline(include_churn_type: bool = True) -> ColumnTransformer:
    """Construye el ColumnTransformer de preprocesado.

    Solo transforma las variables categoricas (One-Hot Encoding). Las
    numericas pasan sin escalar (`remainder='passthrough'`): XGBoost, al
    ser un modelo basado en arboles, no requiere que las features
    numericas esten normalizadas/estandarizadas.
    """
    categorical = CATEGORICAL_FEATURES + (OPTIONAL_CATEGORICAL_FEATURES if include_churn_type else [])
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(include_churn_type: bool = True) -> Pipeline:
    """Pipeline de sklearn con el preprocesado listo para `.fit_transform()`.

    El estimador XGBoost se añadira como ultimo paso de este pipeline en
    `src/models/train.py`, de forma que todo el flujo
    preprocesado + modelo se pueda serializar y reutilizar como una
    unica unidad (necesario para que `predict.py` y la API
    apliquen exactamente el mismo preprocesado que se uso en entrenamiento).
    """
    preprocessing = build_preprocessing_pipeline(include_churn_type=include_churn_type)
    return Pipeline(steps=[("preprocessing", preprocessing)])


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Devuelve los nombres de las columnas de salida tras el
    preprocesado (util para feature importance / SHAP)."""
    return list(pipeline.named_steps["preprocessing"].get_feature_names_out())
