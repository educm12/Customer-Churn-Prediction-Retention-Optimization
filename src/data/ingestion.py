"""
ETL: CSV -> calculo de ChurnType -> PostgreSQL.

Flujo:
    data/raw/churn.csv
        -> limpieza/renombrado de columnas (snake_case, tipos)
        -> compute_value_score() + classify_churn_type() 
        -> tabla `customers` en PostgreSQL

Ejecucion:
    python -m src.data.ingestion
"""

import pandas as pd

from src.business.churn_impact import add_churn_type_columns
from src.data.database import get_engine

CSV_PATH = "data/raw/churn.csv"


def load_and_prepare(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Lee el CSV, calcula ChurnType/EconomicLoss (src/business/churn_impact.py)
    y renombra columnas a snake_case para que coincidan con el esquema SQL."""
    df = pd.read_csv(csv_path)
    df = add_churn_type_columns(df)

    df = df.rename(
        columns={
            "RowNumber": "row_number",
            "CustomerId": "customer_id",
            "Surname": "surname",
            "CreditScore": "credit_score",
            "Geography": "geography",
            "Gender": "gender",
            "Age": "age",
            "Tenure": "tenure",
            "Balance": "balance",
            "NumOfProducts": "num_of_products",
            "HasCrCard": "has_cr_card",
            "IsActiveMember": "is_active_member",
            "EstimatedSalary": "estimated_salary",
            "Exited": "exited",
            "ChurnType": "churn_type",
            "EconomicLoss": "economic_loss",
        }
    )

    df["has_cr_card"] = df["has_cr_card"].astype(bool)
    df["is_active_member"] = df["is_active_member"].astype(bool)
    df["exited"] = df["exited"].astype(bool)

    columns = [
        "customer_id", "row_number", "surname", "credit_score", "geography",
        "gender", "age", "tenure", "balance", "num_of_products", "has_cr_card",
        "is_active_member", "estimated_salary", "exited", "value_score",
        "churn_type", "economic_loss",
    ]
    return df[columns]


def load_to_postgres(df: pd.DataFrame, if_exists: str = "append") -> int:
    """Inserta el DataFrame en la tabla `customers`. Devuelve el numero
    de filas insertadas."""
    engine = get_engine()
    df.to_sql("customers", engine, if_exists=if_exists, index=False, method="multi", chunksize=1000)
    return len(df)


def run_etl(csv_path: str = CSV_PATH) -> int:
    df = load_and_prepare(csv_path)
    n_rows = load_to_postgres(df)
    print(f"ETL completado: {n_rows} clientes cargados en PostgreSQL (tabla 'customers').")
    return n_rows


if __name__ == "__main__":
    run_etl()
