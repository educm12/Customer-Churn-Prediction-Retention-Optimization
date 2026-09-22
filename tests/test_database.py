"""
Tests de integracion para el PostgreSQL + ETL.

Requieren una instancia de PostgreSQL accesible con las credenciales
definidas en .env (local, o vía `docker compose up -d postgres`).
Se marcan con el marker `integration` para poder excluirlos en runs
rapidos de unit tests: `pytest -m "not integration"`.
"""

import pytest

from src.business.churn_impact import classify_churn_type, compute_value_score
from src.data.database import check_connection, get_engine
from src.data.ingestion import load_and_prepare


pytestmark = pytest.mark.integration


def test_database_connection():
    """La conexion a PostgreSQL debe funcionar con las credenciales de .env."""
    assert check_connection() is True


def test_customers_table_has_expected_row_count():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.exec_driver_sql("SELECT COUNT(*) FROM customers").scalar()
    assert result == 10000


def test_customers_table_has_no_null_churn_type():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.exec_driver_sql(
            "SELECT COUNT(*) FROM customers WHERE churn_type IS NULL"
        ).scalar()
    assert result == 0


def test_churn_type_distribution_matches_business_rule():
    """La distribucion de ChurnType en la base de datos debe coincidir
    con las reglas de negocio definidas (bandas fijas + umbrales
    0.4/0.6), no con una version antigua (percentiles/terciles)."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT churn_type, COUNT(*) FROM customers GROUP BY churn_type"
        ).fetchall()
    counts = dict(rows)
    assert counts["softchurn"] == 1667
    assert counts["midchurn"] == 4687
    assert counts["hardchurn"] == 3646


def test_load_and_prepare_applies_business_rule_consistently():
    """`load_and_prepare` (usado por el ETL) debe generar churn_type/
    economic_loss coherentes entre si, sin depender de la base de datos."""
    df = load_and_prepare()
    assert set(df["churn_type"].unique()) == {"softchurn", "midchurn", "hardchurn"}

    expected_loss = df["churn_type"].map({"softchurn": 500, "midchurn": 1000, "hardchurn": 2000})
    assert (df["economic_loss"] == expected_loss).all()

    # classify_churn_type es una funcion pura: mismo value_score -> mismo resultado.
    assert classify_churn_type(0.39) == "softchurn"
    assert classify_churn_type(0.40) == "midchurn"
    assert classify_churn_type(0.60) == "midchurn"
    assert classify_churn_type(0.61) == "hardchurn"
