"""
Conexion a PostgreSQL.

Provee un engine de SQLAlchemy reutilizable en el resto del proyecto
(ETL, API, notebooks). Las credenciales se leen de variables de entorno
(.env), nunca hardcodeadas.

Uso:
    from src.data.database import get_engine
    engine = get_engine()
    df.to_sql("customers", engine, if_exists="append", index=False)
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def _default_host() -> str:
    """Host por defecto para conexiones fuera de Docker.

    Se usa 127.0.0.1 en vez de 'localhost' ya que suele dar menos problemas
    al usar Windows y Docker Desktop."""
    return os.getenv("POSTGRES_HOST", "127.0.0.1")


def _build_connection_url(use_docker_host: bool = False) -> str:
    """Construye la URL de conexion a partir de variables de entorno.

    Parameters
    ----------
    use_docker_host:
        Si True, usa POSTGRES_HOST_DOCKER (nombre del servicio dentro de
        la red de docker-compose) en lugar de POSTGRES_HOST (127.0.0.1
        por defecto). Se activa automaticamente si detectamos que
        corremos dentro de un contenedor Docker (ver
        `_running_inside_docker`).
    """
    user = os.getenv("POSTGRES_USER", "churn_user")
    password = os.getenv("POSTGRES_PASSWORD", "churn_pass")
    db = os.getenv("POSTGRES_DB", "churn_db")
    port = os.getenv("POSTGRES_PORT", "5432")

    if use_docker_host:
        host = os.getenv("POSTGRES_HOST_DOCKER", "postgres")
    else:
        host = _default_host()

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def _running_inside_docker() -> bool:
    """Heuristica simple para saber si el codigo corre dentro de un
    contenedor Docker (y por tanto debe usar el hostname del servicio
    'postgres' en vez de 'localhost')."""
    return os.path.exists("/.dockerenv")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Devuelve un engine de SQLAlchemy (cacheado: una sola instancia
    por proceso, tal como recomienda SQLAlchemy).
    """
    url = _build_connection_url(use_docker_host=_running_inside_docker())
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=280,
        connect_args={"connect_timeout": 5},
    )


def check_connection() -> bool:
    """Comprueba que la conexion a PostgreSQL funciona. Util para el
    endpoint GET /health de FastAPI y para tests de integracion."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    ok = check_connection()
    print("Conexion a PostgreSQL:", "OK" if ok else "FALLO")