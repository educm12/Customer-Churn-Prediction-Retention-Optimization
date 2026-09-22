"""
Configuracion centralizada del proyecto.

Todas las constantes que tiene sentido poder ajustar sin tocar logica
de negocio o de modelado viven aqui: parametros economicos (coste de
campaña, tasa de retencion, perdida por ChurnType), la regla de
ChurnType (pesos y umbrales), y parametros de entrenamiento del modelo
(splits, CV, Optuna, rutas de artefactos).

Cambiar un valor aqui y volver a ejecutar el script correspondiente
(ver README, seccion "Como ejecutar") basta para propagar el cambio a
todo el proyecto — nada queda duplicado o hardcodeado en otros modulos.

El resto de modulos importan estas constantes en vez de definirlas por
su cuenta (ej. `from src.config import CAMPAIGN_COST`), asi que este
archivo es la UNICA fuente de verdad.
"""

import numpy as np

# ============================================================
# Economia de campañas de retencion
# ============================================================
CAMPAIGN_COST = 50.0            # EUR, coste de enviar una campaña a un cliente
RETENTION_SUCCESS_PROB = 0.30   # P(el cliente se queda | recibe campaña)

# Perdida economica (EUR) segun ChurnType, si el cliente abandona.
ECONOMIC_LOSS = {
    "softchurn": 500,
    "midchurn": 1000,
    "hardchurn": 2000,
}

# ============================================================
# Regla de negocio de ChurnType
# ============================================================
# Pesos del value_score (deben sumar 1.0)
CHURN_TYPE_WEIGHTS = {"balance": 0.40, "salary": 0.30, "products": 0.20, "tenure": 0.10}

# Umbrales de clasificacion sobre value_score (bandas fijas, no percentiles)
CHURN_TYPE_SOFT_THRESHOLD = 0.4   # value_score < este valor -> softchurn
CHURN_TYPE_HARD_THRESHOLD = 0.6   # value_score > este valor -> hardchurn (entre medias -> midchurn)

# ============================================================
# Entrenamiento del modelo 
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 100

MODEL_PATH = "models/xgboost_model.pkl"
MODEL_METADATA_PATH = "models/model_metadata.json"
OPTUNA_TRIALS_PATH = "reports/optuna_trials.csv"

# ============================================================
# Analisis de thresholds 
# ============================================================
THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]  # 0.10, 0.15, ..., 0.90
CONFUSION_MATRIX_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# Este es el threshold que maximica F1 por el que el modelo predice si el 
# cliente se queda o no.
# No es el threshold que decide si se envia la campaña al cliente, ya que eso
# se decide si el beneficio neto:
# RETENTION_SUCCESS_PROB * P(Churn) * ECONOMIC_LOSS >= CAMPAIGN_COST 
BUSINESS_THRESHOLD = 0.65
