-- Esquema inicial de la base de datos.
-- Se ejecuta automaticamente al levantar el contenedor de PostgreSQL
-- (montado en /docker-entrypoint-initdb.d/ via docker-compose.yml).

-- ============================================================
-- Tabla: customers
-- Datos "crudos" del cliente, tal como llegan del CSV origen,
-- mas ChurnType/EconomicLoss/value_score
-- (reglas de negocio, no dependen de Exited).
-- ============================================================
CREATE TABLE IF NOT EXISTS customers (
    customer_id        BIGINT PRIMARY KEY,
    row_number         INTEGER,
    surname            VARCHAR(100),
    credit_score       INTEGER NOT NULL,
    geography          VARCHAR(50) NOT NULL,
    gender              VARCHAR(20) NOT NULL,
    age                INTEGER NOT NULL,
    tenure             INTEGER NOT NULL,
    balance            NUMERIC(14, 2) NOT NULL,
    num_of_products    INTEGER NOT NULL,
    has_cr_card        BOOLEAN NOT NULL,
    is_active_member   BOOLEAN NOT NULL,
    estimated_salary   NUMERIC(14, 2) NOT NULL,
    exited             BOOLEAN,               -- NULL para clientes nuevos que aun no tienen desenlace conocido
    value_score        NUMERIC(6, 4),          -- calculado en Fase 1 (src/business/churn_impact.py)
    churn_type         VARCHAR(20),            -- softchurn / midchurn / hardchurn
    economic_loss      NUMERIC(10, 2),         -- 500 / 1000 / 2000 EUR segun churn_type
    created_at         TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_customers_churn_type ON customers (churn_type);
CREATE INDEX IF NOT EXISTS idx_customers_geography ON customers (geography);

-- ============================================================
-- Tabla: predictions
-- Salida del modelo XGBoost + decision de campaña para cada
-- cliente. Se separa de "customers" porque:
--   - un cliente puede tener varias predicciones a lo largo del
--     tiempo (re-scoring periodico), y queremos conservar el
--     historico.
--   - separa claramente "datos del cliente" de "resultado del
--     modelo en un momento dado" (buena practica de MLOps).
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id           SERIAL PRIMARY KEY,
    customer_id             BIGINT NOT NULL REFERENCES customers (customer_id),
    churn_probability       NUMERIC(6, 5) NOT NULL,   -- P(Exited = 1), salida de XGBoost
    threshold_used          NUMERIC(4, 3) NOT NULL,   -- threshold aplicado en el momento de predecir
    churn_prediction        BOOLEAN NOT NULL,         -- churn_probability >= threshold_used
    churn_type              VARCHAR(20) NOT NULL,      -- copiado de customers en el momento de la prediccion
    economic_loss           NUMERIC(10, 2) NOT NULL,
    campaign_cost           NUMERIC(10, 2) NOT NULL DEFAULT 50.00,
    expected_avoided_loss    NUMERIC(10, 2) NOT NULL,
    expected_net_profit     NUMERIC(10, 2) NOT NULL,
    campaign_recommendation BOOLEAN NOT NULL,          -- expected_net_profit > 0
    model_version           VARCHAR(50),                -- ej. 'xgboost_v1_2026-09-18'
    predicted_at            TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON predictions (customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions (predicted_at);