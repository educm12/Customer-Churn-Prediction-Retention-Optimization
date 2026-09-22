"""
Economia de campañas de retencion: simulacion y estrategias baseline.

============================================================
Derivacion de la formula
============================================================
Para un cliente con probabilidad de abandono `P(Churn)`, perdida
economica `L` (segun ChurnType) y coste de campaña `C`:

    Expected Avoided Loss = P(Churn) x P(Retencion | Campaña) x L
    Expected Net Profit    = Expected Avoided Loss - C

Por que es correcta: la perdida solo se evita en la fraccion de casos
en los que el cliente IBA a abandonar (P(Churn)) Y la campaña tiene
exito reteniendolo (P(Retencion|Campaña), configurable via
RETENTION_SUCCESS_PROB en src/config.py). Multiplicar ambas
probabilidades da la probabilidad conjunta de "evento evitado", que al
multiplicarse por la perdida L da el valor esperado en EUR. Restar el
coste de campaña da el beneficio neto esperado de la intervencion.

Propiedad importante (autorregulacion): si P(Churn) es bajo, Expected
Avoided Loss tiende a 0 y Expected Net Profit se vuelve negativo
(-C), penalizando automaticamente contactar a clientes de bajo riesgo,
sin necesidad de reglas adicionales.

------------------------------------------------------------
OJO: Expected Net Profit NO es el resultado economico total
------------------------------------------------------------
`Expected Net Profit` (avoided_loss - cost) mide el VALOR AÑADIDO de
lanzar la campaña frente a no hacer nada. No es el balance economico
completo: no incluye el dinero que el banco pierde por los clientes
que abandonan de todas formas (porque no se les contacto, o porque se
les contacto pero la campaña no funciono).

Por eso `simulate_campaign_strategy` calcula tambien:

    Expected Realized Loss = Expected Loss (sin campaña) - Expected Avoided Loss
    Total Economic Impact  = -(Expected Realized Loss + Campaign Cost)

`Total Economic Impact` es el balance real para el banco (negativo =
dinero perdido en total). Con esta metrica, "Campaign Nobody" YA NO
es "gratis": su Total Economic Impact es igual a toda la perdida
esperada por churn de la cartera (nadie se retiene, todo se pierde),
mientras que `Expected Net Profit` de Campaign Nobody sigue siendo 0
por definicion (no hay avoided loss ni coste, luego no hay "valor
añadido" que medir frente a si misma).

============================================================
Sobre P(Churn) en la evaluación de Baselines
============================================================
Los baselines ("Campaign Everyone" / "Campaign Nobody") NO usan un
modelo de prediccion: deciden la campaña igual para todos los
clientes, sin mirar riesgo. Para poder comparar su resultado
economico de forma realista, esta fase hace un BACKTEST: usa el
desenlace real historico (`exited`, 0 o 1) como P(Churn) en la
formula, es decir, pregunta "¿que habria pasado economicamente si
hubieramos aplicado esta estrategia a la cartera real, sabiendo (a
toro pasado) quien abandono?".

Posteriormente, la misma funcion `simulate_campaign_strategy`
se reutilizara con la probabilidad PREDICHA por XGBoost sobre el
conjunto de test (no el desenlace real), para una evaluacion fuera de
muestra genuina.
"""

import pandas as pd

from src.config import CAMPAIGN_COST, RETENTION_SUCCESS_PROB

# CAMPAIGN_COST y RETENTION_SUCCESS_PROB se definen en src/config.py
# (unica fuente de verdad); se re-exponen aqui con el mismo nombre por
# compatibilidad con el resto del modulo y con los tests existentes.


# ------------------------------------------------------------------
# Formula economica
# ------------------------------------------------------------------
def compute_expected_avoided_loss(
    churn_probability: pd.Series,
    economic_loss: pd.Series,
    retention_prob: float = RETENTION_SUCCESS_PROB,
) -> pd.Series:
    """Expected Avoided Loss = P(Churn) x P(Retencion|Campaña) x L, por cliente."""
    return churn_probability * retention_prob * economic_loss


def compute_expected_net_profit(
    expected_avoided_loss: pd.Series, campaign_cost: float = CAMPAIGN_COST
) -> pd.Series:
    """Expected Net Profit = Expected Avoided Loss - Campaign Cost, por cliente."""
    return expected_avoided_loss - campaign_cost


# ------------------------------------------------------------------
# Estrategias de decision de campaña (baselines)
# ------------------------------------------------------------------
def campaign_everyone(n_customers: int) -> pd.Series:
    """Baseline 1: enviar campaña a TODOS los clientes."""
    return pd.Series([True] * n_customers)


def campaign_nobody(n_customers: int) -> pd.Series:
    """Baseline 2: no enviar campaña a NINGUN cliente."""
    return pd.Series([False] * n_customers)


# ------------------------------------------------------------------
# Simulacion economica (funcion unica, reutilizada por baselines,
# threshold analysisy XGBoost)
# ------------------------------------------------------------------
def simulate_campaign_strategy(
    churn_probability: pd.Series,
    campaign_decision: pd.Series,
    economic_loss: pd.Series,
    campaign_cost: float = CAMPAIGN_COST,
    retention_prob: float = RETENTION_SUCCESS_PROB,
) -> dict:
    """Simula el resultado economico de aplicar una decision de
    campaña sobre una cartera de clientes.

    Parameters
    ----------
    churn_probability:
        P(Exited=1) por cliente. En los baselines se usa el
        desenlace real (`exited`, 0/1) como backtest; Cuando tengamos el modelo sera
        la probabilidad predicha por XGBoost.
    campaign_decision:
        Serie booleana: True si el cliente recibe campaña.
    economic_loss:
        Impacto economico (EUR) si el cliente abandona (segun ChurnType).

    Returns
    -------
    dict con las metricas.
    """
    churn_probability = churn_probability.reset_index(drop=True)
    campaign_decision = campaign_decision.reset_index(drop=True).astype(bool)
    economic_loss = economic_loss.reset_index(drop=True)

    n_total = len(churn_probability)
    n_contacted = int(campaign_decision.sum())
    pct_contacted = (n_contacted / n_total * 100) if n_total else 0.0
    total_campaign_cost = n_contacted * campaign_cost

    avoided_loss_per_customer = compute_expected_avoided_loss(
        churn_probability, economic_loss, retention_prob
    )
    # La perdida solo se evita en los clientes que SI reciben campaña.
    total_expected_avoided_loss = float((avoided_loss_per_customer * campaign_decision).sum())

    retained_per_customer = churn_probability * retention_prob  # cantidad, no EUR
    total_expected_retained = float((retained_per_customer * campaign_decision).sum())

    expected_churners_total = float(churn_probability.sum())
    expected_loss_total_portfolio = float((churn_probability * economic_loss).sum())

    # Dinero que SI se pierde con esta estrategia (clientes no contactados
    # que abandonan + clientes contactados en los que la campaña no
    # funciona). Se deriva de forma exacta como el complementario del
    # avoided loss: lo que no se evita, se pierde.
    expected_realized_loss = expected_loss_total_portfolio - total_expected_avoided_loss

    expected_net_profit = total_expected_avoided_loss - total_campaign_cost
    roi_pct = (expected_net_profit / total_campaign_cost * 100) if total_campaign_cost > 0 else 0.0
    cost_per_retained = (
        total_campaign_cost / total_expected_retained if total_expected_retained > 0 else float("nan")
    )

    # Balance economico TOTAL para el banco: dinero perdido por churn no
    # evitado + dinero gastado en campañas. Es negativo por convencion
    # (dinero que sale de caja). A diferencia de expected_net_profit,
    # esta metrica SI refleja que "no hacer nada" tiene un coste real.
    total_economic_impact = -(expected_realized_loss + total_campaign_cost)

    return {
        "n_total_customers": n_total,
        "n_contacted": n_contacted,
        "pct_contacted": round(pct_contacted, 2),
        "total_campaign_cost": round(total_campaign_cost, 2),
        "expected_churners_total_portfolio": round(expected_churners_total, 2),
        "expected_loss_total_portfolio": round(expected_loss_total_portfolio, 2),
        "expected_avoided_loss": round(total_expected_avoided_loss, 2),
        "expected_realized_loss": round(expected_realized_loss, 2),
        "expected_retained_customers": round(total_expected_retained, 2),
        "expected_net_profit": round(expected_net_profit, 2),
        "total_economic_impact": round(total_economic_impact, 2),
        "roi_pct": round(roi_pct, 2),
        "cost_per_retained_customer": (
            round(cost_per_retained, 2) if cost_per_retained == cost_per_retained else None
        ),  # NaN != NaN -> None si no hay retenidos
    }


# ------------------------------------------------------------------
# Comparacion de los dos baselines
# ------------------------------------------------------------------
def compare_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compara 'Campaign Everyone' vs 'Campaign Nobody' sobre un
    DataFrame de clientes con columnas `exited` y `economic_loss`
    (backtest con desenlace real, ver docstring del modulo).
    """
    n = len(df)
    churn_probability = df["exited"].astype(float)
    economic_loss = df["economic_loss"].astype(float)

    strategies = {
        "Campaign Everyone": campaign_everyone(n),
        "Campaign Nobody": campaign_nobody(n),
    }

    results = []
    for name, decision in strategies.items():
        metrics = simulate_campaign_strategy(churn_probability, decision, economic_loss)
        metrics["strategy"] = name
        results.append(metrics)

    result_df = pd.DataFrame(results).set_index("strategy")
    return result_df[
        [
            "n_total_customers", "n_contacted", "pct_contacted", "total_campaign_cost",
            "expected_churners_total_portfolio", "expected_loss_total_portfolio",
            "expected_avoided_loss", "expected_realized_loss", "expected_retained_customers",
            "expected_net_profit", "total_economic_impact", "roi_pct", "cost_per_retained_customer",
        ]
    ]


if __name__ == "__main__":
    from src.features.feature_engineering import load_customers_from_db

    df = load_customers_from_db()
    comparison = compare_baselines(df)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(comparison)

    import os
    os.makedirs("reports", exist_ok=True)
    comparison.to_csv("reports/baseline_comparison.csv")
    print("\nGuardado en reports/baseline_comparison.csv")
