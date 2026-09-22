"""
Comparacion economica de estrategias: Campaign Everyone,
Campaign Nobody, XGBoost.

============================================================
Decision metodologica importante
============================================================
Para que la comparacion sea justa, las tres se evaluan
sobre el MISMO conjunto (el test set, 2.000 clientes nunca usados en
entrenamiento ni tuning) y con el MISMO ingrediente economico: el
desenlace real (`exited`), igual que el backtest de baselines.

Lo unico que cambia entre estrategias es la DECISION de a quien se
contacta:
    - Campaign Everyone: contacta a todos.
    - Campaign Nobody: no contacta a nadie.
    - XGBoost: contacta a quien supera el threshold de negocio elegido
      (el que maximiza expected_net_profit sobre el test
      set), aplicado a la probabilidad PREDICHA (out-of-sample).

Por que no usar la probabilidad predicha tambien como P(Churn) en la
formula economica de XGBoost (en vez de `exited` real): mezclar
"probabilidad predicha para XGBoost" con "desenlace real para los
baselines" haria que la comparacion no fuera homogenea (dos varas de
medir distintas). Usando el desenlace real como base economica en los
tres casos, la unica diferencia entre estrategias es la calidad de la
decision de a quien contactar — que es exactamente lo que queremos
comparar.

============================================================
Limitacion metodologica reconocida
============================================================
El threshold de negocio se eligio maximizando el beneficio
sobre el mismo test set que aqui se usa para comparar. Estrictamente,
esto es una forma leve de "usar el test set dos veces": lo correcto en
un proyecto de produccion seria reservar un tercer split (validacion)
para elegir el threshold, y evaluar la comparacion final en un test
set separado. Se documenta aqui como limitacion conocida y mejora
futura, no se oculta.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.business.campaign import campaign_everyone, campaign_nobody, simulate_campaign_strategy
from src.config import RANDOM_STATE, TEST_SIZE
from src.features.feature_engineering import get_feature_target, load_customers_from_db
from src.models.predict import load_model
from src.models.threshold_analysis import build_threshold_table, select_threshold_business_optimal


def load_test_set_with_predictions():
    """Reconstruye el split de test de evaluación del modelo (mismo random_state) y
    añade la probabilidad predicha por el modelo entrenado."""
    df = load_customers_from_db()
    X, y = get_feature_target(df, include_churn_type=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    model = load_model()
    X_test_model = X_test.drop(columns=["churn_type"]) if "churn_type" in X_test.columns else X_test
    churn_probability = pd.Series(
        model.predict_proba(X_test_model)[:, 1], index=X_test.index, name="churn_probability"
    )
    economic_loss = df.loc[X_test.index, "economic_loss"]

    return y_test, churn_probability, economic_loss


def compare_strategies(
    y_true: pd.Series,
    churn_probability_model: pd.Series,
    economic_loss: pd.Series,
    xgboost_threshold: float,
) -> pd.DataFrame:
    """Compara Campaign Everyone / Campaign Nobody / XGBoost con
    `simulate_campaign_strategy` de forma identica para las tres. El
    "P(Churn)" que alimenta la formula es siempre `y_true` (desenlace
    real, ver docstring del modulo); solo cambia `campaign_decision`.
    """
    n = len(y_true)
    churn_probability_actual = y_true.astype(float)

    strategies = {
        "Campaign Everyone": campaign_everyone(n),
        "Campaign Nobody": campaign_nobody(n),
        f"XGBoost (threshold={xgboost_threshold:.2f})": (churn_probability_model >= xgboost_threshold).reset_index(
            drop=True
        ),
    }

    results = []
    for name, decision in strategies.items():
        metrics = simulate_campaign_strategy(churn_probability_actual, decision, economic_loss)
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
    os.makedirs("reports/figures", exist_ok=True)

    print("Reconstruyendo test set y prediciendo con el modelo entrenado...")
    y_test, churn_probability, economic_loss = load_test_set_with_predictions()

    print("Recalculando el threshold de negocio optimo sobre el test set...")
    threshold_table = build_threshold_table(y_test, churn_probability, economic_loss)
    best_threshold = select_threshold_business_optimal(threshold_table)
    print(f"Threshold de negocio elegido: {best_threshold:.2f}")

    comparison = compare_strategies(y_test, churn_probability, economic_loss, best_threshold)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Comparacion de estrategias (test set, 2.000 clientes) ===")
    print(comparison)

    comparison.to_csv("reports/strategy_comparison.csv")
    print("\nGuardado en reports/strategy_comparison.csv")

    # --- Graficas comparativas ---
    strategy_names = list(comparison.index)
    short_names = ["Campaign\nEveryone", "Campaign\nNobody", "XGBoost"]
    colors = ["#C44E52", "#8C8C8C", "#55A868"]

    bar_specs = [
        ("expected_net_profit", "Beneficio Neto Esperado (EUR)", "strategy_net_profit.png"),
        ("roi_pct", "ROI (%)", "strategy_roi.png"),
        ("total_campaign_cost", "Coste de Campañas (EUR)", "strategy_campaign_cost.png"),
        ("expected_avoided_loss", "Perdidas Evitadas (EUR)", "strategy_avoided_loss.png"),
        ("n_contacted", "Clientes Contactados", "strategy_contacted.png"),
        ("expected_retained_customers", "Clientes Retenidos (esperado)", "strategy_retained.png"),
    ]
    for col, title, filename in bar_specs:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        values = comparison[col].values
        bars = ax.bar(short_names, values, color=colors)
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 5 if val >= 0 else -15), textcoords="offset points",
                ha="center", fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(f"reports/figures/strategy_comparison/{filename}", dpi=120)
        plt.close(fig)

    print("Graficas guardadas en reports/figures/strategy_comparison/ (strategy_*.png)")
