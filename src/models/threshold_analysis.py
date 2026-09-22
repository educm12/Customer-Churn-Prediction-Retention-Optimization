"""
Analisis de thresholds.

Combina dos capas de evaluacion para cada threshold de clasificacion:
    - Metricas de ML (src/models/evaluate.py): Precision, Recall, F1,
      Specificity, matriz de confusion.
    - Metricas economicas (src/business/campaign.py): coste de campaña,
      perdida evitada, perdida realizada, beneficio neto, ROI.

IMPORTANTE: aqui `P(Churn)` es la probabilidad PREDICHA por XGBoost 
sobre el test set (out-of-sample real), no el desenlace historico real 
como en el backtest de baselines. Esta es la evaluacion economica "de verdad": 
que pasaria si desplegasemos el modelo sobre clientes nuevos.
"""

import numpy as np
import pandas as pd

from src.business.campaign import simulate_campaign_strategy
from src.config import CONFUSION_MATRIX_THRESHOLDS, THRESHOLD_GRID
from src.models.evaluate import threshold_sweep

DEFAULT_THRESHOLDS = THRESHOLD_GRID  # alias local por compatibilidad; fuente: src/config.py


def build_threshold_table(
    y_true: pd.Series,
    churn_probability: pd.Series,
    economic_loss: pd.Series,
    thresholds: list[float] = None,
) -> pd.DataFrame:
    """Tabla unica con metricas de ML + metricas economicas para cada
    threshold."""
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS

    ml_metrics = threshold_sweep(y_true, churn_probability.values, thresholds)

    econ_rows = []
    for t in thresholds:
        decision = churn_probability >= t
        econ = simulate_campaign_strategy(churn_probability, decision, economic_loss)
        econ_rows.append(econ)
    econ_metrics = pd.DataFrame(econ_rows).drop(columns=["n_total_customers", "n_contacted"])

    combined = pd.concat([ml_metrics.reset_index(drop=True), econ_metrics.reset_index(drop=True)], axis=1)
    return combined


# ------------------------------------------------------------------
# Seleccion de threshold segun distintos objetivos
# ------------------------------------------------------------------
def select_threshold_max_f1(table: pd.DataFrame) -> float:
    """Threshold orientado a F1: mejor equilibrio precision/recall."""
    return float(table.loc[table["f1"].idxmax(), "threshold"])


def select_threshold_min_recall(table: pd.DataFrame, min_recall: float = 0.80) -> float:
    """Threshold orientado a Recall: el MAYOR threshold que aun
    garantiza `recall >= min_recall`. Minimiza clientes que abandonan
    sin recibir campaña, sin irse trivialmente al threshold mas bajo
    posible (lo que dispararia el numero de falsos positivos)."""
    candidates = table[table["recall"] >= min_recall]
    if candidates.empty:
        # Ningun threshold alcanza el recall pedido: se devuelve el que
        # da el recall mas alto disponible (el threshold mas bajo).
        return float(table.loc[table["recall"].idxmax(), "threshold"])
    return float(candidates["threshold"].max())


def select_threshold_min_precision(table: pd.DataFrame, min_precision: float = 0.50) -> float:
    """Threshold orientado a Precision: el MENOR threshold que aun
    garantiza `precision >= min_precision`. Evita gastar campañas en
    clientes que finalmente no abandonan, sin irse trivialmente al
    threshold mas alto posible (lo que dispararia los falsos negativos)."""
    candidates = table[table["precision"] >= min_precision]
    if candidates.empty:
        return float(table.loc[table["precision"].idxmax(), "threshold"])
    return float(candidates["threshold"].min())


def select_threshold_business_optimal(table: pd.DataFrame) -> float:
    """Threshold orientado a negocio: el que maximiza el beneficio neto
    esperado (`expected_net_profit`), equivalente a minimizar la
    perdida realizada + coste de campaña (`total_economic_impact`)."""
    return float(table.loc[table["expected_net_profit"].idxmax(), "threshold"])


def summarize_threshold_selection(table: pd.DataFrame, min_recall: float = 0.80, min_precision: float = 0.50) -> pd.DataFrame:
    """Tabla resumen: un threshold por cada criterio de seleccion,
    con sus metricas ML y economicas asociadas."""
    selections = {
        f"Recall (>= {min_recall:.0%})": select_threshold_min_recall(table, min_recall),
        f"Precision (>= {min_precision:.0%})": select_threshold_min_precision(table, min_precision),
        "F1 (equilibrio)": select_threshold_max_f1(table),
        "Negocio (max beneficio)": select_threshold_business_optimal(table),
    }
    rows = []
    for criterion, threshold in selections.items():
        row = table[table["threshold"] == threshold].iloc[0].to_dict()
        row["criterion"] = criterion
        rows.append(row)
    result = pd.DataFrame(rows).set_index("criterion")
    cols = ["threshold", "precision", "recall", "f1", "specificity", "n_predicted_churn",
            "total_campaign_cost", "expected_net_profit", "total_economic_impact", "roi_pct"]
    return result[cols]


# ------------------------------------------------------------------
# Script de ejecucion: genera tabla, graficas y seleccion de thresholds
# sobre el test set real, usando probabilidades predichas
# (out-of-sample), no el desenlace historico.
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    from src.features.feature_engineering import get_feature_target, load_customers_from_db
    from src.models.predict import load_model

    os.makedirs("reports/figures", exist_ok=True)

    print("Cargando datos y reconstruyendo el split del dataset (mismo random_state)...")
    df = load_customers_from_db()
    X, y = get_feature_target(df, include_churn_type=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = load_model()
    # El modelo final se entreno SIN churn_type (ablation study).
    X_test_model = X_test.drop(columns=["churn_type"]) if "churn_type" in X_test.columns else X_test
    churn_probability = pd.Series(model.predict_proba(X_test_model)[:, 1], index=X_test.index, name="churn_probability")
    economic_loss = df.loc[X_test.index, "economic_loss"]

    print(f"Test set: {len(X_test)} clientes | tasa de churn real: {y_test.mean():.4f}")

    # --- Tabla completa (ML + economico) ---
    table = build_threshold_table(y_test, churn_probability, economic_loss)
    table.to_csv("reports/threshold_analysis.csv", index=False)
    print("\nTabla completa guardada en reports/threshold_analysis.csv")
    print(table[["threshold", "precision", "recall", "f1", "specificity", "n_predicted_churn"]].to_string(index=False))

    # --- Seleccion de threshold por objetivo ---
    summary = summarize_threshold_selection(table)
    summary.to_csv("reports/threshold_selection_summary.csv")
    print("\n=== Seleccion de threshold por objetivo ===")
    print(summary.to_string())

    # --- Graficas individuales: metrica vs threshold ---
    plot_specs = [
        ("precision", "Precision vs Threshold", "Precision"),
        ("recall", "Recall vs Threshold", "Recall"),
        ("f1", "F1 vs Threshold", "F1"),
        ("specificity", "Specificity vs Threshold", "Specificity"),
        ("n_predicted_churn", "Numero de clientes contactados vs Threshold", "Clientes contactados"),
    ]
    for col, title, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(table["threshold"], table[col], marker="o", color="#4C72B0")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"reports/figures/threshold_analysis/threshold_{col}.png", dpi=120)
        plt.close(fig)

    # --- Precision-Recall vs Threshold (overlay) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["threshold"], table["precision"], marker="o", label="Precision", color="#4C72B0")
    ax.plot(table["threshold"], table["recall"], marker="s", label="Recall", color="#DD8452")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Valor")
    ax.set_title("Precision y Recall vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/threshold_analysis/threshold_precision_recall.png", dpi=120)
    plt.close(fig)

    # --- Threshold vs Expected Net Profit y vs ROI ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["threshold"], table["expected_net_profit"], marker="o", color="#55A868")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected Net Profit (EUR)")
    ax.set_title("Threshold vs Expected Net Profit")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/threshold_analysis/threshold_net_profit.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["threshold"], table["roi_pct"], marker="o", color="#C44E52")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("ROI (%)")
    ax.set_title("Threshold vs ROI")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/threshold_analysis/threshold_roi.png", dpi=120)
    plt.close(fig)

    # --- Matrices de confusion para thresholds representativos ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, t in zip(axes.flatten(), CONFUSION_MATRIX_THRESHOLDS):
        y_pred = (churn_probability >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No churn", "Churn"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No churn", "Churn"])
        ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
        ax.set_title(f"Threshold = {t}")
    plt.tight_layout()
    plt.savefig("reports/figures/threshold_analysis/threshold_confusion_matrices.png", dpi=120)
    plt.close(fig)

    print("\nGraficas guardadas en reports/figures/threshold_analysis/")
    print("threshold_precision.png, threshold_recall.png, threshold_f1.png,")
    print("threshold_specificity.png, threshold_n_predicted_churn.png,")
    print("threshold_precision_recall.png, threshold_net_profit.png, threshold_roi.png,")
    print("threshold_confusion_matrices.png")
