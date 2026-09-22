"""
Explicabilidad del modelo: feature importance global de
XGBoost y SHAP (global y por cliente).

Se trabaja siempre sobre el pipeline COMPLETO (preprocesado + modelo)
guardado, para que el analisis use exactamente las mismas
transformaciones que en produccion.
"""

import matplotlib

matplotlib.use("Agg")  # backend no interactivo: solo generamos y guardamos figuras

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.features.feature_engineering import get_feature_names
from src.models.predict import load_model


def get_feature_importance(model=None) -> pd.DataFrame:
    """Feature importance global de XGBoost (gain), con los nombres de
    columna ya legibles (tras el One-Hot Encoding del preprocesado)."""
    model = model if model is not None else load_model()
    importances = model.named_steps["model"].feature_importances_
    names = get_feature_names(model)
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def build_shap_explainer(model=None) -> shap.TreeExplainer:
    """TreeExplainer sobre el estimador XGBoost interno del pipeline
    (SHAP necesita el modelo de arboles, no el pipeline completo)."""
    model = model if model is not None else load_model()
    return shap.TreeExplainer(model.named_steps["model"])


def compute_shap_explanation(X: pd.DataFrame, model=None) -> shap.Explanation:
    """Calcula los valores SHAP para un DataFrame de clientes (en su
    forma original, antes del One-Hot). Aplica el mismo preprocesado
    del pipeline antes de pasarlo al explainer, y anota los nombres de
    feature legibles en el resultado."""
    model = model if model is not None else load_model()
    explainer = build_shap_explainer(model)
    X_transformed = model.named_steps["preprocessing"].transform(X)
    explanation = explainer(X_transformed)
    explanation.feature_names = get_feature_names(model)
    return explanation


# ------------------------------------------------------------------
# Graficas
# ------------------------------------------------------------------
def plot_feature_importance(importance_df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_df = importance_df.sort_values("importance")
    ax.barh(sorted_df["feature"], sorted_df["importance"], color="#4C72B0")
    ax.set_xlabel("Importancia (gain)")
    ax.set_title("Feature Importance global — XGBoost")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def plot_shap_summary(explanation: shap.Explanation, path: str) -> None:
    """SHAP summary plot (beeswarm): impacto de cada feature sobre la
    prediccion, para todos los clientes del conjunto pasado."""
    fig = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_shap_waterfall(explanation: shap.Explanation, index: int, path: str) -> None:
    """SHAP waterfall plot para UN cliente concreto (por posicion
    dentro del `explanation` pasado, no por customer_id)."""
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation[index], show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os

    from sklearn.model_selection import train_test_split

    from src.config import RANDOM_STATE, TEST_SIZE
    from src.features.feature_engineering import get_feature_target, load_customers_from_db
    from src.models.predict import predict_churn_probability

    os.makedirs("reports/figures/explainability", exist_ok=True)

    print("Cargando modelo y reconstruyendo test set...")
    model = load_model()
    df = load_customers_from_db()
    X, y = get_feature_target(df, include_churn_type=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_test_model = X_test.drop(columns=["churn_type"]) if "churn_type" in X_test.columns else X_test

    print("\n=== Feature importance global (XGBoost, gain) ===")
    importance_df = get_feature_importance(model)
    print(importance_df.to_string(index=False))
    importance_df.to_csv("reports/feature_importance.csv", index=False)
    plot_feature_importance(importance_df, "reports/figures/explainability/feature_importance.png")

    print("\nCalculando valores SHAP sobre el test set (2.000 clientes)...")
    explanation = compute_shap_explanation(X_test_model, model=model)

    plot_shap_summary(explanation, "reports/figures/explainability/shap_summary.png")
    print("SHAP summary plot guardado en reports/figures/explainability/shap_summary.png")

    # Waterfall para dos clientes representativos: el de MAYOR y MENOR
    # probabilidad de churn predicha, para ilustrar ambos extremos.
    proba_test = predict_churn_probability(X_test_model, model=model)
    idx_high = proba_test.values.argmax()
    idx_low = proba_test.values.argmin()

    customer_id_high = X_test.index[idx_high]
    customer_id_low = X_test.index[idx_low]
    print(f"\nCliente de MAYOR riesgo (fila {idx_high}, id interno {customer_id_high}): "
          f"P(Churn)={proba_test.iloc[idx_high]:.4f}")
    print(f"Cliente de MENOR riesgo (fila {idx_low}, id interno {customer_id_low}): "
          f"P(Churn)={proba_test.iloc[idx_low]:.4f}")

    plot_shap_waterfall(explanation, idx_high, "reports/figures/explainability/shap_waterfall_high_risk.png")
    plot_shap_waterfall(explanation, idx_low, "reports/figures/explainability/shap_waterfall_low_risk.png")
    print("SHAP waterfall plots guardados en reports/figures/explainability/shap_waterfall_high_risk.png y _low_risk.png")
