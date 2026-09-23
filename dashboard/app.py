"""
Dashboard de Streamlit.

La ficha de cliente y la prediccion (Tab "Cliente") ya NO acceden a la
base de datos ni al modelo en local: hablan por HTTP con los routers
de FastAPI (api/routes/customers.py, predict.py, health.py), igual
que haria cualquier otro cliente de la API:
    - GET  /customers/{id}              -> obtener_cliente()
    - GET  /customers/{id}/prediction   -> obtener_prediccion_cliente()
    - POST /predict                     -> predecir()
    - GET  /health                      -> comprobar_salud()

Requiere que el backend este arrancado (uvicorn api.main:app) y
accesible en BACKEND_URL (por defecto http://localhost:8000).

El resto de tabs (Explicabilidad, Evaluacion del Modelo, Analisis de
Negocio) siguen usando el modelo/datos en local porque no hay
endpoints para SHAP, curvas ROC/PR, threshold analysis ni los reports
CSV -- solo se expuso la logica de scoring de un cliente:
    - src.models.explainability
    - src.models.evaluate
    - reports/*.csv ya generados por los scripts de cada fase

Ejecucion:
    BACKEND_URL=http://localhost:8000 streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

# Streamlit ejecuta este script sin añadir la raiz del proyecto a
# sys.path (a diferencia de `python -m src.xxx`, que si lo hace). Sin
# esto, `from src...` fallaria con ModuleNotFoundError en cuanto se
# ejecute via `streamlit run dashboard/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.config import OPERATIVE_THRESHOLD, CAMPAIGN_COST, ECONOMIC_LOSS, RANDOM_STATE, RETENTION_SUCCESS_PROB, TEST_SIZE
from src.features.feature_engineering import get_feature_target, load_customers_from_db
from src.models.evaluate import compute_threshold_metrics, to_clean_arrays
from src.models.explainability import build_shap_explainer, compute_shap_explanation, get_feature_importance
from src.models.predict import load_model, predict_churn_probability

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="🏦", layout="wide")

# ============================================================
# Cliente HTTP del backend FastAPI (api/routes/{customers,predict,health}.py)
# ============================================================
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def obtener_cliente(customer_id: int) -> dict:
    """GET /customers/{customer_id}."""
    resp = requests.get(f"{BACKEND_URL}/customers/{customer_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def obtener_prediccion_cliente(customer_id: int) -> dict:
    """GET /customers/{customer_id}/prediction."""
    resp = requests.get(f"{BACKEND_URL}/customers/{customer_id}/prediction", timeout=30)
    resp.raise_for_status()
    return resp.json()


def predecir(features: dict, customer_id: int | None = None) -> dict:
    """POST /predict -- puntua un cliente (existente o hipotetico)."""
    payload = {**features, "customer_id": customer_id}
    resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def comprobar_salud() -> dict:
    """GET /health -- conexion a PostgreSQL y modelo cargado en el backend."""
    resp = requests.get(f"{BACKEND_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ============================================================
# Carga y cacheado de datos/modelo (evita recalcular en cada rerun)
# ============================================================
@st.cache_resource
def get_cached_model():
    return load_model()


@st.cache_data(ttl=3600)
def get_all_customers() -> pd.DataFrame:
    return load_customers_from_db()


@st.cache_data(ttl=3600)
def get_test_set(_model):
    """Reconstruye el split de test del entrenamiento (mismo random_state) y
    añade la probabilidad predicha. El guion bajo en `_model` le dice a
    Streamlit que no intente hashear el objeto modelo."""
    df = get_all_customers()
    X, y = get_feature_target(df, include_churn_type=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_test_model = X_test.drop(columns=["churn_type"])
    churn_probability = predict_churn_probability(X_test_model, model=_model)
    economic_loss = df.loc[X_test.index, "economic_loss"]
    return X_test_model, y_test, churn_probability, economic_loss


@st.cache_resource
def get_shap_explanation_for_test_set(_model, sample_size: int = 500):
    """SHAP sobre una muestra del test set (global), cacheado como
    'resource' porque un shap.Explanation no es un DataFrame trivial
    de hashear."""
    X_test_model, _, _, _ = get_test_set(_model)
    sample = X_test_model.sample(n=min(sample_size, len(X_test_model)), random_state=RANDOM_STATE)
    return compute_shap_explanation(sample, model=_model)


@st.cache_data(ttl=3600)
def load_report_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


model = get_cached_model()
customers_df = get_all_customers()

st.title("🏦 Customer Churn Prediction & Retention Campaign Dashboard")
st.caption(
    "XGBoost (riesgo de abandono) + reglas de negocio (impacto economico) + "
    "optimizacion de campañas de retencion — proyecto completo."
)

# ============================================================
# Sidebar: selector de cliente
# ============================================================
st.sidebar.header("Seleccionar cliente")
customer_ids = sorted(customers_df["customer_id"].tolist())
selected_customer_id = st.sidebar.selectbox(
    "Customer ID", customer_ids, index=0, help="Elige un cliente para ver su informacion y prediccion."
)
customer_row = customers_df[customers_df["customer_id"] == selected_customer_id].iloc[0]

st.sidebar.divider()
st.sidebar.metric("Total de clientes", f"{len(customers_df):,}")
st.sidebar.metric("Tasa de churn real (cartera)", f"{customers_df['exited'].mean():.1%}")
st.sidebar.caption(
    f"Threshold operativo actual: **{OPERATIVE_THRESHOLD}** · "
    f"Coste campaña: **{CAMPAIGN_COST}€** · "
    f"P(retencion): **{RETENTION_SUCCESS_PROB:.0%}**  \n"
    "(configurables en `src/config.py`)"
)

st.sidebar.divider()
try:
    health = comprobar_salud()
    if health["status"] == "ok":
        st.sidebar.success(f"Backend OK ({BACKEND_URL})")
    else:
        st.sidebar.warning(
            f"Backend degradado — DB: {'✅' if health['database'] else '❌'} · "
            f"Modelo: {'✅' if health['model_loaded'] else '❌'}"
        )
except requests.RequestException as exc:
    st.sidebar.error(f"No se pudo contactar con el backend ({BACKEND_URL}): {exc}")

# ============================================================
# Tabs principales
# ============================================================
tab_customer, tab_explain, tab_eval, tab_business = st.tabs(
    ["👤 Cliente", "🔍 Explicabilidad", "📊 Evaluacion del Modelo", "💰 Analisis de Negocio"]
)

# ------------------------------------------------------------------
# TAB 1: Cliente
# ------------------------------------------------------------------
with tab_customer:
    try:
        cliente = obtener_cliente(int(selected_customer_id))
        prediction = obtener_prediccion_cliente(int(selected_customer_id))
    except requests.RequestException as exc:
        st.error(f"No se pudo obtener el cliente/prediccion desde el backend: {exc}")
        st.stop()

    # Features en el formato que espera PredictRequest / lo que usa
    # SHAP mas abajo (Tab 2), reconstruidas a partir de la respuesta
    # del backend en vez de leerlas directamente del DataFrame local.
    features = {
        "credit_score": int(cliente["credit_score"]),
        "geography": cliente["geography"],
        "gender": cliente["gender"],
        "age": int(cliente["age"]),
        "tenure": int(cliente["tenure"]),
        "balance": float(cliente["balance"]),
        "num_of_products": int(cliente["num_of_products"]),
        "has_cr_card": bool(cliente["has_cr_card"]),
        "is_active_member": bool(cliente["is_active_member"]),
        "estimated_salary": float(cliente["estimated_salary"]),
    }

    col_info, col_pred = st.columns([1, 1])

    with col_info:
        st.subheader("Informacion del cliente")
        info_rows = [
            ("Geography", cliente["geography"]),
            ("Gender", cliente["gender"]),
            ("Age", int(cliente["age"])),
            ("Credit Score", int(cliente["credit_score"])),
            ("Balance", f"{float(cliente['balance']):,.2f} €"),
            ("Estimated Salary", f"{float(cliente['estimated_salary']):,.2f} €"),
            ("Num of Products", int(cliente["num_of_products"])),
            ("Tenure", f"{int(cliente['tenure'])} años"),
            ("Is Active Member", "Si" if cliente["is_active_member"] else "No"),
            ("Has Credit Card", "Si" if cliente["has_cr_card"] else "No"),
        ]
        for label, value in info_rows:
            st.write(f"**{label}:** {value}")

        if cliente["exited"] is not None:
            actual_label = "Abandono" if cliente["exited"] else "Permanece"
            st.info(f"Desenlace real (historico): **{actual_label}**")

    with col_pred:
        st.subheader("Prediccion")
        proba_pct = prediction["churn_probability"] * 100
        st.metric("Churn Probability", f"{proba_pct:.1f}%")
        st.progress(min(max(prediction["churn_probability"], 0.0), 1.0))

        churn_label = "CHURN" if prediction["churn_prediction"] == 1 else "NO CHURN"
        st.write(f"**Churn Prediction** (threshold={prediction['threshold_used']}): `{churn_label}`")

        churn_type_emoji = {"softchurn": "🟢", "midchurn": "🟡", "hardchurn": "🔴"}
        st.write(
            f"**ChurnType:** {churn_type_emoji.get(prediction['churn_type'], '')} "
            f"`{prediction['churn_type']}` → Economic Loss: **{prediction['economic_loss']:,.0f} €**"
        )

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Avoided Loss", f"{prediction['expected_avoided_loss']:,.2f} €")
        c2.metric("Campaign Cost", f"{prediction['campaign_cost']:,.2f} €")
        c3.metric("Expected Net Profit", f"{prediction['expected_net_profit']:,.2f} €")

        if prediction["campaign"]:
            st.success("✅ Recomendacion: ENVIAR CAMPAÑA (beneficio esperado positivo)")
        else:
            st.warning("⛔ Recomendacion: NO enviar campaña (beneficio esperado negativo)")

        st.divider()
        st.subheader("🧪 Simular cliente hipotetico")
        st.caption("Prueba /predict con datos manuales, sin que el cliente exista en la BD.")
        with st.form("simulador_predict"):
            sf1, sf2, sf3 = st.columns(3)
            sim_credit_score = sf1.number_input("Credit Score", 300, 900, int(cliente["credit_score"]))
            sim_geography = sf2.selectbox(
                "Geography", ["France", "Germany", "Spain"],
                index=["France", "Germany", "Spain"].index(cliente["geography"])
                if cliente["geography"] in ["France", "Germany", "Spain"] else 0,
            )
            sim_gender = sf3.selectbox(
                "Gender", ["Male", "Female"],
                index=["Male", "Female"].index(cliente["gender"]) if cliente["gender"] in ["Male", "Female"] else 0,
            )
            sf4, sf5, sf6 = st.columns(3)
            sim_age = sf4.number_input("Age", 18, 100, int(cliente["age"]))
            sim_tenure = sf5.number_input("Tenure", 0, 20, int(cliente["tenure"]))
            sim_num_products = sf6.number_input("Num of Products", 1, 4, int(cliente["num_of_products"]))
            sf7, sf8 = st.columns(2)
            sim_balance = sf7.number_input("Balance", 0.0, value=float(cliente["balance"]))
            sim_salary = sf8.number_input("Estimated Salary", 0.0, value=float(cliente["estimated_salary"]))
            sf9, sf10 = st.columns(2)
            sim_has_card = sf9.checkbox("Has Credit Card", value=bool(cliente["has_cr_card"]))
            sim_is_active = sf10.checkbox("Is Active Member", value=bool(cliente["is_active_member"]))
            simular = st.form_submit_button("Simular")

        if simular:
            try:
                sim_prediction = predecir(
                    {
                        "credit_score": int(sim_credit_score),
                        "geography": sim_geography,
                        "gender": sim_gender,
                        "age": int(sim_age),
                        "tenure": int(sim_tenure),
                        "balance": float(sim_balance),
                        "num_of_products": int(sim_num_products),
                        "has_cr_card": bool(sim_has_card),
                        "is_active_member": bool(sim_is_active),
                        "estimated_salary": float(sim_salary),
                    },
                    customer_id=None,
                )
                churn_label_sim = "CHURN" if sim_prediction["churn_prediction"] == 1 else "NO CHURN"
                st.write(
                    f"Churn Probability: **{sim_prediction['churn_probability']:.1%}** · "
                    f"Churn Prediction (threshold={sim_prediction['threshold_used']}): **{churn_label_sim}** · "
                )
                churn_type_emoji = {"softchurn": "🟢", "midchurn": "🟡", "hardchurn": "🔴"}
                st.write(
                    f"**ChurnType:** {churn_type_emoji.get(sim_prediction['churn_type'], '')} "
                    f"`{sim_prediction['churn_type']}` → Economic Loss: **{sim_prediction['economic_loss']:,.0f} €**"
                )
                
                st.divider()        
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected Avoided Loss", f"{sim_prediction['expected_avoided_loss']:,.2f} €")
                c2.metric("Campaign Cost", f"{sim_prediction['campaign_cost']:,.2f} €")
                c3.metric("Expected Net Profit", f"{sim_prediction['expected_net_profit']:,.2f} €")

                if sim_prediction["campaign"]:
                    st.success("✅ Recomendacion: ENVIAR CAMPAÑA (beneficio esperado positivo)")
                else:
                    st.warning("⛔ Recomendacion: NO enviar campaña (beneficio esperado negativo)")

            except requests.RequestException as exc:
                st.error(f"Error al llamar a /predict: {exc}")

# ------------------------------------------------------------------
# TAB 2: Explicabilidad
# ------------------------------------------------------------------
with tab_explain:
    st.subheader("Feature importance global (XGBoost)")
    importance_df = get_feature_importance(model)
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_df = importance_df.sort_values("importance")
    ax.barh(sorted_df["feature"], sorted_df["importance"], color="#4C72B0")
    ax.set_xlabel("Importancia (gain)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("SHAP summary (global, muestra de 500 clientes del test set)")
    with st.spinner("Calculando valores SHAP..."):
        explanation = get_shap_explanation_for_test_set(model)
    fig2 = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader(f"SHAP waterfall — cliente {selected_customer_id}")
    single_customer_features = pd.DataFrame([features])
    single_explanation = compute_shap_explanation(single_customer_features, model=model)
    fig3 = plt.figure(figsize=(9, 5))
    shap.plots.waterfall(single_explanation[0], show=False)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    st.caption(
        "Contribucion de cada feature a la prediccion de ESTE cliente concreto "
        "(base_value + suma de contribuciones = prediccion final, en escala log-odds)."
    )

# ------------------------------------------------------------------
# TAB 3: Evaluacion del Modelo
# ------------------------------------------------------------------
with tab_eval:
    X_test_model, y_test, churn_probability_test, economic_loss_test = get_test_set(model)

    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.subheader("ROC Curve")
        y_test_arr, proba_arr = to_clean_arrays(y_test, churn_probability_test)
        fpr, tpr, _ = roc_curve(y_test_arr, proba_arr)
        auc = roc_auc_score(y_test_arr, proba_arr)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, color="#4C72B0", label=f"XGBoost (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_pr:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test_arr, proba_arr)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(recall, precision, color="#DD8452")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Analisis de threshold (interactivo)")
    threshold = st.slider("Threshold de clasificacion", 0.05, 0.95, float(OPERATIVE_THRESHOLD), 0.05)
    metrics = compute_threshold_metrics(y_test, churn_probability_test.values, threshold)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{metrics['precision']:.1%}")
    m2.metric("Recall", f"{metrics['recall']:.1%}")
    m3.metric("F1", f"{metrics['f1']:.1%}")
    m4.metric("Specificity", f"{metrics['specificity']:.1%}")
    m5.metric("Clientes contactados", f"{metrics['n_predicted_churn']:,}")

    col_cm, col_table = st.columns([1, 1])
    with col_cm:
        cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No churn", "Churn"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No churn", "Churn"])
        ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusion (threshold={threshold})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_table:
        st.write("**Tabla de threshold analysis**")
        threshold_table = load_report_csv("reports/threshold_analysis.csv")
        if not threshold_table.empty:
            st.dataframe(
                threshold_table[["threshold", "precision", "recall", "f1", "specificity", "n_predicted_churn"]],
                height=380,
            )
        else:
            st.info("Ejecuta `python -m src.models.threshold_analysis` para generar esta tabla.")

# ------------------------------------------------------------------
# TAB 4: Analisis de Negocio
# ------------------------------------------------------------------
with tab_business:
    threshold_table = load_report_csv("reports/threshold_analysis.csv")
    strategy_table = load_report_csv("reports/strategy_comparison.csv")

    if not threshold_table.empty:
        st.subheader("Beneficio neto y ROI por threshold")
        col_profit, col_roi = st.columns(2)
        with col_profit:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(threshold_table["threshold"], threshold_table["expected_net_profit"], marker="o", color="#55A868")
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Threshold"); ax.set_ylabel("Expected Net Profit (EUR)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with col_roi:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(threshold_table["threshold"], threshold_table["roi_pct"], marker="o", color="#C44E52")
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Threshold"); ax.set_ylabel("ROI (%)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        best_row = threshold_table.loc[threshold_table["expected_net_profit"].idxmax()]
        st.success(
            f"Threshold optimo de negocio: **{best_row['threshold']}** → "
            f"Beneficio neto esperado: **{best_row['expected_net_profit']:,.0f} €** "
            f"(ROI: {best_row['roi_pct']:.1f}%)"
        )
    else:
        st.info("Ejecuta `python -m src.models.threshold_analysis` para generar esta seccion.")

    st.divider()

    if not strategy_table.empty:
        st.subheader("XGBoost vs Baselines (test set)")
        strategy_table = strategy_table.rename(columns={strategy_table.columns[0]: "strategy"})
        st.dataframe(strategy_table, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#C44E52" if "Everyone" in s else "#8C8C8C" if "Nobody" in s else "#55A868"
                      for s in strategy_table["strategy"]]
            ax.bar(strategy_table["strategy"], strategy_table["expected_net_profit"], color=colors)
            ax.set_ylabel("Expected Net Profit (EUR)")
            ax.tick_params(axis="x", rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with col_b:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(strategy_table["strategy"], strategy_table["roi_pct"], color=colors)
            ax.set_ylabel("ROI (%)")
            ax.tick_params(axis="x", rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Ejecuta `python -m src.business.strategy_comparison` para generar esta seccion.")

    st.divider()
    st.subheader("Economia de la cartera completa (backtest sobre 10.000 clientes)")
    baseline_table = load_report_csv("reports/baseline_comparison.csv")
    if not baseline_table.empty:
        baseline_table = baseline_table.rename(columns={baseline_table.columns[0]: "strategy"})
        st.dataframe(baseline_table, use_container_width=True)
    else:
        st.info("Ejecuta `python -m src.business.campaign` para generar esta seccion.")