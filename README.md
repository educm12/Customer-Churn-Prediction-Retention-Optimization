# Customer Churn Prediction & Retention Optimization

> **End-to-end Machine Learning project combining customer churn prediction, explainable AI and economic decision-making for banking retention campaigns.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit\&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?logo=postgresql\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?logo=docker\&logoColor=white)
![pytest](https://img.shields.io/badge/Testing-pytest-0A9EDC)

---

## Overview

This project implements a complete **customer churn prediction and retention decision system** for a banking dataset.

Instead of treating churn prediction as a purely classification problem, the system separates two different questions:

**Risk**

> How likely is this customer to churn?

**Business value**

> How much economic impact would losing this customer have?

The first question is addressed using an **XGBoost classification model**, while the second is handled through a separate business-rule layer that estimates customer value and potential economic loss.

Both signals are combined to support a **data-driven retention campaign decision**.

The system is designed as an end-to-end ML application, covering:

* Data ingestion
* Database storage
* Feature engineering
* Model training and hyperparameter optimization
* Threshold analysis
* Economic optimization
* Model explainability
* REST API
* Interactive dashboard
* Automated testing

---

# ML Pipeline

```text
                    ┌──────────────────────┐
                    │     Raw Dataset      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Ingestion     │
                    │      + ETL           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     PostgreSQL       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ + Preprocessing      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       XGBoost        │
                    │  Churn Prediction    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Churn Probability   │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │ Business Rules   │        │ Model            │
       │                  │        │ Explainability   │
       │ ChurnType        │        │                  │
       │ Economic Loss    │        │ SHAP             │
       └────────┬─────────┘        │ Feature          │
                │                  │ Importance       │
                │                  └──────────────────┘
                ▼
       ┌──────────────────┐
       │ Campaign         │
       │ Optimization     │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────────────┐
       │ FastAPI + Streamlit      │
       │ Production Interface     │
       └──────────────────────────┘
```

---

# Machine Learning

## Model

The project uses **XGBoost** for binary churn classification.

The model predicts:

```text
P(Churn) = P(Exited = 1 | customer features)
```

The training pipeline includes:

* Stratified train/test split
* Feature preprocessing with scikit-learn
* One-hot encoding for categorical variables
* Ablation analysis
* Stratified cross-validation
* Hyperparameter optimization with **Optuna**
* Final model training
* Out-of-sample evaluation
* Model serialization

The complete preprocessing pipeline and model are stored together as a single artifact, allowing the same transformations to be reused during inference.

---

## Evaluation

The project uses metrics appropriate for an imbalanced classification problem:

### Probability metrics

* ROC-AUC
* PR-AUC
* Log Loss

### Classification metrics

* Precision
* Recall
* F1
* Specificity
* Confusion Matrix

The project deliberately avoids relying on accuracy alone because the target variable is imbalanced.

---

# Business Decision Layer

A key design decision is keeping **ML risk** and **business value** independent.

The `ChurnType` is calculated from customer-value variables such as:

* Balance
* Estimated Salary
* Number of Products
* Tenure

It does **not** use the target variable `Exited`.

This produces three customer-value categories:

```text
softchurn
midchurn
hardchurn
```

Each category represents a different potential economic loss.

The campaign layer then combines:

```text
Churn Probability
        +
Economic Loss
        +
Campaign Cost
        +
Retention Success Probability
        ↓
Expected Campaign Profitability
```

This allows the system to answer a more useful business question than churn prediction alone:

> **Which customers should the bank actually contact?**

---

# Threshold Optimization

The classification threshold is not treated as a fixed 0.5 value.

The project evaluates different thresholds using both:

### ML objectives

* Precision
* Recall
* F1
* Specificity

### Business objectives

* Expected Net Profit
* ROI
* Campaign Cost
* Expected Loss
* Economic Impact

This creates a distinction between:

```text
Best classification threshold
            ≠
Best business threshold
```

The threshold analysis is therefore based on the **business objective**, rather than purely on model classification performance.

---

# Explainability

The project includes model explainability using:

### Feature Importance

Global XGBoost feature importance is used to identify the variables contributing most strongly to the model.

### SHAP

SHAP is used for both:

* Global model interpretation
* Individual customer explanations

For an individual customer, the dashboard can show how each feature contributes to the predicted churn probability.

```text
Customer
   │
   ├── Age              ──┐
   ├── NumOfProducts     │
   ├── Balance           ├──→ SHAP → Churn Prediction
   ├── Geography         │
   └── Activity         ─┘
```

---

# Backend Architecture

The system separates the prediction logic from the API layer.

```text
Streamlit
    │
    │ HTTP / direct service usage
    ▼
FastAPI
    │
    ▼
Prediction Service
    │
    ├── XGBoost
    ├── ChurnType
    ├── Economic Loss
    └── Campaign Decision
    │
    ▼
PostgreSQL
```

The central prediction service is independent from FastAPI so that the same business logic can be reused by the dashboard or other interfaces without duplicating code.

---

# API

The REST API is implemented with **FastAPI** and **Pydantic** validation.

| Method | Endpoint                              | Purpose                            |
| ------ | ------------------------------------- | ---------------------------------- |
| `GET`  | `/health`                             | Service, database and model status |
| `GET`  | `/customers/{customer_id}`            | Retrieve customer information      |
| `GET`  | `/customers/{customer_id}/prediction` | Generate customer prediction       |
| `POST` | `/predict`                            | Predict churn for a customer       |

Interactive API documentation is automatically available through:

```text
http://localhost:8000/docs
```

---

# Dashboard

The Streamlit dashboard provides an interface for exploring the complete system.

### Customer Analysis

* Customer information
* Churn probability
* Churn classification
* ChurnType
* Economic loss
* Campaign decision

### Model Explainability

* Global feature importance
* SHAP summary
* Individual SHAP waterfall

### Model Evaluation

* ROC curve
* Precision-Recall curve
* Interactive threshold analysis
* Confusion matrix
* Precision / Recall / F1 / Specificity

### Business Analysis

* Expected Net Profit
* ROI
* Campaign performance
* XGBoost vs baseline strategies

---

# Project Structure

```text
customer-churn-ml/
│
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # Processed datasets
│
├── notebooks/                  # EDA and experimentation
│
├── sql/
│   └── init_schema.sql         # Database schema
│
├── src/
│   ├── data/
│   │   ├── ingestion.py        # ETL pipeline
│   │   └── database.py         # PostgreSQL connection
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py            # Training + Optuna
│   │   ├── predict.py          # Inference
│   │   ├── evaluate.py         # Model metrics
│   │   ├── threshold_analysis.py
│   │   └── explainability.py   # SHAP + feature importance
│   │
│   ├── business/
│   │   ├── churn_impact.py     # Customer value rules
│   │   ├── campaign.py         # Economic simulation
│   │   └── strategy_comparison.py
│   │
│   └── visualization/
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/
│
├── dashboard/
│   └── app.py
│
├── models/
│   └── xgboost_model.pkl
│
├── reports/
│   └── figures/
│
├── tests/
│
├── src/config.py
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

# Tech Stack

| Category         | Technologies           |
| ---------------- | ---------------------- |
| Programming      | Python                 |
| Data             | Pandas, PostgreSQL     |
| Machine Learning | XGBoost, scikit-learn  |
| Optimization     | Optuna                 |
| Explainability   | SHAP                   |
| API              | FastAPI, Pydantic      |
| Dashboard        | Streamlit              |
| Infrastructure   | Docker, Docker Compose |
| Testing          | pytest                 |
| Visualization    | Matplotlib             |

---

# Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/educm12/Customer-Churn-Prediction-Retention-Optimization
cd customer-churn-ml
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure environment

```bash
cp .env.example .env
```

Configure the PostgreSQL credentials if required.

## 4. Start PostgreSQL

```bash
docker compose up -d postgres
```

## 5. Load the dataset

```bash
python -m src.data.ingestion
```

## 6. Train the model

```bash
python -m src.models.train
```

## 7. Analysis & Explainability

Additional analysis can be generated with:

```bash
# Threshold optimization
python -m src.models.threshold_analysis

# Strategy comparison
python -m src.business.strategy_comparison

# SHAP + feature importance
python -m src.models.explainability
```

Generated reports and visualizations are stored in:

```text
reports/
```


## 8. Run the Application

The application can be run either locally or entirely with Docker Compose.

### Option 1 — Run locally

Start PostgreSQL with Docker:

```bash
docker compose up -d postgres
```

Then run the API:

```bash
uvicorn api.main:app --reload --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

In a separate terminal, start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Dashboard:

```text
http://localhost:8501
```

### Option 2 — Run the full application with Docker Compose

Docker Compose can start the complete application stack:

* **PostgreSQL** — database
* **FastAPI** — ML prediction API
* **Streamlit** — interactive dashboard

Start all services with:

```bash
docker compose up --build
```

Once the containers are running:

**API**

```text
http://localhost:8000
```

**API documentation**

```text
http://localhost:8000/docs
```

**Streamlit dashboard**

```text
http://localhost:8501
```

To run the containers in the background:

```bash
docker compose up --build -d
```

To stop the application:

```bash
docker compose down
```

PostgreSQL data is persisted through the `pgdata` Docker volume, so restarting the containers does not remove the database contents.
---


# Testing

Run all tests:

```bash
pytest tests/ -v
```

Run only unit tests:

```bash
pytest tests/ -v -m "not integration"
```

Integration tests require PostgreSQL and the trained model.

---

# Engineering Principles

This project follows several principles commonly used in production ML systems:

* **Separation of concerns** between data, ML, business logic and interfaces.
* **Reusable preprocessing pipelines** to ensure consistent training and inference.
* **Out-of-sample evaluation** for model performance and economic analysis.
* **Centralized configuration** instead of hardcoded business parameters.
* **Explainability** through SHAP and model feature importance.
* **Reusable prediction service** independent of the API framework.
* **Automated testing** for data, business logic, ML components and API endpoints.
* **Containerized infrastructure** using Docker.
* **Separation of prediction and decision-making**, allowing model risk estimates to be combined with business constraints.

---

# Project Goal

The main objective is not simply to predict which customers will churn.

It is to build an end-to-end ML system capable of connecting:

```text
Data
  ↓
Machine Learning
  ↓
Risk Prediction
  ↓
Customer Value
  ↓
Economic Optimization
  ↓
Business Decision
  ↓
API / Dashboard
```

This makes the project a combination of **Machine Learning, Data Engineering, MLOps, Explainable AI and Business Analytics** rather than a standalone classification model.
