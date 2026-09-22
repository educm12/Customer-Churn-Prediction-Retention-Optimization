"""
Tests de src/models/threshold_analysis.py.

Todos unitarios, con datos sinteticos en memoria (no requieren
PostgreSQL ni el modelo entrenado).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.threshold_analysis import (
    build_threshold_table,
    select_threshold_business_optimal,
    select_threshold_max_f1,
    select_threshold_min_precision,
    select_threshold_min_recall,
    summarize_threshold_selection,
)


def make_synthetic_predictions(n=500, seed=0):
    """Genera y_true/churn_probability/economic_loss sinteticos pero
    realistas: la probabilidad predicha esta correlacionada con el
    desenlace real (como un modelo razonable), no es ruido puro."""
    rng = np.random.default_rng(seed)
    y_true = pd.Series(rng.binomial(1, 0.20, n))
    # Probabilidad correlacionada con y_true + ruido, recortada a [0, 1]
    noise = rng.normal(0, 0.2, n)
    churn_probability = pd.Series(np.clip(y_true * 0.6 + 0.15 + noise, 0.01, 0.99))
    economic_loss = pd.Series(rng.choice([500, 1000, 2000], n))
    return y_true, churn_probability, economic_loss


class TestBuildThresholdTable:
    def test_returns_one_row_per_threshold(self):
        y_true, proba, loss = make_synthetic_predictions()
        thresholds = [0.2, 0.4, 0.6, 0.8]
        table = build_threshold_table(y_true, proba, loss, thresholds)
        assert len(table) == len(thresholds)

    def test_contains_ml_and_economic_columns(self):
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss, [0.3, 0.5, 0.7])
        ml_cols = {"precision", "recall", "f1", "specificity", "tp", "fp", "tn", "fn"}
        econ_cols = {"expected_net_profit", "total_economic_impact", "roi_pct", "total_campaign_cost"}
        assert ml_cols.issubset(table.columns)
        assert econ_cols.issubset(table.columns)

    def test_default_thresholds_span_010_to_090(self):
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        assert table["threshold"].min() == pytest.approx(0.10)
        assert table["threshold"].max() == pytest.approx(0.90)

    def test_higher_threshold_reduces_campaign_cost(self):
        """A mayor threshold, se contacta a menos gente -> menos coste."""
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss, [0.2, 0.5, 0.8])
        costs = table.sort_values("threshold")["total_campaign_cost"].tolist()
        assert costs == sorted(costs, reverse=True)


class TestThresholdSelection:
    def test_select_max_f1_matches_manual_argmax(self):
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        selected = select_threshold_max_f1(table)
        expected = table.loc[table["f1"].idxmax(), "threshold"]
        assert selected == pytest.approx(expected)

    def test_select_min_recall_meets_the_recall_floor(self):
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        selected = select_threshold_min_recall(table, min_recall=0.7)
        row = table[table["threshold"] == selected].iloc[0]
        assert row["recall"] >= 0.7

    def test_select_min_recall_is_the_highest_threshold_meeting_the_floor(self):
        """De entre los thresholds que cumplen recall>=floor, debe
        elegir el MAS ALTO (maximiza precision sin romper el floor)."""
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        selected = select_threshold_min_recall(table, min_recall=0.7)
        higher_thresholds = table[table["threshold"] > selected]
        assert (higher_thresholds["recall"] < 0.7).all()

    def test_select_min_precision_meets_the_precision_floor(self):
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        selected = select_threshold_min_precision(table, min_precision=0.4)
        row = table[table["threshold"] == selected].iloc[0]
        assert row["precision"] >= 0.4

    def test_select_min_precision_is_the_lowest_threshold_meeting_the_floor(self):
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        selected = select_threshold_min_precision(table, min_precision=0.4)
        lower_thresholds = table[table["threshold"] < selected]
        assert (lower_thresholds["precision"] < 0.4).all()

    def test_select_min_recall_falls_back_when_floor_unreachable(self):
        """Si ningun threshold alcanza el floor pedido, no debe lanzar
        excepcion: debe devolver el de mayor recall disponible."""
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        selected = select_threshold_min_recall(table, min_recall=1.5)  # imposible
        expected = table.loc[table["recall"].idxmax(), "threshold"]
        assert selected == pytest.approx(expected)

    def test_select_business_optimal_matches_manual_argmax(self):
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        selected = select_threshold_business_optimal(table)
        expected = table.loc[table["expected_net_profit"].idxmax(), "threshold"]
        assert selected == pytest.approx(expected)

    def test_business_optimal_never_gives_worse_profit_than_extremes(self):
        """El threshold optimo de negocio debe dar, como minimo, el
        mismo beneficio que enviar a todos (threshold minimo) o a nadie
        (equivalente a threshold=1.0, fuera de la rejilla pero
        comprobamos contra el threshold mas alto disponible)."""
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        best_profit = table["expected_net_profit"].max()
        assert best_profit >= table["expected_net_profit"].iloc[0]
        assert best_profit >= table["expected_net_profit"].iloc[-1]


class TestSummarizeThresholdSelection:
    def test_returns_one_row_per_criterion(self):
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        summary = summarize_threshold_selection(table)
        assert len(summary) == 4

    def test_summary_thresholds_are_all_valid_grid_points(self):
        y_true, proba, loss = make_synthetic_predictions()
        table = build_threshold_table(y_true, proba, loss)
        summary = summarize_threshold_selection(table)
        valid_thresholds = set(table["threshold"].round(2))
        assert set(summary["threshold"].round(2)).issubset(valid_thresholds)
