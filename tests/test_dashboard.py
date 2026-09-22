"""
Tests del dashboard de Streamlit, usando
`streamlit.testing.v1.AppTest` para ejecutar el script de forma
headless y comprobar que no lanza excepciones.

Marcados `integration`: el dashboard necesita PostgreSQL y el modelo
entrenado (igual que la API).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration

APP_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    return at


class TestDashboardLoadsWithoutErrors:
    def test_initial_run_has_no_exceptions(self, app):
        assert not app.exception

    def test_title_is_rendered(self, app):
        assert len(app.title) > 0
        assert "Churn" in app.title[0].value

    def test_renders_four_tabs(self, app):
        assert len(app.tabs) == 4

    def test_has_a_customer_selectbox(self, app):
        assert len(app.selectbox) > 0

    def test_has_a_threshold_slider(self, app):
        assert len(app.slider) > 0


class TestDashboardInteractivity:
    def test_changing_selected_customer_does_not_raise(self, app):
        selectbox = app.selectbox[0]
        original_value = selectbox.value
        other_option = next(opt for opt in selectbox.options if opt != original_value)

        result = selectbox.select(other_option).run(timeout=120)

        assert not result.exception
        assert selectbox.value == other_option

    def test_moving_threshold_slider_does_not_raise(self, app):
        slider = app.slider[0]

        result = slider.set_value(0.5).run(timeout=120)

        assert not result.exception
        assert slider.value == 0.5

    def test_extreme_low_threshold_does_not_raise(self, app):
        """Threshold muy bajo -> casi todos predichos como churn;
        comprueba que la matriz de confusion y las metricas no rompen
        en el caso extremo."""
        slider = app.slider[0]
        result = slider.set_value(0.05).run(timeout=120)
        assert not result.exception

    def test_extreme_high_threshold_does_not_raise(self, app):
        """Threshold muy alto -> casi nadie predicho como churn
        (posible division por cero en alguna metrica si no se maneja
        bien)."""
        slider = app.slider[0]
        result = slider.set_value(0.95).run(timeout=120)
        assert not result.exception