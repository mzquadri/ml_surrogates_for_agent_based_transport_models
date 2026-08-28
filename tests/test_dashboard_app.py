from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "thesis_dashboard" / "app.py"


def test_dashboard_renders_primary_evidence() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    assert not app.exception
    assert app.selectbox[0].value == "Trial 8 MC Dropout"
    assert app.slider[0].value == 50
    assert len(app.metric) >= 5
    assert any("Full-data R2" in metric.label for metric in app.metric)


def test_trial_7_selection_does_not_claim_trial_8_calibration() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    app.selectbox[0].set_value("Trial 7 MC Dropout")
    app.run()

    assert not app.exception
    assert any(
        "Calibration below belongs to Trial 8" in warning.value
        for warning in app.warning
    )
