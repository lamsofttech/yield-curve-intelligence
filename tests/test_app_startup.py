"""Application startup test -- imports app.py (which loads data and builds
the full Gradio Blocks UI at module scope) and checks it constructs
without raising. Does not launch a real server; that was verified manually
(see LEARNING_LOG.md) and is Docker/CI's job (Phase 9), not a unit test's.
"""

import gradio as gr


def test_app_module_imports_and_builds_blocks_without_error():
    import app

    assert isinstance(app.demo, gr.Blocks)


def test_date_choices_are_populated():
    import app

    assert len(app._date_choices) > 0
    assert "2023-10-19" in app._date_choices  # notable date, always included


def test_pca_and_anomaly_results_are_available_at_import_time():
    import app

    assert app._pca_result is not None
    assert app._anomaly_result is not None
    assert 0 <= app._current_pct <= 100
