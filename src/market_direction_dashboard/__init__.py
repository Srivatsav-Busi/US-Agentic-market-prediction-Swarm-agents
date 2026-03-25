"""US market prediction package."""

__all__ = ["run_daily_prediction", "run_dashboard_pipeline", "run_prediction_scheduler"]


def run_daily_prediction(*args, **kwargs):
    from .pipeline import run_daily_prediction as _run_daily_prediction

    return _run_daily_prediction(*args, **kwargs)


def run_dashboard_pipeline(*args, **kwargs):
    from .pipeline import run_dashboard_pipeline as _run_dashboard_pipeline

    return _run_dashboard_pipeline(*args, **kwargs)


def run_prediction_scheduler(*args, **kwargs):
    from .pipeline import run_prediction_scheduler as _run_prediction_scheduler

    return _run_prediction_scheduler(*args, **kwargs)
