"""Electricity production forecasting.

Time-series forecasting of monthly US electric and gas utility production,
benchmarked against a seasonal-naive baseline with walk-forward backtesting.
"""

from elecforecast.analysis import adf_test, decompose, make_stationary, seasonal_profile
from elecforecast.data import SEASONAL_PERIOD, describe, load_series, train_test_split
from elecforecast.evaluate import compare, mae, mape, mase, rmse, score, walk_forward
from elecforecast.models import MODELS, HoltWinters, Sarima, SeasonalNaive, build

__version__ = "1.0.0"

__all__ = [
    "SEASONAL_PERIOD",
    "MODELS",
    "HoltWinters",
    "Sarima",
    "SeasonalNaive",
    "adf_test",
    "build",
    "compare",
    "decompose",
    "describe",
    "load_series",
    "mae",
    "make_stationary",
    "mape",
    "mase",
    "rmse",
    "score",
    "seasonal_profile",
    "train_test_split",
    "walk_forward",
]
