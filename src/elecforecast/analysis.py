"""Stationarity testing and seasonal decomposition.

This is the diagnostic layer - the work you do *before* choosing a model, which
tells you whether the model's assumptions are satisfied at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from elecforecast.data import SEASONAL_PERIOD


@dataclass
class StationarityReport:
    """Outcome of an Augmented Dickey-Fuller test."""

    statistic: float
    p_value: float
    critical_values: dict[str, float]
    n_lags: int
    is_stationary: bool
    alpha: float

    @property
    def verdict(self) -> str:
        if self.is_stationary:
            return (
                f"Stationary (p = {self.p_value:.4g} < {self.alpha}). "
                "The null of a unit root is rejected."
            )
        return (
            f"Non-stationary (p = {self.p_value:.4g} >= {self.alpha}). "
            "Cannot reject the null of a unit root - differencing is needed."
        )


def adf_test(series: pd.Series, alpha: float = 0.05) -> StationarityReport:
    """Run ADF and interpret the result.

    The null hypothesis is that a unit root is present, i.e. the series is
    *non*-stationary. A small p-value rejects that null - so low p is the
    outcome you want, which is the opposite of the intuition most people carry
    over from other tests.
    """
    clean = series.dropna()
    if len(clean) < 2 * SEASONAL_PERIOD:
        raise ValueError("series too short for a meaningful ADF test")

    statistic, p_value, n_lags, _, critical, _ = adfuller(clean, autolag="AIC")

    return StationarityReport(
        statistic=float(statistic),
        p_value=float(p_value),
        critical_values={k: float(v) for k, v in critical.items()},
        n_lags=int(n_lags),
        is_stationary=bool(p_value < alpha),
        alpha=alpha,
    )


def decompose(series: pd.Series, model: str = "additive") -> DecomposeResult:
    """Split the series into trend, seasonal, and residual components."""
    if len(series) < 2 * SEASONAL_PERIOD:
        raise ValueError("need at least two full seasonal cycles to decompose")
    return seasonal_decompose(series, model=model, period=SEASONAL_PERIOD)


def make_stationary(
    series: pd.Series, max_diff: int = 2, alpha: float = 0.05
) -> tuple[pd.Series, int, StationarityReport]:
    """Difference until ADF reports stationarity, and report how many passes it took.

    The number of differences taken is the ``d`` term an ARIMA model needs, so
    this doubles as parameter selection rather than just a transformation.
    """
    current = series.copy()
    for order in range(max_diff + 1):
        report = adf_test(current, alpha=alpha)
        if report.is_stationary:
            return current, order, report
        current = current.diff().dropna()

    return current, max_diff, adf_test(current, alpha=alpha)


def rolling_stats(series: pd.Series, window: int = SEASONAL_PERIOD) -> pd.DataFrame:
    """Rolling mean and standard deviation - the visual companion to the ADF test.

    A flat rolling mean and a stable rolling standard deviation are what
    stationarity looks like on a chart.
    """
    return pd.DataFrame(
        {
            "series": series,
            "rolling_mean": series.rolling(window).mean(),
            "rolling_std": series.rolling(window).std(),
        }
    )


def seasonal_profile(series: pd.Series) -> pd.DataFrame:
    """Average value per calendar month, to expose the shape of the annual cycle."""
    frame = pd.DataFrame({"value": series, "month": series.index.month})
    profile = (
        frame.groupby("month")["value"]
        .agg(["mean", "std", "min", "max"])
        .round(2)
    )
    profile.index = pd.to_datetime(profile.index, format="%m").strftime("%b")
    profile.index.name = "month"
    return profile


def yoy_growth(series: pd.Series) -> pd.Series:
    """Year-over-year percentage change - removes seasonality by construction."""
    growth = series.pct_change(SEASONAL_PERIOD) * 100
    growth.name = "yoy_growth_pct"
    return growth.replace([np.inf, -np.inf], np.nan)
