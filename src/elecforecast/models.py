"""Forecasting models, behind one interface.

Every model here answers the same two questions - ``fit(train)`` and
``predict(horizon)`` - so the backtester can treat them interchangeably and the
app can offer them as a dropdown.

The baseline is not a formality. Seasonal-naive is genuinely hard to beat on a
strongly seasonal series, and a "sophisticated" model that loses to it is telling
you something important about itself.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from elecforecast.data import SEASONAL_PERIOD


class Forecaster:
    """Interface every model implements."""

    name: str = "forecaster"
    description: str = ""

    def fit(self, train: pd.Series) -> Forecaster:
        raise NotImplementedError

    def predict(self, horizon: int) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
        return pd.date_range(
            train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )


@dataclass
class SeasonalNaive(Forecaster):
    """This month next year = this month last year.

    The bar every other model has to clear. It has no parameters, cannot overfit,
    and captures the single strongest signal in the series.
    """

    name: str = "Seasonal naive (baseline)"
    description: str = "Repeats the value from the same month one year earlier."
    period: int = SEASONAL_PERIOD
    _train: pd.Series | None = field(default=None, repr=False)

    def fit(self, train: pd.Series) -> SeasonalNaive:
        if len(train) < self.period:
            raise ValueError(
                f"need at least {self.period} observations, got {len(train)}"
            )
        self._train = train
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._train is None:
            raise RuntimeError("call fit() before predict()")

        last_cycle = self._train.iloc[-self.period :].to_numpy()
        # Tile the final observed year forward, repeating as far as the horizon needs.
        values = np.resize(last_cycle, horizon)
        return pd.Series(
            values,
            index=self._future_index(self._train, horizon),
            name="forecast",
        )


@dataclass
class HoltWinters(Forecaster):
    """Triple exponential smoothing - level, trend, and seasonality."""

    name: str = "Holt-Winters"
    description: str = (
        "Exponential smoothing with additive trend and seasonal components."
    )
    seasonal: str = "add"
    trend: str = "add"
    period: int = SEASONAL_PERIOD
    _fitted: object | None = field(default=None, repr=False)
    _train: pd.Series | None = field(default=None, repr=False)

    def fit(self, train: pd.Series) -> HoltWinters:
        if len(train) < 2 * self.period:
            raise ValueError(
                f"need at least {2 * self.period} observations for a seasonal fit"
            )
        self._train = train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted = ExponentialSmoothing(
                train,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.period,
                initialization_method="estimated",
            ).fit()
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        forecast = self._fitted.forecast(horizon)
        forecast.name = "forecast"
        return forecast


@dataclass
class Sarima(Forecaster):
    """Seasonal ARIMA.

    The notebook this project grew out of used plain ARIMA on a differenced series.
    SARIMA models the seasonality directly through the seasonal order, rather than
    hoping differencing scrubs it out - which is the right tool for a series whose
    dominant signal is an annual cycle.
    """

    name: str = "SARIMA"
    description: str = "Seasonal ARIMA - models the annual cycle explicitly."
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, SEASONAL_PERIOD)
    _fitted: object | None = field(default=None, repr=False)

    def fit(self, train: pd.Series) -> Sarima:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted = SARIMAX(
                train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        return self

    def predict(self, horizon: int) -> pd.Series:
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        forecast = self._fitted.forecast(horizon)
        forecast.name = "forecast"
        return forecast

    def predict_interval(
        self, horizon: int, alpha: float = 0.05
    ) -> pd.DataFrame:
        """Forecast with a confidence interval.

        A point forecast with no uncertainty attached invites false confidence;
        the interval is what makes the forecast honest about what it doesn't know.
        """
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        result = self._fitted.get_forecast(horizon)
        frame = result.conf_int(alpha=alpha)
        frame.columns = ["lower", "upper"]
        frame.insert(0, "forecast", result.predicted_mean)
        return frame


#: Registry the app and CLI iterate over.
MODELS: dict[str, type[Forecaster]] = {
    "seasonal_naive": SeasonalNaive,
    "holt_winters": HoltWinters,
    "sarima": Sarima,
}


def build(key: str) -> Forecaster:
    """Instantiate a model by registry key."""
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose from {sorted(MODELS)}")
    return MODELS[key]()
