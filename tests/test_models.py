"""Model contract tests.

Every forecaster has to honour the same interface, and the seasonal-naive
baseline has to do exactly what it claims - because every other model's MASE is
scaled against it, a broken baseline silently corrupts every reported score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elecforecast.data import SEASONAL_PERIOD, load_series
from elecforecast.models import MODELS, HoltWinters, Sarima, SeasonalNaive, build


@pytest.fixture(scope="module")
def series() -> pd.Series:
    return load_series()


@pytest.fixture(scope="module")
def train(series: pd.Series) -> pd.Series:
    return series.iloc[:-12]


class TestSeasonalNaive:
    def test_repeats_last_observed_year(self, train: pd.Series) -> None:
        model = SeasonalNaive().fit(train)
        forecast = model.predict(SEASONAL_PERIOD)
        expected = train.iloc[-SEASONAL_PERIOD:].to_numpy()
        np.testing.assert_allclose(forecast.to_numpy(), expected)

    def test_tiles_beyond_one_cycle(self, train: pd.Series) -> None:
        forecast = SeasonalNaive().fit(train).predict(SEASONAL_PERIOD * 2)
        assert len(forecast) == SEASONAL_PERIOD * 2
        # Second year repeats the first.
        np.testing.assert_allclose(
            forecast.iloc[:SEASONAL_PERIOD].to_numpy(),
            forecast.iloc[SEASONAL_PERIOD:].to_numpy(),
        )

    def test_rejects_short_training_series(self) -> None:
        short = pd.Series(
            np.arange(1, 6, dtype=float),
            index=pd.date_range("2020-01-01", periods=5, freq="MS"),
        )
        with pytest.raises(ValueError, match="at least"):
            SeasonalNaive().fit(short)


@pytest.mark.parametrize("cls", [SeasonalNaive, HoltWinters, Sarima])
class TestForecasterContract:
    def test_fit_returns_self(self, cls, train: pd.Series) -> None:
        model = cls()
        assert model.fit(train) is model

    def test_predict_length_matches_horizon(self, cls, train: pd.Series) -> None:
        assert len(cls().fit(train).predict(6)) == 6

    def test_forecast_starts_after_training_ends(self, cls, train: pd.Series) -> None:
        forecast = cls().fit(train).predict(6)
        assert forecast.index[0] > train.index[-1]
        assert forecast.index.freqstr == "MS"

    def test_predict_before_fit_raises(self, cls) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            cls().predict(6)

    def test_forecast_is_finite_and_plausible(self, cls, train: pd.Series) -> None:
        forecast = cls().fit(train).predict(12)
        assert np.isfinite(forecast.to_numpy()).all()
        # A wildly out-of-range forecast means the model has diverged.
        assert forecast.min() > train.min() * 0.5
        assert forecast.max() < train.max() * 1.5


class TestSarimaIntervals:
    def test_interval_brackets_point_forecast(self, train: pd.Series) -> None:
        frame = Sarima().fit(train).predict_interval(6)
        assert list(frame.columns) == ["forecast", "lower", "upper"]
        assert (frame["lower"] <= frame["forecast"]).all()
        assert (frame["forecast"] <= frame["upper"]).all()

    def test_interval_widens_with_horizon(self, train: pd.Series) -> None:
        frame = Sarima().fit(train).predict_interval(24)
        width = frame["upper"] - frame["lower"]
        # Uncertainty should grow the further out you forecast.
        assert width.iloc[-1] > width.iloc[0]


class TestRegistry:
    def test_registry_keys_build(self) -> None:
        for key in MODELS:
            assert build(key).name

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown model"):
            build("prophet")
