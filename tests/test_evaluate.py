"""Metric and backtest tests.

The leakage test is the important one here. A backtest that lets a fold see its
own test period reports excellent scores and is completely worthless, and the
failure is invisible unless you assert against it directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elecforecast.data import load_series
from elecforecast.evaluate import (
    compare,
    mae,
    mape,
    mase,
    rmse,
    score,
    walk_forward,
)
from elecforecast.models import MODELS, SeasonalNaive


@pytest.fixture(scope="module")
def series() -> pd.Series:
    return load_series()


def make(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="MS"), dtype="float64"
    )


class TestMetrics:
    def test_perfect_forecast_scores_zero(self) -> None:
        actual = make([10.0, 20.0, 30.0])
        assert rmse(actual, actual) == 0.0
        assert mae(actual, actual) == 0.0
        assert mape(actual, actual) == 0.0

    def test_rmse_penalises_large_errors_more_than_mae(self) -> None:
        actual = make([10.0, 10.0, 10.0, 10.0])
        # One big miss rather than four small ones.
        spiky = make([10.0, 10.0, 10.0, 18.0])
        assert rmse(actual, spiky) > mae(actual, spiky)

    def test_mae_is_mean_absolute_difference(self) -> None:
        actual = make([10.0, 20.0])
        predicted = make([12.0, 18.0])
        assert mae(actual, predicted) == pytest.approx(2.0)

    def test_mape_is_scale_free(self) -> None:
        actual = make([100.0, 200.0])
        predicted = make([110.0, 220.0])
        # Both predictions are 10% high.
        assert mape(actual, predicted) == pytest.approx(10.0)

    def test_mape_rejects_zeros(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            mape(make([0.0, 1.0]), make([1.0, 1.0]))

    def test_mase_below_one_beats_baseline(self, series: pd.Series) -> None:
        train = series.iloc[:-12]
        test = series.iloc[-12:]

        baseline = SeasonalNaive().fit(train).predict(12)
        baseline.index = test.index

        # Scoring the baseline against itself anchors MASE near 1.0 by construction.
        assert mase(test, baseline, train) == pytest.approx(1.0, abs=0.6)

    def test_score_returns_all_metrics(self, series: pd.Series) -> None:
        train, test = series.iloc[:-12], series.iloc[-12:]
        predicted = SeasonalNaive().fit(train).predict(12)
        predicted.index = test.index

        result = score(test, predicted, train)
        assert set(result) == {"RMSE", "MAE", "MAPE %", "MASE"}
        assert all(np.isfinite(v) for v in result.values())

    def test_score_rejects_disjoint_index(self, series: pd.Series) -> None:
        train = series.iloc[:-12]
        actual = make([1.0, 2.0], start="1800-01-01")
        predicted = make([1.0, 2.0], start="1900-01-01")
        with pytest.raises(ValueError, match="do not overlap"):
            score(actual, predicted, train)


class TestWalkForward:
    def test_produces_one_row_per_fold(self, series: pd.Series) -> None:
        result = walk_forward(SeasonalNaive, series, horizon=12, n_folds=3)
        assert len(result.folds) == 3
        assert list(result.folds["fold"]) == [1, 2, 3]

    def test_folds_advance_chronologically(self, series: pd.Series) -> None:
        result = walk_forward(SeasonalNaive, series, horizon=12, n_folds=3)
        starts = pd.to_datetime(result.folds["test_start"], format="%b %Y")
        assert starts.is_monotonic_increasing

    def test_no_leakage_train_ends_before_test_starts(self, series: pd.Series) -> None:
        result = walk_forward(SeasonalNaive, series, horizon=12, n_folds=3)
        train_ends = pd.to_datetime(result.folds["train_end"], format="%b %Y")
        test_starts = pd.to_datetime(result.folds["test_start"], format="%b %Y")
        assert (train_ends < test_starts).all()

    def test_summary_averages_the_folds(self, series: pd.Series) -> None:
        result = walk_forward(SeasonalNaive, series, horizon=12, n_folds=3)
        assert result.summary["RMSE"] == pytest.approx(
            result.folds["RMSE"].mean(), abs=1e-3
        )

    def test_rejects_series_too_short_for_folds(self) -> None:
        short = make([float(i) for i in range(30)])
        with pytest.raises(ValueError, match="need at least"):
            walk_forward(SeasonalNaive, short, horizon=12, n_folds=5)

    def test_rejects_invalid_horizon(self, series: pd.Series) -> None:
        with pytest.raises(ValueError, match="horizon must be positive"):
            walk_forward(SeasonalNaive, series, horizon=0)


class TestCompare:
    def test_ranks_every_model_by_mase(self, series: pd.Series) -> None:
        table = compare(MODELS, series, horizon=12, n_folds=2)
        assert len(table) == len(MODELS)
        assert table["MASE"].is_monotonic_increasing
        assert "model" in table.columns
