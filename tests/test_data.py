"""Loader and validation tests.

The validation tests matter more than they look: the month-first/day-first bug
these guard against parsed cleanly, produced a plausible-looking Series, and only
surfaced as nonsense twenty steps downstream.
"""

from __future__ import annotations

import pandas as pd
import pytest

from elecforecast.data import (
    DataValidationError,
    describe,
    load_series,
    train_test_split,
    validate_series,
)


@pytest.fixture(scope="module")
def series() -> pd.Series:
    return load_series()


def make_series(values: list[float], start: str = "1991-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=index, name="production", dtype="float64")


class TestLoad:
    def test_loads_expected_shape(self, series: pd.Series) -> None:
        assert len(series) == 397
        assert series.name == "production"
        assert series.dtype == "float64"

    def test_index_is_monthly_and_sorted(self, series: pd.Series) -> None:
        assert isinstance(series.index, pd.DatetimeIndex)
        assert series.index.freqstr == "MS"
        assert series.index.is_monotonic_increasing

    def test_dates_parsed_month_first(self, series: pd.Series) -> None:
        # The second row of the CSV is 02-01-1991, which is February 1991.
        # Parsed day-first it would be 2 January 1991 and the whole series
        # would collapse into a single month.
        assert series.index[0] == pd.Timestamp("1991-01-01")
        assert series.index[1] == pd.Timestamp("1991-02-01")
        assert series.index[-1] == pd.Timestamp("2024-01-01")

    def test_spans_expected_range(self, series: pd.Series) -> None:
        assert series.index.min().year == 1991
        assert series.index.max().year == 2024

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_series("does/not/exist.csv")


class TestValidation:
    def test_rejects_empty(self) -> None:
        with pytest.raises(DataValidationError, match="empty"):
            validate_series(pd.Series([], dtype="float64"))

    def test_rejects_missing_values(self) -> None:
        s = make_series([1.0, 2.0, 3.0])
        s.iloc[1] = None
        with pytest.raises(DataValidationError, match="missing value"):
            validate_series(s)

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(DataValidationError, match="non-positive"):
            validate_series(make_series([1.0, 0.0, 3.0]))

    def test_rejects_duplicate_timestamps(self) -> None:
        s = make_series([1.0, 2.0, 3.0])
        s.index = pd.DatetimeIndex(
            ["1991-01-01", "1991-01-01", "1991-03-01"], name="date"
        )
        with pytest.raises(DataValidationError, match="duplicate"):
            validate_series(s)

    def test_rejects_daily_spacing(self) -> None:
        # Exactly the shape a day-first parsing bug produces.
        s = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range("1991-01-01", periods=3, freq="D", name="date"),
            name="production",
        )
        with pytest.raises(DataValidationError, match="not evenly spaced"):
            validate_series(s)

    def test_accepts_valid_monthly(self) -> None:
        s = make_series([1.0, 2.0, 3.0])
        assert validate_series(s) is s


class TestSplit:
    def test_split_is_chronological(self, series: pd.Series) -> None:
        train, test = train_test_split(series, test_size=24)
        assert len(test) == 24
        assert len(train) == len(series) - 24
        # No leakage: every training timestamp precedes every test timestamp.
        assert train.index.max() < test.index.min()

    def test_split_preserves_all_observations(self, series: pd.Series) -> None:
        train, test = train_test_split(series, test_size=24)
        assert len(train) + len(test) == len(series)

    def test_rejects_oversized_test(self, series: pd.Series) -> None:
        with pytest.raises(ValueError, match="smaller than the series"):
            train_test_split(series, test_size=len(series))

    def test_rejects_non_positive_test_size(self, series: pd.Series) -> None:
        with pytest.raises(ValueError, match="positive"):
            train_test_split(series, test_size=0)


def test_describe_reports_span(series: pd.Series) -> None:
    facts = describe(series)
    assert facts["observations"] == 397
    assert facts["start"] == "Jan 1991"
    assert facts["end"] == "Jan 2024"
    assert facts["min"] < facts["mean"] < facts["max"]
