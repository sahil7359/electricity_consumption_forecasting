"""Loading and validating the electric production series.

The dataset is a monthly index of US industrial production for electric and gas
utilities (FRED series IPG2211A2N). It ships with the repo so the demo works with
no network access and no credentials.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "Electric_Production.csv"

#: The series is monthly. Twelve observations make a full seasonal cycle, which is
#: the period used by seasonal decomposition, SARIMA, and the seasonal-naive baseline.
SEASONAL_PERIOD = 12


class DataValidationError(ValueError):
    """Raised when the loaded series violates an assumption the models depend on."""


def load_series(path: Path | str | None = None) -> pd.Series:
    """Load the production series, indexed by month-start timestamps.

    Returns a float Series named ``production`` with a monthly ``DatetimeIndex``.
    The index carries an explicit ``MS`` frequency — statsmodels silently degrades
    to non-seasonal behaviour when the frequency is missing, which is the kind of
    bug that shows up as "the forecast is a flat line" three steps later.
    """
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"dataset not found at {path}")

    frame = pd.read_csv(path)
    expected = {"DATE", "Value"}
    missing = expected - set(frame.columns)
    if missing:
        raise DataValidationError(f"missing expected column(s): {sorted(missing)}")

    # Source file is month-first: 02-01-1991 is 1 February 1991, not 2 January.
    # Parsing this day-first silently yields a series spaced one *day* apart, which
    # every seasonal model downstream would then quietly misinterpret.
    index = pd.to_datetime(frame["DATE"], format="%m-%d-%Y")
    series = pd.Series(
        pd.to_numeric(frame["Value"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(index, name="date"),
        name="production",
        dtype="float64",
    ).sort_index()

    return validate_series(series).asfreq("MS")


def validate_series(series: pd.Series) -> pd.Series:
    """Check the invariants every model downstream assumes.

    Failing loudly here is deliberate. A silently-missing month shifts every
    seasonal lag by one and quietly corrupts a seasonal model's notion of
    "same month last year".
    """
    if series.empty:
        raise DataValidationError("series is empty")

    if series.isna().any():
        n = int(series.isna().sum())
        raise DataValidationError(f"series contains {n} missing value(s)")

    if not series.index.is_monotonic_increasing:
        raise DataValidationError("index is not sorted ascending")

    if series.index.has_duplicates:
        dupes = series.index[series.index.duplicated()].tolist()
        raise DataValidationError(f"duplicate timestamps: {dupes[:5]}")

    if (series <= 0).any():
        raise DataValidationError(
            "series contains non-positive values; log transform would be undefined"
        )

    gaps = series.index.to_series().diff().dropna()
    if not gaps.empty:
        # Month lengths vary (28-31 days), so allow a window rather than an exact match.
        irregular = gaps[(gaps < pd.Timedelta(days=28)) | (gaps > pd.Timedelta(days=31))]
        if not irregular.empty:
            raise DataValidationError(
                f"series is not evenly spaced monthly; {len(irregular)} irregular gap(s)"
            )

    return series


def train_test_split(
    series: pd.Series, test_size: int = 24
) -> tuple[pd.Series, pd.Series]:
    """Split chronologically, holding out the most recent ``test_size`` months.

    This is the whole point: a random split would let the model train on 2023 and
    be tested on 2019, which leaks the future into the past and produces scores
    that cannot survive contact with a real forecast.
    """
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if test_size >= len(series):
        raise ValueError(
            f"test_size ({test_size}) must be smaller than the series ({len(series)})"
        )

    return series.iloc[:-test_size], series.iloc[-test_size:]


def describe(series: pd.Series) -> dict[str, object]:
    """Summary facts about the series, for display in the app."""
    return {
        "observations": int(series.size),
        "start": series.index.min().strftime("%b %Y"),
        "end": series.index.max().strftime("%b %Y"),
        "mean": round(float(series.mean()), 2),
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
        "years": round(series.size / SEASONAL_PERIOD, 1),
    }
