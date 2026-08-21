"""Metrics and walk-forward backtesting.

Two deliberate choices here:

* **No "accuracy".** This is a regression problem. Accuracy is a classification
  metric and reporting it on a continuous series is meaningless - an earlier
  version of this project's README claimed "90% accuracy", which is exactly the
  kind of number that falls apart under a follow-up question.
* **Walk-forward, not a single split.** One split gives one number with no sense
  of whether it was luck. Re-fitting across several successive origins shows
  whether a model is *consistently* better or just better once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from elecforecast.data import SEASONAL_PERIOD


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root mean squared error - in the units of the series, penalises big misses."""
    return float(np.sqrt(np.mean((actual.to_numpy() - predicted.to_numpy()) ** 2)))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute error - in the units of the series, robust to outliers."""
    return float(np.mean(np.abs(actual.to_numpy() - predicted.to_numpy())))


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute percentage error, as a percentage.

    Scale-free, so it's comparable across series - but undefined at zero, which
    is safe here because the loader rejects non-positive values.
    """
    a = actual.to_numpy()
    p = predicted.to_numpy()
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when actuals contain zero")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def mase(actual: pd.Series, predicted: pd.Series, train: pd.Series) -> float:
    """Mean absolute scaled error, scaled by the in-sample seasonal-naive error.

    Reads as a direct verdict: **below 1.0 beats the seasonal baseline, above 1.0
    loses to it.** That single anchor is more informative than any raw error
    number, because it answers "compared to what?".
    """
    train_values = train.to_numpy()
    if len(train_values) <= SEASONAL_PERIOD:
        raise ValueError("training series too short to scale against")

    naive_error = np.mean(
        np.abs(train_values[SEASONAL_PERIOD:] - train_values[:-SEASONAL_PERIOD])
    )
    if naive_error == 0:
        raise ValueError("seasonal-naive error is zero; cannot scale")

    return float(mae(actual, predicted) / naive_error)


def score(actual: pd.Series, predicted: pd.Series, train: pd.Series) -> dict[str, float]:
    """All metrics for one forecast, aligned on the overlapping index."""
    aligned = pd.concat([actual.rename("actual"), predicted.rename("pred")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("actual and predicted series do not overlap")

    a, p = aligned["actual"], aligned["pred"]
    return {
        "RMSE": round(rmse(a, p), 3),
        "MAE": round(mae(a, p), 3),
        "MAPE %": round(mape(a, p), 2),
        "MASE": round(mase(a, p, train), 3),
    }


@dataclass
class BacktestResult:
    """Per-fold scores plus their average."""

    model: str
    folds: pd.DataFrame
    summary: dict[str, float]

    @property
    def beats_baseline(self) -> bool:
        return self.summary.get("MASE", float("inf")) < 1.0


def walk_forward(
    model_factory,
    series: pd.Series,
    horizon: int = 12,
    n_folds: int = 5,
    label: str = "model",
) -> BacktestResult:
    """Backtest by re-fitting at successive origins and forecasting forward.

    Fold *i* trains on everything up to its origin and is scored on the ``horizon``
    months that follow. The training window only ever grows backwards from the
    origin, so no fold can see data from its own test period.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")

    min_train = 4 * SEASONAL_PERIOD
    required = min_train + horizon * n_folds
    if len(series) < required:
        raise ValueError(
            f"need at least {required} observations for {n_folds} folds of "
            f"{horizon} months; series has {len(series)}"
        )

    rows = []
    for fold in range(n_folds):
        # Oldest fold first, so the reported order reads chronologically.
        end = len(series) - horizon * (n_folds - fold)
        train = series.iloc[:end]
        test = series.iloc[end : end + horizon]

        fitted = model_factory().fit(train)
        predicted = fitted.predict(horizon)
        predicted.index = test.index

        rows.append(
            {
                "fold": fold + 1,
                "train_end": train.index[-1].strftime("%b %Y"),
                "test_start": test.index[0].strftime("%b %Y"),
                **score(test, predicted, train),
            }
        )

    folds = pd.DataFrame(rows)
    metrics = ["RMSE", "MAE", "MAPE %", "MASE"]
    summary = {m: round(float(folds[m].mean()), 3) for m in metrics}

    return BacktestResult(model=label, folds=folds, summary=summary)


def compare(
    models: dict[str, type],
    series: pd.Series,
    horizon: int = 12,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Backtest every model and rank them by MASE (lower is better)."""
    rows = []
    for key, cls in models.items():
        result = walk_forward(cls, series, horizon=horizon, n_folds=n_folds, label=key)
        rows.append({"model": cls().name, **result.summary})

    return (
        pd.DataFrame(rows)
        .sort_values("MASE")
        .reset_index(drop=True)
    )
