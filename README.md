<div align="center">

# ⚡ Electricity Production Forecasting

**Forecasting monthly US electric &amp; gas utility production — with a baseline that
keeps the results honest.**

[![CI](https://github.com/sahil7359/electricity_consumption_forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/sahil7359/electricity_consumption_forecasting/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-8CAAE6)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-53%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

**[▶ Live demo](https://elecforecast.streamlit.app)** · no signup, no keys — the dataset ships with the app

</div>

---

## The finding

Across a 5-fold walk-forward backtest at a 12-month horizon:

| Model | RMSE | MAE | MAPE % | MASE |
|---|---|---|---|---|
| **SARIMA** | **3.483** | **2.677** | **2.54** | **0.959** ✅ |
| Seasonal naive (baseline) | 4.326 | 3.277 | 3.08 | 1.174 |
| Holt-Winters | 4.441 | 3.656 | 3.50 | 1.308 ❌ |

**MASE below 1.0 beats the seasonal-naive baseline; above 1.0 loses to it.**

Two things worth sitting with:

1. **SARIMA wins, but not by much** — 0.959 means it beats "just repeat last year"
   by about 4%. Real, measurable, and far less dramatic than it would sound if I
   reported the 2.5% MAPE on its own.
2. **Holt-Winters loses to the baseline.** A model can post a respectable-looking
   2.9% MAPE and still be worse than doing nothing clever at all. Without the
   baseline in the table, you would never know.

That second point is the whole reason the baseline is in here.

## Try it

The [live demo](https://elecforecast.streamlit.app) opens on the full dataset with four tabs:

| Tab | What you can do |
|---|---|
| **Overview** | The series, its 12-month rolling mean, the monthly seasonal profile, and year-over-year growth |
| **Diagnostics** | Run the ADF test on raw vs. differenced data, adjust the rolling window, inspect the seasonal decomposition |
| **Forecast** | Pick a model and horizon, and get a forecast — with 95% confidence intervals for SARIMA |
| **Backtest** | Change the horizon and fold count, and watch the ranking shift |

## The problem

Utility production is strongly seasonal — demand swings with heating and cooling —
so naive trend-fitting forecasts badly. The real questions are whether the series can
be made **stationary** enough for ARIMA's assumptions to hold, and whether any model
actually earns its complexity against a trivial baseline.

The diagnostics answer the first question directly: the raw series is **non-stationary**
(ADF p = 0.186), and **one difference** is enough to fix it (p = 4.1e-10). That `d = 1`
isn't just a transformation — it's the ARIMA parameter, derived rather than guessed.

## Dataset

`data/Electric_Production.csv` — monthly US industrial production index for electric and
gas utilities (FRED series `IPG2211A2N`). Committed to the repo, so the demo works
offline with no credentials.

| | |
|---|---|
| **Frequency** | Monthly (`MS`) |
| **Coverage** | January 1991 → January 2024 |
| **Observations** | 397 |
| **Range** | 55.32 – 129.40 (index, mean 88.85) |

## Architecture

```
src/elecforecast/
├── data.py        # loading + validation + chronological split
├── analysis.py    # ADF, decomposition, rolling stats, seasonal profile
├── models.py      # SeasonalNaive · HoltWinters · SARIMA, one interface
├── evaluate.py    # RMSE/MAE/MAPE/MASE + walk-forward backtesting
└── cli.py         # headless entry point
app/streamlit_app.py
tests/             # 53 tests
notebooks/         # the original exploratory analysis, kept as a record
```

Three decisions shape the whole thing:

**Validation fails loudly.** The loader rejects gaps, duplicates, missing values, and
non-monthly spacing. This is not defensive padding — during the rebuild it caught a real
bug: the CSV is **month-first** (`02-01-1991` is February), and parsing it day-first
produced a clean-looking Series spaced one *day* apart. It parsed without error and
would have quietly corrupted every seasonal model downstream.

**No random splits, ever.** `train_test_split` holds out the most recent months, and the
backtester re-fits at successive origins. A random split would train on 2023 and test on
2019, leaking the future into the past.

**One model interface.** Every forecaster implements `fit(train)` / `predict(horizon)`,
so the backtester treats them interchangeably and adding a fourth model means adding one
class and one registry entry.

## Running locally

```bash
git clone https://github.com/sahil7359/electricity_consumption_forecasting.git
```

```bash
cd electricity_consumption_forecasting && pip install -e ".[dev]"
```

```bash
streamlit run app/streamlit_app.py
```

### Command line

```bash
elecforecast info
```

```bash
elecforecast forecast --model sarima --horizon 24
```

```bash
elecforecast backtest --folds 5 --horizon 12
```

### Tests

```bash
pytest -q
```

## Metrics, and one that isn't here

| Metric | Why |
|---|---|
| **RMSE** | In series units; penalises large misses |
| **MAE** | In series units; robust to outliers |
| **MAPE** | Scale-free percentage error |
| **MASE** | Scaled against the baseline — *below 1.0 wins, above 1.0 loses* |

**"Accuracy" is deliberately absent.** This is a regression problem; accuracy is a
classification metric. An earlier version of this project claimed "90% accuracy" — a
number that means nothing on a continuous series, and the first thing that would fall
apart under an interview follow-up.

## What I'd still improve

- **Grid-search the SARIMA orders.** `(1,1,1)(1,1,1,12)` is a sensible default, not a
  tuned choice — AIC-based selection would likely improve on it.
- **Add exogenous regressors.** Temperature and heating/cooling degree days drive this
  series directly; SARIMAX would let the model use them.
- **Widen the model set.** Prophet and gradient-boosted trees on lag features are the
  obvious next comparisons.
- **Prediction intervals for all models**, not just SARIMA — bootstrapped residuals
  would cover Holt-Winters and the baseline.

## The original notebook

`notebooks/exploratory-analysis.ipynb` is the exploratory work this grew out of, kept
for provenance along with the original report and figures. The package supersedes it —
the notebook used plain ARIMA with no baseline and no held-out evaluation, which is
exactly what produced an unsupportable accuracy claim.

## License

[MIT](LICENSE) · Data: [FRED IPG2211A2N](https://fred.stlouisfed.org/series/IPG2211A2N)
