"""Interactive forecasting demo.

Runs entirely on the committed dataset - no API keys, no database, no network.
Open the link and everything is already there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elecforecast.analysis import (  # noqa: E402
    adf_test,
    decompose,
    make_stationary,
    rolling_stats,
    seasonal_profile,
    yoy_growth,
)
from elecforecast.data import describe, load_series  # noqa: E402
from elecforecast.evaluate import compare  # noqa: E402
from elecforecast.models import MODELS, Sarima, SeasonalNaive, build  # noqa: E402

st.set_page_config(
    page_title="Electricity Production Forecasting",
    page_icon="⚡",
    layout="wide",
)

INK = "#2563eb"
ACCENT = "#f97316"
MUTED = "#94a3b8"


@st.cache_data(show_spinner=False)
def get_series() -> pd.Series:
    return load_series()


@st.cache_data(show_spinner="Backtesting every model…")
def get_comparison(horizon: int, folds: int) -> pd.DataFrame:
    return compare(MODELS, get_series(), horizon=horizon, n_folds=folds)


@st.cache_data(show_spinner="Fitting model…")
def get_forecast(model_key: str, horizon: int) -> pd.DataFrame:
    series = get_series()
    model = build(model_key).fit(series)
    if isinstance(model, Sarima):
        return model.predict_interval(horizon)
    forecast = model.predict(horizon)
    return pd.DataFrame({"forecast": forecast})


def line(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)")
    return fig


series = get_series()
facts = describe(series)

st.title("⚡ Electricity Production Forecasting")
st.markdown(
    "Forecasting monthly **US electric &amp; gas utility production** "
    "(FRED series `IPG2211A2N`) with SARIMA, Holt-Winters, and a seasonal-naive "
    "baseline — evaluated by walk-forward backtesting."
)

cols = st.columns(4)
cols[0].metric("Observations", facts["observations"])
cols[1].metric("Coverage", f"{facts['start']} → {facts['end']}")
cols[2].metric("Years", facts["years"])
cols[3].metric("Mean index", f"{facts['mean']:.1f}")

overview, diagnostics, forecast_tab, backtest = st.tabs(
    ["Overview", "Diagnostics", "Forecast", "Backtest"]
)

# ---------------------------------------------------------------- Overview
with overview:
    st.subheader("The series")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=series.index, y=series.to_numpy(), name="Production", line=dict(color=INK, width=1.4))
    )
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.rolling(12).mean().to_numpy(),
            name="12-month rolling mean",
            line=dict(color=ACCENT, width=2.4),
        )
    )
    st.plotly_chart(line(fig), use_container_width=True)

    st.markdown(
        "Two things are visible immediately, and they drive every modelling decision "
        "that follows: a **long-run upward trend**, and a **strong annual cycle** "
        "from heating and cooling demand. Neither is optional to handle — a model "
        "that ignores the seasonality will lose to simply repeating last year."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Average production by calendar month**")
        profile = seasonal_profile(series)
        pf = go.Figure()
        pf.add_trace(go.Bar(x=profile.index, y=profile["mean"], marker_color=INK, name="Mean"))
        st.plotly_chart(line(pf), use_container_width=True)
        st.caption(
            "Twin peaks — winter heating and summer cooling — with mild shoulder "
            "months in spring and autumn."
        )

    with right:
        st.markdown("**Year-over-year growth (%)**")
        growth = yoy_growth(series).dropna()
        gf = go.Figure()
        gf.add_trace(
            go.Scatter(x=growth.index, y=growth.to_numpy(), name="YoY %", line=dict(color=ACCENT, width=1.4))
        )
        gf.add_hline(y=0, line_dash="dash", line_color=MUTED)
        st.plotly_chart(line(gf), use_container_width=True)
        st.caption(
            "Comparing each month to the same month a year earlier cancels the "
            "seasonal cycle, leaving underlying growth and shocks visible."
        )

# ------------------------------------------------------------- Diagnostics
with diagnostics:
    st.subheader("Is the series stationary?")
    st.markdown(
        "ARIMA assumes a **stationary** series — stable mean and variance over time. "
        "The Augmented Dickey-Fuller test's null hypothesis is that a unit root is "
        "present, so a **low p-value is the good outcome**: it rejects "
        "non-stationarity."
    )

    raw = adf_test(series)
    differenced, n_diffs, after = make_stationary(series)

    a, b = st.columns(2)
    with a:
        st.markdown("**Raw series**")
        st.metric("ADF p-value", f"{raw.p_value:.4g}")
        (st.success if raw.is_stationary else st.error)(raw.verdict)
    with b:
        st.markdown(f"**After {n_diffs} difference(s)**")
        st.metric("ADF p-value", f"{after.p_value:.4g}")
        (st.success if after.is_stationary else st.error)(after.verdict)

    st.info(
        f"The number of differences required (**d = {n_diffs}**) is not just a "
        "transformation — it is the `d` parameter the ARIMA model needs. The "
        "diagnostic *is* the parameter selection."
    )

    st.markdown("**Rolling mean and standard deviation** — the visual counterpart")
    window = st.slider("Rolling window (months)", 3, 36, 12, key="roll")
    stats = rolling_stats(differenced, window=window).dropna()
    rf = go.Figure()
    rf.add_trace(go.Scatter(x=stats.index, y=stats["series"], name="Differenced", line=dict(color=MUTED, width=0.9)))
    rf.add_trace(go.Scatter(x=stats.index, y=stats["rolling_mean"], name="Rolling mean", line=dict(color=INK, width=2.4)))
    rf.add_trace(go.Scatter(x=stats.index, y=stats["rolling_std"], name="Rolling std", line=dict(color=ACCENT, width=2.4)))
    st.plotly_chart(line(rf), use_container_width=True)
    st.caption("A flat rolling mean and a stable rolling standard deviation are what stationarity looks like.")

    st.subheader("Seasonal decomposition")
    result = decompose(series)
    for label, component, colour in [
        ("Trend", result.trend, INK),
        ("Seasonal", result.seasonal, ACCENT),
        ("Residual", result.resid, MUTED),
    ]:
        cf = go.Figure()
        cf.add_trace(go.Scatter(x=component.index, y=component.to_numpy(), name=label, line=dict(color=colour, width=1.4)))
        cf.update_layout(title=label, height=240, margin=dict(l=10, r=10, t=36, b=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(cf, use_container_width=True)

    st.caption(
        "Trend rises steadily and the seasonal component repeats almost identically "
        "each year. What's left in the residual is what any model has to actually predict."
    )

# ---------------------------------------------------------------- Forecast
with forecast_tab:
    st.subheader("Forecast forward")

    c1, c2 = st.columns([2, 1])
    with c1:
        model_key = st.selectbox(
            "Model",
            list(MODELS),
            index=list(MODELS).index("sarima"),
            format_func=lambda k: build(k).name,
        )
    with c2:
        horizon = st.slider("Horizon (months)", 6, 36, 24, step=6)

    st.caption(build(model_key).description)

    frame = get_forecast(model_key, horizon)
    recent = series.iloc[-96:]

    ff = go.Figure()
    ff.add_trace(go.Scatter(x=recent.index, y=recent.to_numpy(), name="Observed", line=dict(color=INK, width=1.6)))
    ff.add_trace(go.Scatter(x=frame.index, y=frame["forecast"].to_numpy(), name="Forecast", line=dict(color=ACCENT, width=2.6)))

    if {"lower", "upper"}.issubset(frame.columns):
        ff.add_trace(
            go.Scatter(
                x=list(frame.index) + list(frame.index[::-1]),
                y=list(frame["upper"]) + list(frame["lower"][::-1]),
                fill="toself",
                fillcolor="rgba(249,115,22,0.16)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% interval",
                hoverinfo="skip",
            )
        )

    st.plotly_chart(line(ff), use_container_width=True)

    if {"lower", "upper"}.issubset(frame.columns):
        st.markdown(
            "The shaded band is the **95% confidence interval**, and it widens the "
            "further out the forecast goes. That widening is the honest part — a "
            "point forecast alone invites false confidence about month 24."
        )
    else:
        st.info(
            f"{build(model_key).name} produces point forecasts only. Switch to "
            "**SARIMA** to see confidence intervals."
        )

    with st.expander("Forecast values"):
        st.dataframe(frame.round(2), use_container_width=True)

# ---------------------------------------------------------------- Backtest
with backtest:
    st.subheader("Which model actually wins?")
    st.markdown(
        "Each model is re-fitted at several successive origins and scored on the "
        "months that follow — so no fold can see its own test period. "
        "**MASE below 1.0 beats the seasonal-naive baseline; above 1.0 loses to it.**"
    )

    c1, c2 = st.columns(2)
    h = c1.slider("Forecast horizon per fold", 6, 24, 12, step=6, key="bt_h")
    n = c2.slider("Number of folds", 2, 6, 5, key="bt_n")

    table = get_comparison(h, n)

    def mark_baseline(column: pd.Series) -> list[str]:
        """Green where the model beats the baseline, red where it loses."""
        return [
            "background-color: rgba(34,197,94,0.18)"
            if v < 1.0
            else "background-color: rgba(239,68,68,0.18)"
            for v in column
        ]

    st.dataframe(
        table.style.apply(mark_baseline, subset=["MASE"]),
        use_container_width=True,
    )

    best = table.iloc[0]
    if best["MASE"] < 1.0:
        st.success(
            f"**{best['model']}** wins with MASE **{best['MASE']}** — it beats the "
            f"seasonal-naive baseline, with mean absolute percentage error of "
            f"**{best['MAPE %']}%**."
        )
    else:
        st.warning(
            f"No model beats the baseline at this setting — best MASE is "
            f"**{best['MASE']}**. On a series this seasonal, that is a genuine "
            "possibility, and worth reporting rather than hiding."
        )

    baseline_name = SeasonalNaive().name
    losers = table[(table["MASE"] >= 1.0) & (table["model"] != baseline_name)][
        "model"
    ].tolist()
    if losers:
        st.markdown(
            "Models that **lose to simply repeating last year**: "
            + ", ".join(f"`{m}`" for m in losers)
            + ". This is exactly why the baseline is included — without it, a "
            "respectable-looking MAPE hides the fact that a model adds nothing."
        )

    st.caption(
        f"MASE is scaled against the *in-sample* seasonal-naive error, so the "
        f"baseline's own score ({table.loc[table['model'] == baseline_name, 'MASE'].iloc[0]}) "
        "is near 1.0 rather than exactly 1.0."
    )

st.divider()
st.caption(
    "Built by Sahil Chakraborty · "
    "[Source](https://github.com/sahil7359/electricity_consumption_forecasting) · "
    "Data: FRED IPG2211A2N"
)
