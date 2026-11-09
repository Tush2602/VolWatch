import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:   # lets `streamlit run dashboard/app.py` import the dashboard package
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from dashboard.charts import ChartBuilder
from dashboard.client import ApiClient, ApiUnavailable

API_URL = os.getenv("API_URL", "https://volwatch-api.onrender.com")
RESULTS = json.loads((Path(__file__).parent / "results.json").read_text())

st.set_page_config(page_title="VolWatch", page_icon="📈", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def load_api_data(base_url: str) -> dict:
    client = ApiClient(base_url)
    return {"predict": client.predict(), "history": client.history(30) or [], "status": client.status()}


def dash(value):
    return "—" if value is None else value


st.title("VolWatch")
st.caption("Self-monitoring forecasts of next-session NIFTY 50 volatility. "
           "A technical demonstration, not investment advice.")

try:
    with st.spinner("Waking up the forecast API (free hosting can take up to a minute)..."):
        data = load_api_data(API_URL)
except ApiUnavailable:
    st.error("The forecast API is not responding right now. Please try again in a minute.")
    st.stop()

predict, history, status = data["predict"], data["history"], data["status"]
if history and "source" not in history[0]:
    st.error("The API is running an older version than this dashboard expects. Please try again shortly.")
    st.stop()

# ---- headline ----
if predict is None:
    st.info("No live forecast has been published yet.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next-session volatility", f"{predict['forecast_vol']:.1f}%",
              help="Range-based estimate from the day's high and low, annualized, in the same units as India VIX.")
    c2.metric("Forecast issued", predict["as_of_date"])
    st.caption("This is the model's forecast for the next trading session after that date. "
               "The dashboard updates on weekdays after the market closes (about 18:00 IST).")
    c3.metric("Model version", f"v{predict['model_version']}")
    c4.metric("Freshness", "Stale" if predict["stale"] else "Fresh",
              delta=f"{predict['age_days']} days old", delta_color="off")


# ---- history ----
st.subheader("Recent forecasts vs what happened")
if history:
    st.plotly_chart(ChartBuilder.forecast_vs_realized(history), use_container_width=True)
    st.caption("Filled markers are live nightly forecasts. Hollow markers are a backtest: the deployed model "
               "(trained only through 2019) applied to each past day using only information available that day. "
               "The line is the volatility that actually followed.")
else:
    st.info("History will appear here once forecasts have been stored.")

# ---- model comparison ----
st.subheader("How the models compare")
st.caption(f"Scored once on a sealed test set: {RESULTS['period']} ({RESULTS['n_days']:,} trading days). "
           f"{RESULTS['note']}")
left, right = st.columns([3, 2])
with left:
    table = pd.DataFrame(RESULTS["models"]).drop(columns="kind").rename(
        columns={"model": "Model", "rmse": "RMSE", "mae": "MAE", "qlike": "QLIKE"})
    st.dataframe(table.sort_values("QLIKE"), hide_index=True, use_container_width=True)
with right:
    st.plotly_chart(ChartBuilder.model_comparison(RESULTS["models"]), use_container_width=True)
st.caption("Learned models clearly beat naive baselines. The market's own benchmark (India VIX, rescaled) is "
           "still slightly ahead on QLIKE, and the deployed model leads on MAE.")

# ---- system health ----
st.subheader("System health")
if status["healthy"]:
    st.success("The nightly pipeline is healthy.")
else:
    st.warning("The nightly pipeline needs attention: the last run did not succeed recently.")

pipeline, model, drift = status["pipeline"], status["model"], status["drift"]
h1, h2, h3 = st.columns(3)
h1.metric("Last run", dash(pipeline["last_run_status"]))
h1.caption(f"Hours since last success: {dash(pipeline['hours_since_last_success'])}")
h2.metric("Champion model", f"v{model['champion_version']}" if model["champion_version"] else "—")
h2.caption(f"Last retrain decision: {dash(model['last_retrain_decision'])}")
warning = drift["feature_drift_warning"]
h3.metric("Input drift warning", "—" if warning is None else ("Yes" if warning else "No"))
h3.caption("Informational. Retraining is triggered by performance, not input drift.")

with st.expander("What this does not do"):
    st.markdown(
        "- Covers the NIFTY 50 only, and forecasts only the next trading session.\n"
        "- Forecasts volatility, not prices or returns. It makes no trading or profit claims.\n"
        "- Volatility here is a range-based proxy from daily high and low, so it understates overnight gaps.\n"
        "- The learned model roughly ties a rescaled India VIX benchmark. That is the honest result.\n"
        "- The free API sleeps when idle, so the first load can take up to a minute."
    )

st.markdown(f"[Source code](https://github.com/Tush2602/VolWatch) · [API docs]({API_URL}/docs)")