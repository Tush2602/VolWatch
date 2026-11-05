import pandas as pd
import plotly.graph_objects as go


class ChartBuilder:
    REALIZED = "Realized volatility"
    LIVE = "Forecast (live)"
    BACKFILL = "Forecast (backfilled)"

    @classmethod
    def forecast_vs_realized(cls, history: list) -> go.Figure:
        df = pd.DataFrame(history)
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["as_of_date"], y=df["realized_vol"], name=cls.REALIZED, mode="lines"))
        for source, name, symbol in (("backfill", cls.BACKFILL, "circle-open"), ("live", cls.LIVE, "circle")):
            part = df[df["source"] == source]
            if len(part):
                fig.add_trace(go.Scatter(x=part["as_of_date"], y=part["forecast_vol"], name=name,
                                         mode="markers", marker={"symbol": symbol, "size": 9}))
        fig.update_layout(
            xaxis_title="Day the forecast was issued", yaxis_title="Volatility (annualized %)",
            legend={"orientation": "h", "y": -0.25}, margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=380,
        )
        return fig

    @staticmethod
    def model_comparison(models: list, metric: str = "qlike") -> go.Figure:
        df = pd.DataFrame(models).sort_values(metric)
        colors = ["#1D9E75" if kind == "deployed" else "#888780" for kind in df["kind"]]
        fig = go.Figure(go.Bar(x=df[metric], y=df["model"], orientation="h", marker_color=colors))
        fig.update_layout(
            xaxis_title=f"{metric.upper()} (lower is better)", yaxis={"autorange": "reversed"},
            margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=320,
        )
        return fig