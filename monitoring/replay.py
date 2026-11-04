import pandas as pd


class PolicyReplay:
    """Walks through history in steps, applying a retrain policy the way the nightly job will."""

    def __init__(self, table: pd.DataFrame, monitor, model_factory, benchmark_factory,
                 fit_end, start, end=None, step: int = 5, recent_days: int = 60, reference_days: int = 750):
        self.labeled = table.dropna(subset=["target"])
        self.monitor = monitor
        self.model_factory = model_factory
        self.benchmark_factory = benchmark_factory
        self.fit_end = pd.Timestamp(fit_end)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end) if end is not None else None
        self.step = step
        self.recent_days = recent_days
        self.reference_days = reference_days
        self.forecasts = pd.DataFrame(columns=["actual", "forecast"])

    def run(self) -> pd.DataFrame:
        labeled = self.labeled
        fit = labeled[labeled.index <= self.fit_end]
        model = self.model_factory().fit(fit)
        benchmark_model = self.benchmark_factory().fit(fit)
        last_retrain_pos = len(fit) - 1   # the initial fit counts as the most recent retrain

        in_range = labeled.index >= self.start
        if self.end is not None:
            in_range &= labeled.index <= self.end
        check_dates = labeled.index[in_range][:: self.step]

        records, forecast_parts = [], []
        for date in check_dates:
            pos = labeled.index.get_loc(date)
            known = labeled.iloc[:pos]   # only rows strictly before today have a known label
            if len(known) < self.recent_days:
                continue

            recent = known.iloc[-self.recent_days:]
            reference = fit.iloc[-self.reference_days:]
            decision = self.monitor.check(
                reference, recent, model.predict(recent), benchmark_model.predict(recent),
                days_since_retrain=pos - last_retrain_pos,
            )
            records.append({"date": date, **decision.to_dict()})

            if decision.retrain:
                fit = known
                model = self.model_factory().fit(fit)
                benchmark_model = self.benchmark_factory().fit(fit)
                last_retrain_pos = pos

            forward = labeled.iloc[pos: pos + self.step]
            if self.end is not None:
                forward = forward[forward.index <= self.end]
            if len(forward):
                forecast_parts.append(pd.DataFrame({"actual": forward["target"], "forecast": model.predict(forward)}))

        if forecast_parts:
            self.forecasts = pd.concat(forecast_parts)
        return pd.DataFrame(records)