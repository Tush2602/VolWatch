from tracking.pyfunc_wrapper import ForecasterModel


class Retrainer:
    """Runs a challenger trial, records it in MLflow, and promotes the challenger only if it wins."""

    def __init__(self, tracker, registry, trial):
        self.tracker = tracker
        self.registry = registry
        self.trial = trial

    def run(self, labeled, champion, data_end: str) -> dict:
        result = self.trial.run(labeled, champion.model)
        run_id = self.tracker.log_run(
            run_name=f"challenger_{data_end}",
            params={**result.challenger.params, "log_target": result.challenger.log_target},
            metrics={"holdout_qlike": result.challenger_qlike, "champion_holdout_qlike": result.champion_qlike},
            tags={
                "stage": "nightly_trial",
                "fit_end": str(result.fit_end.date()),
                "holdout_start": str(result.holdout_start.date()),
                "data_end": data_end,
                "promoted": str(result.wins),
            },
            model=ForecasterModel(result.challenger) if result.wins else None,
            registered_model_name=self.registry.model_name if result.wins else None,
        )

        outcome = {
            "promoted": result.wins,
            "challenger_qlike": round(result.challenger_qlike, 4),
            "champion_qlike": round(result.champion_qlike, 4),
            "previous_champion": champion.version,
            "run_id": run_id,
        }
        if result.wins:
            version = self.registry.version_for_run(run_id)
            self.registry.set_version_tag(version, "fit_end", str(result.fit_end.date()))
            self.registry.set_champion(version)
            outcome["new_version"] = version
        return outcome