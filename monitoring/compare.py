from monitoring.drift import FeatureDriftResult, PerformanceDriftResult, RetrainDecision


def _decision(retrain: bool, triggers: list) -> RetrainDecision:
    return RetrainDecision(
        retrain=retrain, triggers=triggers, suppressed_by_cooldown=False,
        feature=FeatureDriftResult({}, {}, [], False),
        performance=PerformanceDriftResult(0, None, None, None, False, False),
    )


class NeverMonitor:
    """The control group: the first champion forever."""

    def check(self, reference, recent, champion, benchmark, days_since_retrain) -> RetrainDecision:
        return _decision(False, [])


class ScheduleMonitor:
    """The simple alternative: retrain on a fixed calendar, whatever the data does."""

    def __init__(self, every_days: int):
        self.every_days = every_days

    def check(self, reference, recent, champion, benchmark, days_since_retrain) -> RetrainDecision:
        due = days_since_retrain >= self.every_days
        return _decision(due, [f"scheduled: every {self.every_days} trading days"] if due else [])