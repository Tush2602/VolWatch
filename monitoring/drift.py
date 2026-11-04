from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from models.metrics import Metrics


@dataclass
class FeatureDriftResult:
    statistics: dict
    p_values: dict
    flagged: list
    drifted: bool


@dataclass
class PerformanceDriftResult:
    n_obs: int
    champion_loss: float | None
    benchmark_loss: float | None
    ratio: float | None
    enough_data: bool
    drifted: bool


class FeatureDriftDetector:
    """Two-sample KS test per feature, with a multiple-testing correction and an effect-size gate."""

    def __init__(self, features: list, alpha: float = 0.05, min_effect: float = 0.40,
                 min_features: int = 3, min_rows: int = 30):
        self.features = features
        self.alpha = alpha
        self.min_effect = min_effect
        self.min_features = min_features
        self.min_rows = min_rows

    def detect(self, reference: pd.DataFrame, recent: pd.DataFrame) -> FeatureDriftResult:
        if len(reference) < self.min_rows or len(recent) < self.min_rows:
            raise ValueError(f"need at least {self.min_rows} rows in both windows")

        statistics, p_values = {}, {}
        for feature in self.features:
            result = ks_2samp(reference[feature].dropna(), recent[feature].dropna())
            statistics[feature] = float(result.statistic)
            p_values[feature] = float(result.pvalue)

        p_limit = self.alpha / len(self.features)  # Bonferroni
        flagged = [f for f in self.features
                   if statistics[f] >= self.min_effect and p_values[f] < p_limit]
        return FeatureDriftResult(statistics, p_values, flagged, len(flagged) >= self.min_features)


class PerformanceDriftDetector:
    """Compares the champion's recent QLIKE with the benchmark's over the same days."""

    def __init__(self, max_ratio: float = 1.35, min_obs: int = 40):
        self.max_ratio = max_ratio
        self.min_obs = min_obs

    def detect(self, actual, champion, benchmark) -> PerformanceDriftResult:
        a = np.asarray(actual, dtype=float)
        c = np.asarray(champion, dtype=float)
        b = np.asarray(benchmark, dtype=float)
        usable = np.isfinite(a) & np.isfinite(c) & np.isfinite(b)
        n = int(usable.sum())
        if n < self.min_obs:
            return PerformanceDriftResult(n, None, None, None, False, False)

        champion_loss = Metrics.qlike(a[usable], c[usable])
        benchmark_loss = Metrics.qlike(a[usable], b[usable])
        ratio = champion_loss / max(benchmark_loss, 1e-9)
        return PerformanceDriftResult(n, champion_loss, benchmark_loss, ratio, True, ratio > self.max_ratio)


@dataclass
class RetrainDecision:
    retrain: bool
    triggers: list
    suppressed_by_cooldown: bool
    feature: FeatureDriftResult
    performance: PerformanceDriftResult

    def to_dict(self) -> dict:
        """Plain values only, so Phase 6 can store it as JSON."""
        return {
            "retrain": bool(self.retrain),
            "triggers": list(self.triggers),
            "suppressed": bool(self.suppressed_by_cooldown),
            "feature_drift": bool(self.feature.drifted),
            "flagged": list(self.feature.flagged),
            "stats": dict(self.feature.statistics),
            "max_ks": float(max(self.feature.statistics.values(), default=0.0)),
            "performance_drift": bool(self.performance.drifted),
            "perf_ratio": self.performance.ratio,
            "n_obs": int(self.performance.n_obs),
        }


class RetrainPolicy:
    """Decides whether to retrain. mode: 'either' signal, 'performance' only, or 'both' signals together."""

    MODES = ("either", "performance", "both")

    def __init__(self, cooldown_days: int = 20, mode: str = "either", soft_ratio: float = 1.15):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.cooldown_days = cooldown_days
        self.mode = mode
        self.soft_ratio = soft_ratio

    def decide(self, feature: FeatureDriftResult, performance: PerformanceDriftResult,
               days_since_retrain: int) -> RetrainDecision:
        ratio = performance.ratio
        feature_reason = f"feature drift: {len(feature.flagged)} features flagged ({', '.join(feature.flagged)})"
        perf_reason = f"performance drift: champion QLIKE is {ratio:.2f}x the benchmark's" if ratio is not None else ""

        if self.mode == "performance":
            active = performance.drifted
            reasons = [perf_reason] if active else []
        elif self.mode == "both":
            active = feature.drifted and ratio is not None and ratio > self.soft_ratio
            reasons = [feature_reason, f"performance corroborates: QLIKE ratio {ratio:.2f}"] if active else []
        else:
            reasons = [r for r, hit in ((feature_reason, feature.drifted), (perf_reason, performance.drifted)) if hit]
            active = bool(reasons)

        in_cooldown = days_since_retrain < self.cooldown_days
        return RetrainDecision(
            retrain=active and not in_cooldown,
            triggers=reasons,
            suppressed_by_cooldown=active and in_cooldown,
            feature=feature,
            performance=performance,
        )


class DriftMonitor:
    """One call that runs both detectors and the policy. The nightly flow will use this."""

    def __init__(self, feature_detector: FeatureDriftDetector,
                 performance_detector: PerformanceDriftDetector, policy: RetrainPolicy):
        self.feature_detector = feature_detector
        self.performance_detector = performance_detector
        self.policy = policy

    def check(self, reference: pd.DataFrame, recent: pd.DataFrame, champion_forecast,
              benchmark_forecast, days_since_retrain: int) -> RetrainDecision:
        feature = self.feature_detector.detect(reference, recent)
        performance = self.performance_detector.detect(recent["target"], champion_forecast, benchmark_forecast)
        return self.policy.decide(feature, performance, days_since_retrain)