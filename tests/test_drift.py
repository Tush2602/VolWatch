import json

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (DriftMonitor, FeatureDriftDetector, FeatureDriftResult,
                              PerformanceDriftDetector, PerformanceDriftResult, RetrainPolicy)

from monitoring.production import build_production_monitor

FEATURES = ["a", "b", "c", "d"]


def make_frame(n, shift=0.0, shifted=FEATURES, seed=0):
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(0, 1, n) + (shift if f in shifted else 0.0) for f in FEATURES}
    return pd.DataFrame(data, index=pd.bdate_range("2020-01-01", periods=n))


def feature_result(drifted):
    return FeatureDriftResult({"a": 0.5}, {"a": 0.0}, ["a"] if drifted else [], drifted)


def perf_result(drifted):
    return PerformanceDriftResult(60, 1.0, 0.5, 2.0, True, drifted)


# ---------- feature drift ----------

def test_same_distribution_is_not_flagged():
    result = FeatureDriftDetector(FEATURES).detect(make_frame(750, seed=1), make_frame(60, seed=2))
    assert result.flagged == []
    assert result.drifted is False


def test_large_shift_in_every_feature_is_flagged():
    result = FeatureDriftDetector(FEATURES).detect(make_frame(750, seed=1), make_frame(60, shift=3, seed=2))
    assert sorted(result.flagged) == FEATURES
    assert result.drifted is True


def test_one_drifting_feature_is_below_the_feature_count_threshold():
    recent = make_frame(60, shift=3, shifted=["a"], seed=2)
    result = FeatureDriftDetector(FEATURES).detect(make_frame(750, seed=1), recent)
    assert result.flagged == ["a"]
    assert result.drifted is False


def test_tiny_shift_is_not_flagged_even_when_p_value_is_tiny():
    detector = FeatureDriftDetector(FEATURES)
    result = detector.detect(make_frame(5000, seed=1), make_frame(5000, shift=0.15, seed=2))
    assert result.p_values["a"] < detector.alpha / len(FEATURES)   # p-value alone would flag it
    assert result.flagged == []                                      # the effect-size gate does not


def test_too_few_rows_raises():
    with pytest.raises(ValueError):
        FeatureDriftDetector(FEATURES).detect(make_frame(10), make_frame(60))


# ---------- performance drift ----------

ACTUAL = np.random.default_rng(0).uniform(5, 15, 100)


def test_champion_matching_benchmark_is_not_drift():
    result = PerformanceDriftDetector().detect(ACTUAL, ACTUAL * 0.9, ACTUAL * 0.9)
    assert result.ratio == pytest.approx(1.0)
    assert result.drifted is False


def test_champion_much_worse_than_benchmark_is_drift():
    result = PerformanceDriftDetector().detect(ACTUAL, ACTUAL * 0.5, ACTUAL * 0.9)
    assert result.ratio > 1.35
    assert result.drifted is True


def test_not_enough_observations_means_no_decision():
    result = PerformanceDriftDetector().detect(ACTUAL[:10], ACTUAL[:10] * 0.5, ACTUAL[:10] * 0.9)
    assert result.enough_data is False
    assert result.drifted is False


def test_missing_forecasts_are_ignored():
    champion = ACTUAL * 0.9
    champion[:5] = np.nan   # like the first rows of a sequence model
    result = PerformanceDriftDetector().detect(ACTUAL, champion, ACTUAL * 0.9)
    assert result.n_obs == 95


# ---------- policy ----------

def test_no_triggers_means_no_retrain():
    decision = RetrainPolicy(20).decide(feature_result(False), perf_result(False), days_since_retrain=100)
    assert decision.retrain is False
    assert decision.suppressed_by_cooldown is False
    assert decision.triggers == []


def test_trigger_after_cooldown_retrains_and_lists_reasons():
    decision = RetrainPolicy(20).decide(feature_result(True), perf_result(True), days_since_retrain=25)
    assert decision.retrain is True
    assert len(decision.triggers) == 2


def test_trigger_inside_cooldown_is_suppressed():
    decision = RetrainPolicy(20).decide(feature_result(True), perf_result(False), days_since_retrain=5)
    assert decision.retrain is False
    assert decision.suppressed_by_cooldown is True
    assert len(decision.triggers) == 1

def perf_with_ratio(ratio, drifted):
    return PerformanceDriftResult(60, 1.0, 0.5, ratio, True, drifted)


def test_performance_mode_ignores_feature_drift_alone():
    decision = RetrainPolicy(20, mode="performance").decide(feature_result(True), perf_with_ratio(1.0, False), 100)
    assert decision.retrain is False


def test_both_mode_needs_feature_drift_and_slipping_performance():
    policy = RetrainPolicy(20, mode="both", soft_ratio=1.15)
    assert policy.decide(feature_result(True), perf_with_ratio(1.3, False), 100).retrain is True
    assert policy.decide(feature_result(True), perf_with_ratio(1.0, False), 100).retrain is False
    assert policy.decide(feature_result(False), perf_with_ratio(2.0, True), 100).retrain is False


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        RetrainPolicy(20, mode="sometimes")

def recent_with_target(shift, seed):
    recent = make_frame(60, shift=shift, seed=seed)
    recent["target"] = np.random.default_rng(3).uniform(5, 15, 60)
    return recent


def test_production_policy_warns_on_feature_drift_but_does_not_retrain():
    recent = recent_with_target(shift=3, seed=2)
    forecast = recent["target"] * 0.9
    decision = build_production_monitor(FEATURES).check(
        make_frame(750, seed=1), recent, forecast, forecast, days_since_retrain=100)
    assert decision.feature.drifted is True
    assert decision.retrain is False


def test_production_policy_retrains_when_champion_underperforms_benchmark():
    recent = recent_with_target(shift=0, seed=5)
    decision = build_production_monitor(FEATURES).check(
        make_frame(750, seed=1), recent, recent["target"] * 0.5, recent["target"] * 0.9, days_since_retrain=100)
    assert decision.feature.drifted is False
    assert decision.retrain is True


# ---------- the combined monitor ----------

def test_monitor_end_to_end_and_decision_is_json_serializable():
    reference, recent = make_frame(750, seed=1), make_frame(60, shift=3, seed=2)
    recent["target"] = np.random.default_rng(3).uniform(5, 15, 60)
    forecast = recent["target"] * 0.9

    monitor = DriftMonitor(FeatureDriftDetector(FEATURES), PerformanceDriftDetector(), RetrainPolicy(20))
    decision = monitor.check(reference, recent, forecast, forecast, days_since_retrain=100)

    assert decision.retrain is True
    assert decision.triggers[0].startswith("feature drift")
    json.dumps(decision.to_dict())   # raises if any value is not plain JSON