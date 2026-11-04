from monitoring.drift import DriftMonitor, FeatureDriftDetector, PerformanceDriftDetector, RetrainPolicy

# Chosen from the replay comparison in monitoring/run_compare.py.
# Performance drift triggers retraining. Feature drift is reported as a warning only.
PRODUCTION = {
    "recent_days": 60,
    "reference_days": 750,
    "max_ratio": 1.35,
    "min_obs": 40,
    "cooldown_days": 20,
    "min_effect": 0.40,
    "min_features": 3,
}


def build_production_monitor(features: list) -> DriftMonitor:
    return DriftMonitor(
        FeatureDriftDetector(features, min_effect=PRODUCTION["min_effect"], min_features=PRODUCTION["min_features"]),
        PerformanceDriftDetector(max_ratio=PRODUCTION["max_ratio"], min_obs=PRODUCTION["min_obs"]),
        RetrainPolicy(cooldown_days=PRODUCTION["cooldown_days"], mode="performance"),
    )