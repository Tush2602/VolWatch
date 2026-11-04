from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from ingest.backfill import TICKERS, BackFillJob
from ingest.fetch import MarketDataFetcher
from ingest.validate import BarFilter, QualityChecker
from models.baselines import ScaledVIXForecaster
from models.run_final_test import XGB_PARAMS
from models.xgb_model import XGBForecaster
from monitoring.production import PRODUCTION, build_production_monitor
from pipeline.champion import ChampionLoader
from pipeline.nightly import NightlyPipeline
from pipeline.retrainer import Retrainer
from pipeline.trial import ChallengerTrial
from tracking.register_champion import MODEL_NAME
from tracking.registry import ModelRegistry
from tracking.tracker import ExperimentTracker


def build_pipeline() -> NightlyPipeline:
    db = Database()
    builder = FeatureBuilder(VolatilityEstimator())
    tracker = ExperimentTracker()
    registry = ModelRegistry(tracker, MODEL_NAME)

    ingest = BackFillJob(
        TICKERS, MarketDataFetcher(start="2008-01-01", retries=3), BarFilter(),
        QualityChecker(max_gap_days=5, max_stale_days=5), db,
    )
    trial = ChallengerTrial(
        lambda: XGBForecaster(log_target=False, **XGB_PARAMS),
        holdout_days=PRODUCTION["recent_days"],
    )

    return NightlyPipeline(
        ingest=ingest,
        table_provider=lambda: builder.build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX")),
        champion_loader=ChampionLoader(registry),
        monitor=build_production_monitor(builder.feature_names),
        retrainer=Retrainer(tracker, registry, trial),
        store=db,
        benchmark_factory=ScaledVIXForecaster,
        recent_days=PRODUCTION["recent_days"],
        reference_days=PRODUCTION["reference_days"],
    )