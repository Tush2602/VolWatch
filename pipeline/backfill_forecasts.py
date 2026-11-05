import sys

import numpy as np

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from pipeline.champion import ChampionLoader
from tracking.register_champion import MODEL_NAME
from tracking.registry import ModelRegistry
from tracking.tracker import ExperimentTracker


def main(days: int = 30) -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    champion = ChampionLoader(ModelRegistry(ExperimentTracker(), MODEL_NAME)).load()

    rows = table.iloc[-(days + 1):-1]   # the newest day is left to the live nightly forecast
    if rows.index.min() <= champion.fit_end:
        raise RuntimeError("backfill window overlaps the champion's training data")

    predictions = np.asarray(champion.model.predict(rows)).ravel()
    for date, value in zip(rows.index, predictions):
        db.save_forecast("^NSEI", date, champion.version, value, source="backfill", overwrite=False)
    print(f"backfilled {len(rows)} forecasts, {rows.index.min().date()} to {rows.index.max().date()}, "
          f"champion v{champion.version} (trained through {champion.fit_end.date()})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)