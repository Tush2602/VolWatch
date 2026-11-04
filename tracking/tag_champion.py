import pandas as pd

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.splitter import Timesplitter
from tracking.register_champion import MODEL_NAME
from tracking.registry import ModelRegistry
from tracking.tracker import ExperimentTracker


def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    splits = Timesplitter().split(table)
    fit_end = pd.concat([splits.train, splits.valid]).index.max()

    registry = ModelRegistry(ExperimentTracker(), MODEL_NAME)
    champion = registry.get_champion()
    if champion is None:
        raise RuntimeError("no champion registered yet")

    registry.set_version_tag(int(champion.version), "fit_end", str(fit_end.date()))
    print(f"champion version {champion.version}: fit_end = {fit_end.date()}")


if __name__ == "__main__":
    main()