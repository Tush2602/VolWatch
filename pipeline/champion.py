from dataclasses import dataclass

import pandas as pd


@dataclass
class ChampionBundle:
    version: int
    fit_end: pd.Timestamp
    model: object   # anything with predict(DataFrame)


class ChampionLoader:
    """Loads the registry's champion together with the date its training data ended."""

    def __init__(self, registry):
        self.registry = registry

    def load(self) -> ChampionBundle:
        champion = self.registry.get_champion()
        if champion is None:
            raise RuntimeError("no champion registered; run tracking.register_champion first")
        version = int(champion.version)
        fit_end = self.registry.get_version_tag(version, "fit_end")
        if not fit_end:
            raise RuntimeError(f"champion version {version} has no 'fit_end' tag; run tracking.tag_champion")
        return ChampionBundle(version, pd.Timestamp(fit_end), self.registry.load_champion())