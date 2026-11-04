import numpy as np
import pandas as pd
import pytest

from models.xgb_model import XGBForecaster
from pipeline.champion import ChampionBundle
from pipeline.retrainer import Retrainer
from pipeline.trial import ChallengerTrial
from tracking.registry import ModelRegistry
from tracking.tracker import ExperimentTracker


def make_labeled(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    vol = rng.uniform(5, 20, n)
    df = pd.DataFrame({"vol": vol, "vix": vol * 1.5 + rng.normal(0, 1, n)}, index=idx)
    df["target"] = vol * 1.1
    return df


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, df):
        return np.full(len(df), self.value)


class PerfectModel:
    def predict(self, df):
        return df["target"].to_numpy()


def make_trial():
    return ChallengerTrial(lambda: XGBForecaster(n_estimators=10, max_depth=2),
                           holdout_days=40, min_train_rows=100)


def test_challenger_wins_against_a_terrible_champion():
    result = make_trial().run(make_labeled(), ConstantModel(100.0))
    assert result.wins is True
    assert result.challenger_qlike < result.champion_qlike


def test_challenger_cannot_beat_a_perfect_champion():
    assert make_trial().run(make_labeled(), PerfectModel()).wins is False


def test_trial_leaves_a_purge_gap_before_the_holdout():
    labeled = make_labeled()
    result = make_trial().run(labeled, ConstantModel(100.0))
    gap = labeled.index.get_loc(result.holdout_start) - labeled.index.get_loc(result.fit_end)
    assert gap >= 2


def test_trial_needs_enough_rows():
    with pytest.raises(ValueError):
        make_trial().run(make_labeled(n=100), ConstantModel(100.0))


def make_setup(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = ExperimentTracker("test-exp", tracking_uri=uri, artifact_dir=str(tmp_path / "art"))
    registry = ModelRegistry(tracker, "test-model")
    return tracker, registry, Retrainer(tracker, registry, make_trial())


def test_winning_challenger_is_registered_tagged_and_promoted(tmp_path):
    labeled = make_labeled()
    tracker, registry, retrainer = make_setup(tmp_path)
    champion = ChampionBundle(version=1, fit_end=labeled.index[100], model=ConstantModel(100.0))

    outcome = retrainer.run(labeled, champion, data_end="2022-01-01")

    assert outcome["promoted"] is True
    assert int(registry.get_champion().version) == outcome["new_version"]
    assert registry.get_version_tag(outcome["new_version"], "fit_end") is not None


def test_losing_challenger_is_logged_but_not_registered(tmp_path):
    labeled = make_labeled()
    tracker, registry, retrainer = make_setup(tmp_path)
    champion = ChampionBundle(version=1, fit_end=labeled.index[100], model=PerfectModel())

    outcome = retrainer.run(labeled, champion, data_end="2022-01-01")

    assert outcome["promoted"] is False
    assert registry.get_champion() is None
    assert tracker.client.search_model_versions("name='test-model'") == []