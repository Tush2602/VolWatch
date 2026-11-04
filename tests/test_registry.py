import numpy as np
import pandas as pd
import pytest

from models.xgb_model import XGBForecaster
from tracking.pyfunc_wrapper import ForecasterModel
from tracking.registry import ModelRegistry
from tracking.tracker import ExperimentTracker

NAME = "test-model"


def make_table(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    vol = rng.uniform(5, 20, n)
    df = pd.DataFrame({"vol": vol, "vix": vol * 1.5 + rng.normal(0, 1, n)}, index=idx)
    df["target"] = vol * 1.1
    return df


def make_setup(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = ExperimentTracker("test-exp", tracking_uri=uri, artifact_dir=str(tmp_path / "art"))
    return tracker, ModelRegistry(tracker, NAME)


def register(tracker, registry, table, qlike):
    model = XGBForecaster(n_estimators=10, max_depth=2).fit(table)
    run_id = tracker.log_run("r", {}, {"test_qlike": qlike},
                             model=ForecasterModel(model), registered_model_name=NAME)
    return registry.version_for_run(run_id), model


def test_first_model_becomes_champion(tmp_path):
    tracker, registry = make_setup(tmp_path)
    version, _ = register(tracker, registry, make_table(), 0.40)
    assert registry.promote_if_better(version) is True
    assert int(registry.get_champion().version) == version


def test_worse_challenger_does_not_replace_champion(tmp_path):
    tracker, registry = make_setup(tmp_path)
    table = make_table()
    v1, _ = register(tracker, registry, table, 0.40)
    registry.promote_if_better(v1)
    v2, _ = register(tracker, registry, table, 0.50)
    assert registry.promote_if_better(v2) is False
    assert int(registry.get_champion().version) == v1


def test_better_challenger_replaces_champion(tmp_path):
    tracker, registry = make_setup(tmp_path)
    table = make_table()
    v1, _ = register(tracker, registry, table, 0.40)
    registry.promote_if_better(v1)
    v2, _ = register(tracker, registry, table, 0.30)
    assert registry.promote_if_better(v2) is True
    assert int(registry.get_champion().version) == v2


def test_loaded_champion_predicts_like_the_original(tmp_path):
    tracker, registry = make_setup(tmp_path)
    table = make_table()
    version, model = register(tracker, registry, table, 0.40)
    registry.promote_if_better(version)

    loaded = registry.load_champion()
    expected = model.predict(table).to_numpy()
    np.testing.assert_allclose(np.asarray(loaded.predict(table)).ravel(), expected, rtol=1e-5)