import pytest

from tracking.tracker import ExperimentTracker


def make_tracker(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    return ExperimentTracker("test-exp", tracking_uri=uri, artifact_dir=str(tmp_path / "art"))


def test_logged_run_can_be_read_back(tmp_path):
    tracker = make_tracker(tmp_path)
    run_id = tracker.log_run("r1", {"max_depth": 2}, {"qlike": 0.4}, tags={"stage": "validation"})

    run = tracker.client.get_run(run_id)
    assert run.data.params["max_depth"] == "2"
    assert run.data.metrics["qlike"] == pytest.approx(0.4)
    assert run.data.tags["stage"] == "validation"
    assert run.info.status == "FINISHED"


def test_experiment_is_reused_not_duplicated(tmp_path):
    first = make_tracker(tmp_path)
    second = make_tracker(tmp_path)
    assert first.experiment_id == second.experiment_id


def test_missing_uri_fails_loudly(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError):
        ExperimentTracker(tracking_uri=None)