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

def test_artifact_root_from_the_environment_is_used_for_new_experiments(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", "s3://example-bucket/mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = ExperimentTracker("env-exp", tracking_uri=uri)
    location = tracker.client.get_experiment(tracker.experiment_id).artifact_location
    assert location.startswith("s3://example-bucket/mlflow")


def test_explicit_artifact_dir_beats_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", "s3://example-bucket/mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = ExperimentTracker("local-exp", tracking_uri=uri, artifact_dir=str(tmp_path / "art"))
    assert tracker.client.get_experiment(tracker.experiment_id).artifact_location.startswith("file:")


def test_experiment_name_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "named-by-env")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = ExperimentTracker(tracking_uri=uri, artifact_dir=str(tmp_path / "art"))
    assert tracker.experiment_name == "named-by-env"
    