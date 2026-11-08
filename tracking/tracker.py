import os
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

load_dotenv()

FINISHED = "FINISHED"
FAILED = "FAILED"
MODEL_PIP_REQUIREMENTS = ["xgboost", "pandas", "numpy"]


class ExperimentTracker:
    """The only place that talks to MLflow. Metadata goes to the database; model files go to local disk or S3."""

    def __init__(self, experiment_name: str | None = None, tracking_uri: str | None = None,
                 artifact_dir: str | None = None, artifact_root: str | None = None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if not self.tracking_uri:
            raise RuntimeError("MLFLOW_TRACKING_URI is not set (check your .env file)")
        mlflow.set_tracking_uri(self.tracking_uri)

        self.experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "volwatch-forecasting")
        self.artifact_location = self._resolve_artifact_location(artifact_dir, artifact_root)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self.experiment_id = self._get_or_create_experiment()

    @staticmethod
    def _resolve_artifact_location(artifact_dir, artifact_root) -> str:
        if artifact_dir is not None:   # an explicit local folder always wins, which keeps the tests offline
            path = Path(artifact_dir).resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path.as_uri()
        root = artifact_root or os.getenv("MLFLOW_ARTIFACT_ROOT")
        if root:
            endpoint = os.getenv("AWS_ENDPOINT_URL_S3")
            if root.startswith("s3://") and endpoint:
                os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", endpoint)
            return root
        path = Path("mlartifacts").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path.as_uri()

    def _get_or_create_experiment(self) -> str:
        existing = self.client.get_experiment_by_name(self.experiment_name)
        if existing is not None:
            return existing.experiment_id
        return self.client.create_experiment(self.experiment_name, artifact_location=self.artifact_location)

    def log_run(self, run_name: str, params: dict, metrics: dict, tags: dict | None = None,
                model=None, registered_model_name: str | None = None) -> str:
        """Record one finished training run, optionally saving and registering its model."""
        run = self.client.create_run(self.experiment_id, run_name=run_name, tags=tags or {})
        run_id = run.info.run_id
        now = int(time.time() * 1000)
        try:
            self.client.log_batch(
                run_id,
                metrics=[Metric(key, float(value), now, 0) for key, value in metrics.items()],
                params=[Param(key, str(value)) for key, value in params.items()],
            )
            if model is not None:
                with mlflow.start_run(run_id=run_id):
                    mlflow.pyfunc.log_model(
                        name="model",
                        python_model=model,
                        pip_requirements=MODEL_PIP_REQUIREMENTS,
                        registered_model_name=registered_model_name,
                    )
        except Exception:
            self.client.set_terminated(run_id, status=FAILED)
            raise
        self.client.set_terminated(run_id, status=FINISHED)
        return run_id