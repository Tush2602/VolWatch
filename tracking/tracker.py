import os 
from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from mlflow.entities import Param, Metric
import time
import mlflow 
import mlflow.pyfunc

load_dotenv()

MODEL_PIP_REQUIREMENTS = ["xgboost", "pandas", "numpy"]

class ExperimentTracker:
    """The only place that talks to MLflow. Metadata goes to database, files to local folder."""
    def __init__(self, experiment_name : str = "volwatch-forecasting", tracking_uri : str | None = None, artifact_dir :str = "mlartifacts") -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if not self.tracking_uri: 
            raise RuntimeError("MLflow tracking uri is not set. check your .env file.")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_dir).resolve()
        self.client = MlflowClient(tracking_uri= self.tracking_uri)
        self.experiment_id = self._get_or_create_experiment()

    def _get_or_create_experiment(self) -> str:
        existing = self.client.get_experiment_by_name(self.experiment_name)
        if existing is not None:
            return existing.experiment_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        return self.client.create_experiment(self.experiment_name, artifact_location=self.artifact_dir.as_uri())

    def log_run(self, run_name :str, params: dict, metrics : dict, tags:dict | None = None, 
                model= None, registered_model_name: str | None = None)-> str:
        """Record one finished training run .return its run id"""
        run = self.client.create_run(self.experiment_id, run_name=run_name, tags= tags or {})
        run_id = run.info.run_id
        now = int(time.time() * 1000)
        try:
            self.client.log_batch(
                run_id, 
                metrics= [Metric(key, float(value) , now, 0) for key, value in metrics.items()],
                params= [Param(key, str(value)) for key, value in params.items()],
            )
            if model is not None:
                with mlflow.start_run(run_id=run_id):
                    mlflow.pyfunc.log_model(
                        name= "model",
                        python_model = model,
                        pip_requirements = MODEL_PIP_REQUIREMENTS,
                        registered_model_name = registered_model_name
                    )

        except Exception as e:
            self.client.set_terminated(run_id,status="FAILED")
            raise
        self.client.set_terminated(run_id, status = "FINISHED")
        return run_id