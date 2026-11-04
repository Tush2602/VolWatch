import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from tracking.tracker import ExperimentTracker

class ModelRegistry:
    """Decides which registered model version is the champion, i.e, the one served in production."""
    CHAMPION = "champion"
    def __init__(self, tracker: ExperimentTracker, model_name: str = "volwatch-xgb"):
        self.client = tracker.client
        self.model_name =model_name

    def version_for_run(self, run_id :str) -> int:
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        matches = [v for v in versions if v.run_id == run_id]
        if not matches:
            raise LookupError(f"No registered version of {self.model_name} came from run {run_id}")
        return int(matches[0].version)
    def set_champion(self, version : int) -> None:
        self.client.set_registered_model_alias(self.model_name, self.CHAMPION ,str(version))
    
    def get_champion(self):
        try:
            return self.client.get_model_version_by_alias(self.model_name, self.CHAMPION)
        except MlflowException:
            return None

    def promote_if_better(self, version : int, metric : str = "test_qlike", lower_is_better : bool = True)-> bool:
        """Make `version` champion if it beats the current one on `metric`. Returns True if promoted."""
        champion = self.get_champion()
        if champion is None:
            self.set_champion(version)
            return True

        candidate = self._metric_of(version, metric)
        current = self._metric_of(int(champion.version), metric)
        better = candidate < current if lower_is_better else candidate > current
        if better:
            self.set_champion(version)
        return better
    
    def load_champion(self):
        return mlflow.pyfunc.load_model(f"models:/{self.model_name}@{self.CHAMPION}")

    def _metric_of(self, version: int, metric : str)-> float:
        mv = self.client.get_model_version(self.model_name, str(version))
        value = self.client.get_run(mv.run_id).data.metrics.get(metric)
        if value is None: 
            raise KeyError(f"Version {version} has no metric of '{metric}'")
        return value 

    def set_version_tag(self, version: int, key: str, value: str) -> None:
        self.client.set_model_version_tag(self.model_name, str(version), key, str(value))

    def get_version_tag(self, version: int, key: str):
        return self.client.get_model_version(self.model_name, str(version)).tags.get(key)