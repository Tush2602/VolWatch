from db.client import Database
from tracking.register_champion import MODEL_NAME
from tracking.registry import ModelRegistry
from tracking.tracker import ExperimentTracker


def main() -> None:
    db = Database()
    run = db.latest_run()
    details = (run or {}).get("details") or {}
    print("latest run:", run["status"] if run else None, "| finished:", run["finished_at"] if run else None)
    print("as_of:", details.get("as_of"), "| forced:", details.get("forced"))
    print("retrain outcome:", details.get("retrain_outcome"))

    registry = ModelRegistry(ExperimentTracker(), MODEL_NAME)
    champion = registry.get_champion()
    print("champion version:", champion.version if champion else None)
    for version in registry.client.search_model_versions(f"name='{MODEL_NAME}'"):
        print(f"v{version.version} aliases={version.aliases}")

    print("latest forecasts:")
    print(db.read_forecasts("^NSEI", limit=5).to_string(index=False))


if __name__ == "__main__":
    main()