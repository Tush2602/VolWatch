import pandas as pd

from db.client import Database


def main() -> None:
    db = Database()
    db.run_script("db/schema_pipeline.sql")

    run_id = db.start_run()
    db.finish_run(run_id, "TEST", {"note": "connection check"})
    db.save_decision(pd.Timestamp("1900-01-01"), False, {"note": "check"})
    db.save_forecast("TEST", pd.Timestamp("1900-01-01"), 0, 1.0)
    print("latest run:", db.latest_run())
    print("forecasts:", db.read_forecasts("TEST").to_dict("records"))

    with db._connect() as conn:
        conn.execute("DELETE FROM forecasts WHERE ticker = 'TEST'")
        conn.execute("DELETE FROM monitoring_decisions WHERE as_of_date = '1900-01-01'")
        conn.execute("DELETE FROM pipeline_runs WHERE status = 'TEST'")
    print("pipeline tables work")


if __name__ == "__main__":
    main()