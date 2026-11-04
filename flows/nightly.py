import sys

from prefect import flow, task

from pipeline.factory import build_pipeline


@task(name="ingest", retries=3, retry_delay_seconds=[60, 300, 900])
def ingest_task() -> None:
    build_pipeline().ingest_step()


@task(name="analyze")   # no automatic retries: a failed retrain should be looked at, not repeated blindly
def analyze_task(force_retrain: bool) -> dict:
    return build_pipeline().analyze_step(force_retrain)


@flow(name="volwatch-nightly", log_prints=True)
def nightly_flow(force_retrain: bool = False) -> dict:
    pipeline = build_pipeline()
    run_id = pipeline.start()
    try:
        ingest_task()
        result = analyze_task(force_retrain)
    except Exception as error:
        pipeline.finish_failed(run_id, error)
        raise
    pipeline.finish_ok(run_id, result)
    return result


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # 12:30 UTC = 18:00 IST, Monday to Friday. This needs this process to stay running.
        nightly_flow.serve(name="volwatch-nightly-local", cron="30 12 * * 1-5")
    else:
        nightly_flow(force_retrain="--force-retrain" in sys.argv)