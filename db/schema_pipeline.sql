CREATE TABLE IF NOT EXISTS forecasts (
    ticker         TEXT             NOT NULL,
    as_of_date     DATE             NOT NULL,
    model_version  INTEGER          NOT NULL,
    forecast_vol   DOUBLE PRECISION NOT NULL,
    created_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS monitoring_decisions (
    as_of_date  DATE        PRIMARY KEY,
    retrain     BOOLEAN     NOT NULL,
    details     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id       BIGSERIAL   PRIMARY KEY,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    status       TEXT        NOT NULL,
    details      TEXT
);