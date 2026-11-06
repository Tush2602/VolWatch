import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from api.schemas import ForecastOut, HistoryPoint

logger = logging.getLogger("volwatch.api")
DEFAULT_TICKER = "^NSEI"


def create_app(service) -> FastAPI:
    app = FastAPI(
        title="VolWatch API",
        version="1.0.0",
        description="Next-session NIFTY 50 volatility forecasts from a self-monitoring pipeline. "
                    "A technical demonstration, not investment advice.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}   # no database call: cheap to hit, and it wakes a sleeping service

    @app.get("/predict", response_model=ForecastOut)
    def predict(ticker: str = Query(DEFAULT_TICKER)):
        result = service.latest(ticker)
        if result is None:
            raise HTTPException(status_code=404, detail="no forecast available for this ticker")
        return result

    @app.get("/history", response_model=list[HistoryPoint])
    def history(ticker: str = Query(DEFAULT_TICKER), limit: int = Query(30, ge=1, le=250)):
        return service.history(ticker, limit)

    @app.get("/status")
    def status() -> dict:
        return service.status()

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.error("unhandled error: %s", type(exc).__name__)   # the type only: messages can contain URLs
        return JSONResponse(status_code=503, content={"detail": "service temporarily unavailable"},
                            headers={"Retry-After": "30"})

    
    return app