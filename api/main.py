from api.app import create_app
from api.service import ForecastService
from db.client import Database

app = create_app(ForecastService(Database()))