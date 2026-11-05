import time

import requests


class ApiUnavailable(RuntimeError):
    pass


class ApiClient:
    """Calls the VolWatch API, patiently: the free tier takes 30 to 60 seconds to wake from sleep."""

    def __init__(self, base_url: str, session=None, attempts: int = 4, wait_seconds: float = 15,
                 timeout: float = 70, sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.attempts = attempts
        self.wait_seconds = wait_seconds
        self.timeout = timeout
        self.sleep = sleep

    def _get(self, path: str, params: dict | None = None):
        last = "no response"
        for attempt in range(1, self.attempts + 1):
            try:
                response = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 404:
                    return None   # a clean "nothing there", not a failure
                last = f"HTTP {response.status_code}"
            except requests.RequestException as error:
                last = type(error).__name__
            if attempt < self.attempts:
                self.sleep(self.wait_seconds)
        raise ApiUnavailable(f"the API did not respond ({last})")

    def predict(self):
        return self._get("/predict")

    def history(self, limit: int = 30):
        return self._get("/history", {"limit": limit})

    def status(self):
        return self._get("/status")