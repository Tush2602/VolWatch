import os 
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

def main() -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(f"MLFLOW_TRACKING_URI is not set.")
    subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri, "--port", "5000"], check = True)

if __name__ =="__main__":
    main()