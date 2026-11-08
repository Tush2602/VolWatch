import os
import tempfile
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )


def raw_check(bucket: str) -> None:
    client = s3_client()
    key = f"smoke/{uuid.uuid4().hex}.txt"
    client.put_object(Bucket=bucket, Key=key, Body=b"hello volwatch")
    print("raw put ok")
    listed = [o["Key"] for o in client.list_objects_v2(Bucket=bucket, Prefix="smoke/").get("Contents", [])]
    print("raw list ok:", key in listed)
    print("raw get ok:", client.get_object(Bucket=bucket, Key=key)["Body"].read() == b"hello volwatch")
    client.delete_object(Bucket=bucket, Key=key)
    print("raw delete ok")


def mlflow_check(bucket: str) -> None:
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.environ["AWS_ENDPOINT_URL_S3"])
    prefix = f"smoke-mlflow/{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = MlflowClient(tracking_uri=f"sqlite:///{(Path(tmp) / 'mlflow.db').as_posix()}")
        experiment = client.create_experiment("storage-check", artifact_location=f"s3://{bucket}/{prefix}")
        run = client.create_run(experiment)
        client.log_text(run.info.run_id, "hello volwatch", "check/hello.txt")
        print("mlflow upload ok")
        local = client.download_artifacts(run.info.run_id, "check/hello.txt", tmp)
        print("mlflow download ok:", Path(local).read_text() == "hello volwatch")

    s3 = s3_client()
    for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
        s3.delete_object(Bucket=bucket, Key=obj["Key"])
    print("mlflow cleanup ok")


if __name__ == "__main__":
    bucket_name = os.environ["S3_BUCKET"]
    raw_check(bucket_name)
    mlflow_check(bucket_name)