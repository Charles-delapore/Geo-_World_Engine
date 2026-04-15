from __future__ import annotations

from functools import cached_property

import boto3
from botocore.client import Config

from app.config import settings


class S3Client:
    @cached_property
    def client(self):
        endpoint = settings.MINIO_ENDPOINT
        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            protocol = "https" if settings.MINIO_USE_SSL else "http"
            endpoint = f"{protocol}://{endpoint}"
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            use_ssl=settings.MINIO_USE_SSL,
        )

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=settings.MINIO_BUCKET)
        except Exception:
            self.client.create_bucket(Bucket=settings.MINIO_BUCKET)

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.ensure_bucket()
        self.client.put_object(Bucket=settings.MINIO_BUCKET, Key=key, Body=data, ContentType=content_type)

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=settings.MINIO_BUCKET, Key=key)
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=settings.MINIO_BUCKET, Key=key)
            return True
        except Exception:
            return False


s3_client = S3Client()
