"""Fetch stored video clips from AWS S3."""

from __future__ import annotations

import asyncio
import logging

import boto3

logger = logging.getLogger(__name__)


def _parse_ref(ref: str, default_bucket: str) -> tuple[str, str]:
    """Accept either ``s3://bucket/key`` or a bare ``key``."""
    if ref.startswith("s3://"):
        without_scheme = ref[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        return bucket, key
    return default_bucket, ref


class S3MediaStore:
    def __init__(self, bucket: str, region: str) -> None:
        self._s3 = boto3.client("s3", region_name=region)
        self.bucket = bucket

    async def fetch_clip(self, ref: str) -> bytes | None:
        bucket, key = _parse_ref(ref, self.bucket)
        if not key:
            return None

        def _get() -> bytes:
            obj = self._s3.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()

        try:
            return await asyncio.to_thread(_get)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch clip %s: %s", ref, exc)
            return None
