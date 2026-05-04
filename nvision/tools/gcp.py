from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("nvision")


def _credentials_file() -> str | None:
    """Return the path to the GCP credentials file if set."""
    return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")


def verify_credentials() -> str:
    """Verify GCP credentials are available and valid.

    Supports either a service-account key (GOOGLE_APPLICATION_CREDENTIALS)
    or Application Default Credentials from ``gcloud auth application-default login``.

    Returns a success message or raises RuntimeError with a clear diagnostic.
    """
    creds_path = _credentials_file()
    if creds_path:
        creds_file = Path(creds_path)
        if not creds_file.exists():
            raise RuntimeError(
                f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file:\n  {creds_path}\n"
                "Check the path and try again."
            )

    try:
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import storage

        client = storage.Client()
        # Light-weight validation: list buckets to confirm auth works
        list(client.list_buckets(max_results=1))
    except DefaultCredentialsError as exc:
        raise RuntimeError(
            "GCP credentials could not be determined.\n"
            "  Options:\n"
            "  1. Set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key file.\n"
            "  2. Run: gcloud auth application-default login\n"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"GCP credentials test failed: {exc}") from exc

    if creds_path:
        return f"Credentials OK (service-account: {creds_path})"
    return "Credentials OK (Application Default Credentials)"


def verify_bucket(bucket_name: str) -> str:
    """Verify the bucket exists and is accessible.

    Returns a success message or raises RuntimeError.
    """
    from google.api_core.exceptions import Forbidden, NotFound
    from google.cloud import storage

    client = storage.Client()
    try:
        client.get_bucket(bucket_name)
        return f"Bucket OK: gs://{bucket_name}"
    except NotFound as _exc:
        raise RuntimeError(
            f"Bucket gs://{bucket_name} not found.\n"
            "  - Verify the bucket name is correct, or\n"
            "  - Create it in the GCP Console."
        ) from None
    except Forbidden as _exc:
        raise RuntimeError(
            f"Access denied to gs://{bucket_name}.\n  - Check IAM permissions on the service account."
        ) from None


def upload_artifacts(directory: Path, bucket_name: str) -> None:
    """Upload a directory of artifacts to a GCP bucket."""
    # Fail fast before touching the network
    verify_credentials()
    verify_bucket(bucket_name)

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    log.info(f"Uploading artifacts from {directory} to gs://{bucket_name}/{directory.name}...")

    # Upload files recursively
    count = 0
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            # Create a blob with a path relative to the directory's parent
            # So if directory is 'artifacts', blob path is 'artifacts/...'
            blob_path = f"{directory.name}/{file_path.relative_to(directory).as_posix()}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(file_path))
            count += 1

    log.info(f"Successfully uploaded {count} files to GCP bucket {bucket_name}.")


def download_artifacts(directory: Path, bucket_name: str) -> None:
    """Download artifacts from a GCP bucket to a local directory."""
    verify_credentials()
    verify_bucket(bucket_name)

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    prefix = f"{directory.name}/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    log.info("Downloading %s files from gs://%s/%s to %s...", len(blobs), bucket_name, prefix, directory)

    count = 0
    for blob in blobs:
        relative_path = blob.name[len(prefix) :]
        local_path = directory / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        count += 1

    log.info("Downloaded %s files to %s.", count, directory)


def get_public_url(bucket_name: str, directory_name: str) -> str:
    """Get the base public URL for a directory in a GCP bucket."""
    return f"https://storage.googleapis.com/{bucket_name}/{directory_name}/index.html"
