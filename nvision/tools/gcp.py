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

    Returns a success message or raises RuntimeError with a clear diagnostic.
    """
    creds_path = _credentials_file()
    if not creds_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set.\n"
            "  Set it to the path of your service-account JSON key file.\n"
            "  Example: GOOGLE_APPLICATION_CREDENTIALS=C:/Users/me/keys/nvision-key.json"
        )

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
            "GCP default credentials could not be determined.\n"
            "  - Ensure the service account key is valid\n"
            "  - Verify GOOGLE_APPLICATION_CREDENTIALS points to the correct JSON file"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"GCP credentials test failed: {exc}") from exc

    return f"Credentials OK ({creds_path})"


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
            f"Access denied to gs://{bucket_name}.\n"
            "  - Check IAM permissions on the service account."
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


def get_public_url(bucket_name: str, directory_name: str) -> str:
    """Get the base public URL for a directory in a GCP bucket."""
    return f"https://storage.googleapis.com/{bucket_name}/{directory_name}/index.html"
