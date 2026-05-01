from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("nvision")


def upload_artifacts(directory: Path, bucket_name: str) -> None:
    """Upload a directory of artifacts to a GCP bucket."""
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
