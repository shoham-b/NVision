"""Download a shared Google Drive folder with rclone (read-only), for the lab's Bayesian ESR runs.

Needs ``rclone`` on PATH and a Drive remote (default ``gdrive``, scope ``drive.readonly``; set up or
refresh it with ``rclone config`` / ``rclone config reconnect gdrive:``). Nothing is uploaded or changed on Drive.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_FOLDER_URL_RE = re.compile(r"/folders/([A-Za-z0-9_-]+)")
_BARE_ID_RE = re.compile(r"[A-Za-z0-9_-]{20,}")
# Figures are large and not needed to recompute the statistics.
_EXCLUDES = ("*.png", "*.fig")


def parse_drive_folder_id(url_or_id: str) -> str:
    """Folder id from a ``drive.google.com/.../folders/<id>`` link, or the id itself."""
    text = url_or_id.strip()
    if m := _FOLDER_URL_RE.search(text):
        return m.group(1)
    if _BARE_ID_RE.fullmatch(text):
        return text
    raise ValueError(f"Not a Google Drive folder link or id: {url_or_id!r}")


def fetch_drive_folder(url_or_id: str, dest: Path, remote: str = "gdrive") -> Path:
    """``rclone copy`` the Drive folder into *dest* (created if needed); return *dest*.

    Raises ``RuntimeError`` when rclone is missing or fails (e.g. the remote's login expired).
    """
    folder_id = parse_drive_folder_id(url_or_id)
    rclone = shutil.which("rclone")
    if rclone is None:
        raise RuntimeError("rclone is not on PATH; install it and configure a Drive remote (rclone config)")
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [rclone, "copy", f"{remote}:", str(dest), "--drive-root-folder-id", folder_id, "--stats-one-line", "-v"]
    for pattern in _EXCLUDES:
        cmd += ["--exclude", pattern]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"rclone exited with {result.returncode}; if the login expired run: rclone config reconnect {remote}:"
        )
    return dest
