"""Shared CLI default constants."""

import os

from dotenv import load_dotenv

from nvision.sim import presets as sim_presets

# Load environment variables from .env here so it's guaranteed to load
# before any of these constants are evaluated, regardless of where
# defaults.py is imported from.
load_dotenv()

# Core Execution Config
DEFAULT_REPEATS: int = int(os.getenv("NVISION_DEFAULT_REPEATS", "5"))
DEFAULT_RUNNERS: int = int(os.getenv("NVISION_DEFAULT_RUNNERS", "4"))
DEFAULT_LOC_MAX_STEPS: int = int(os.getenv("NVISION_DEFAULT_LOC_MAX_STEPS", str(sim_presets.DEFAULT_LOC_MAX_STEPS)))
DEFAULT_LOC_TIMEOUT_S: int = int(os.getenv("NVISION_DEFAULT_LOC_TIMEOUT_S", "1500"))
DEFAULT_RUN_ALL: bool = os.getenv("NVISION_DEFAULT_RUN_ALL", "False").lower() in ("true", "1", "yes")

# UI & Browser Flags
DEFAULT_OPEN_BROWSER: bool = os.getenv("NVISION_DEFAULT_OPEN_BROWSER", "False").lower() in ("true", "1", "yes")

# Output & Logs Config
DEFAULT_OUT: str | None = os.getenv("NVISION_DEFAULT_OUT", None)
DEFAULT_LOGS_ROOT: str | None = os.getenv("NVISION_DEFAULT_LOGS_ROOT", None)
DEFAULT_LOG_LEVEL: str = os.getenv("NVISION_DEFAULT_LOG_LEVEL", "INFO")

# GCP Integration
DEFAULT_GCP: bool = os.getenv("NVISION_GCP", "False").lower() in ("true", "1", "yes")
DEFAULT_GCP_BUCKET: str | None = os.getenv("NVISION_GCP_BUCKET", None)

# Demo & Beta Specific
DEMO_REPEATS: int = int(os.getenv("NVISION_DEMO_REPEATS", "3"))
DEMO_LOC_MAX_STEPS: int = int(os.getenv("NVISION_DEMO_LOC_MAX_STEPS", "60"))
DEMO_LOC_TIMEOUT_S: int = int(os.getenv("NVISION_DEMO_LOC_TIMEOUT_S", "300"))
DEMO_OUT: str | None = os.getenv("NVISION_DEMO_OUT", None)
DEMO_LOGS_ROOT: str | None = os.getenv("NVISION_DEMO_LOGS_ROOT", None)
BETA_OUT: str | None = os.getenv("NVISION_BETA_OUT", None)

DEFAULT_GCP: bool = os.getenv("NVISION_GCP_ENABLED", "False").lower() in ("true", "1", "yes")
DEFAULT_GCP_BUCKET: str | None = os.getenv("NVISION_GCP_BUCKET")
