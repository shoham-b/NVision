"""Shared CLI default constants."""

import os

from dotenv import load_dotenv

from nvision.sim import presets as sim_presets

# Load environment variables from .env here so it's guaranteed to load
# before any of these constants are evaluated, regardless of where
# defaults.py is imported from.
load_dotenv()

DEFAULT_REPEATS: int = int(os.getenv("NVISION_DEFAULT_REPEATS", "5"))
DEFAULT_RUNNERS: int = int(os.getenv("NVISION_DEFAULT_RUNNERS", "4"))
DEFAULT_LOC_MAX_STEPS: int = int(os.getenv("NVISION_DEFAULT_LOC_MAX_STEPS", str(sim_presets.DEFAULT_LOC_MAX_STEPS)))
DEFAULT_LOC_TIMEOUT_S: int = int(os.getenv("NVISION_DEFAULT_LOC_TIMEOUT_S", "1500"))
DEFAULT_RUN_ALL: bool = os.getenv("NVISION_DEFAULT_RUN_ALL", "False").lower() in ("true", "1", "yes")

DEMO_REPEATS: int = int(os.getenv("NVISION_DEMO_REPEATS", "3"))
DEMO_LOC_MAX_STEPS: int = int(os.getenv("NVISION_DEMO_LOC_MAX_STEPS", "60"))
DEMO_LOC_TIMEOUT_S: int = int(os.getenv("NVISION_DEMO_LOC_TIMEOUT_S", "300"))
