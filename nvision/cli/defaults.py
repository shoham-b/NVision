"""Shared CLI default constants."""

from nvision.sim import presets as sim_presets

DEFAULT_REPEATS: int = 5
DEFAULT_RUNNERS: int = 4
DEFAULT_RUNNERS_ALL: int = 8
DEFAULT_LOC_MAX_STEPS: int = sim_presets.DEFAULT_LOC_MAX_STEPS
DEFAULT_LOC_TIMEOUT_S: int = 1500
DEFAULT_RUN_ALL: bool = False

DEMO_REPEATS: int = 3
DEMO_LOC_MAX_STEPS: int = 60
DEMO_RUNNERS: int = 8
DEMO_LOC_TIMEOUT_S: int = 300
