"""Shared CLI default constants."""

import os

from dotenv import load_dotenv

from nvision.sim import presets as sim_presets
from nvision.sim.defaults import NVISION_CONVERGENCE_THRESHOLD

# Load environment variables from .env here so it's guaranteed to load
# before any of these constants are evaluated, regardless of where
# defaults.py is imported from.
load_dotenv()

# Core Execution Config
DEFAULT_REPEATS: int = int(os.getenv("NVISION_DEFAULT_REPEATS", "5"))

def _default_runners() -> int:
    """Runner processes to use when the caller does not specify.

    Scales with the machine instead of a fixed 4: tasks are independent, so throughput
    tracks process count. Uses half the logical CPUs, which approximates the physical
    core count on hyperthreaded machines -- HT siblings share an FP unit, so the second
    sibling adds little for this FP-bound work while doubling memory footprint (each
    worker carries its own numpy/numba/polars import and particle arrays). Capped at 8
    to bound that footprint on many-core boxes; raise via NVISION_DEFAULT_RUNNERS.
    Paired with the per-worker thread cap in `cli/run.py::_thread_budget`, which hands
    each worker the leftover cores so the two never oversubscribe.
    """
    cores = os.cpu_count() or 1
    return max(1, min(8, cores // 2))


DEFAULT_RUNNERS: int = int(os.getenv("NVISION_DEFAULT_RUNNERS", str(_default_runners())))
MIN_RUNNERS: int = int(os.getenv("NVISION_MIN_RUNNERS", "1"))
DEFAULT_LOC_MAX_STEPS: int = int(os.getenv("NVISION_DEFAULT_LOC_MAX_STEPS", str(sim_presets.DEFAULT_LOC_MAX_STEPS)))
DEFAULT_LOC_TIMEOUT_S: int = int(os.getenv("NVISION_DEFAULT_LOC_TIMEOUT_S", "1500"))
DEFAULT_RUN_ALL: bool = os.getenv("NVISION_DEFAULT_RUN_ALL", "False").lower() in ("true", "1", "yes")
STREAMING_REPEAT_THRESHOLD: int = int(os.getenv("NVISION_STREAMING_REPEAT_THRESHOLD", "0"))

# UI & Browser Flags
DEFAULT_OPEN_BROWSER: bool = os.getenv("NVISION_DEFAULT_OPEN_BROWSER", "False").lower() in ("true", "1", "yes")

# Noise presets limits
DEFAULT_NOISE_MAX_GAUSS: float = float(os.getenv("NVISION_NOISE_MAX_GAUSS", "0.01"))
DEFAULT_NOISE_GAUSS_STEPS: int = int(os.getenv("NVISION_NOISE_GAUSS_STEPS", "5"))

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

# Locator convergence (relative fraction of parameter bound width; 0.01 = 1%).
# Per-parameter absolute overrides: NVISION_FREQ_CONVERGENCE_THRESHOLD,
# NVISION_K_NP_CONVERGENCE_THRESHOLD, etc. (see nvision.sim.defaults).
DEFAULT_CONVERGENCE_THRESHOLD: float = NVISION_CONVERGENCE_THRESHOLD

# SMC Belief parameters
MIN_STEPS_BEFORE_NARROWING: int = int(os.getenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "8"))
SMC_FOCUSING_COVER_FACTOR: float = float(os.getenv("NVISION_SMC_FOCUSING_COVER_FACTOR", "3.0"))
SMC_FOCUSING_TAIL_PERCENTILE: float = float(os.getenv("NVISION_SMC_FOCUSING_TAIL_PERCENTILE", "1.0"))
