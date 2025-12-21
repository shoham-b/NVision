from nvision.sim.locs.nv_center.sequential_bayesian_locator import NVCenterSequentialBayesianLocator
import inspect
import polars as pl

print("Signature:", inspect.signature(NVCenterSequentialBayesianLocator.propose_next))

try:
    loc = NVCenterSequentialBayesianLocator()
    # Mock set_context if needed, but propose_next signature is what matters for TypeError at call site
    print("Instance created")
except Exception as e:
    print(f"Init failed: {e}")
