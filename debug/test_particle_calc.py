# Debug script to check particle count
import sys
from pathlib import Path

# Add src to pythonpath if needed, but uv run will handle it
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief

def main():
    # Typically, the target uncertainties are defined somewhere in the Bayesian SBED locator
    # Let's mock what Bayesian SBED would use for convergence targets.
    # We can look up NVCenter defaults or use some representative values.
    # For NVCenter, the domain width for frequency is ~100 MHz, target is usually ~1e5 Hz (0.1 MHz).
    target_uncertainties = {
        "frequency": 2e5,      # 200 kHz
        "linewidth": 5e5,      # 500 kHz
        "dip_depth": 0.05,
        "k_np": 0.05,
        "split": 1e5,
    }
    
    print("Testing nv_center_smc_belief with dynamic target uncertainties:")
    print(f"Target uncertainties: {target_uncertainties}")
    
    belief = nv_center_smc_belief(target_uncertainties=target_uncertainties)
    
    print(f"Calculated num_particles: {belief.num_particles}")
    print(f"Original default was: 1000")

if __name__ == "__main__":
    main()
