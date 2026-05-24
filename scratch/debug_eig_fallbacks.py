"""Debug script to inspect SBED EIG calculations and why f_lo (2.6e9) is frequently selected.

This script runs a short NV center simulation using SBED, monitors proposed
acquisitions, and halts/inspects detailed EIG variables whenever f_lo (2.6e9)
is proposed as the next measurement point.
"""
from __future__ import annotations

import random
import numpy as np
from pathlib import Path

from nvision import CoreExperiment, NVCenterCoreGenerator
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator as SbedLocator

def main() -> None:
    # Setup standard experiment
    rng = random.Random(42)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    true_signal = gen.generate(rng)
    
    x_min, x_max = None, None
    for name in true_signal.parameter_names:
        if "frequency" in name:
            x_min, x_max = true_signal.get_param_bounds(name)
            break
    assert x_min is not None
    
    exp = CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)
    
    # Create SBED locator
    loc = SbedLocator.create(
        builder=nv_center_smc_belief,
        max_steps=150,
        n_candidates=3125,
        num_particles=5000,
        parameter_bounds=None,
        initial_sweep_steps=4,
        noise_std=0.02,
    )
    
    # Intercept select_max_information_gain to log state when it chooses the boundary
    original_select = loc.belief.select_max_information_gain
    
    def debug_select_max_information_gain(candidates: np.ndarray, n: int, noise_std: float = 0.02) -> np.ndarray:
        eig_scores = loc.belief.expected_information_gain(candidates, noise_std=noise_std)
        best_indices = original_select(candidates, n, noise_std=noise_std)
        chosen_val = best_indices[0] if len(best_indices) > 0 else None
        
        # If the boundary (2.6e9) is chosen, print out detailed diagnostic info
        if chosen_val is not None and abs(chosen_val - 2.6e9) < 1.0:
            print("\n!!! BOUNDARY ACQUISITION DETECTED !!!")
            print(f"Chosen value: {chosen_val}")
            print(f"Total candidates: {len(candidates)}")
            print(f"Candidates min: {np.min(candidates)}, max: {np.max(candidates)}")
            
            # Show top EIG scores and their indices/values
            top_indices = np.argsort(eig_scores)[::-1]
            print("\nTop 10 EIG scores overall:")
            for i in range(min(10, len(top_indices))):
                idx = top_indices[i]
                print(f"  Rank {i+1}: Index {idx}, Candidate {candidates[idx]:.4f}, EIG {eig_scores[idx]:.2e}")
                
            # Check winner indices and scores from _chunk_argmax
            from nvision.belief.smc_marginal import _chunk_argmax, _EIG_CHUNK_SIZE
            winner_indices = _chunk_argmax(eig_scores.astype(np.float32), _EIG_CHUNK_SIZE)
            winner_scores = eig_scores[winner_indices]
            
            print(f"\nChunk Winner statistics (total {len(winner_indices)} chunks of size {_EIG_CHUNK_SIZE}):")
            print(f"  Winner scores min: {np.min(winner_scores):.2e}, max: {np.max(winner_scores):.2e}")
            
            # Show first 10 chunk winners
            print("\nFirst 10 chunk winners:")
            for c in range(min(10, len(winner_indices))):
                idx = winner_indices[c]
                print(f"  Chunk {c}: Winner Index {idx}, Candidate {candidates[idx]:.4f}, EIG {eig_scores[idx]:.2e}")
                
            # Look at Boltzmann sampling math
            temp = 0.01
            shifted_scores = (winner_scores - np.max(winner_scores)) / temp
            probs = np.exp(shifted_scores)
            probs /= np.sum(probs)
            
            top_probs_idx = np.argsort(probs)[::-1]
            print("\nTop 5 chunk winners by Boltzmann probability:")
            for i in range(min(5, len(top_probs_idx))):
                c_idx = top_probs_idx[i]
                g_idx = winner_indices[c_idx]
                print(f"  Rank {i+1}: Chunk {c_idx} (global idx {g_idx}), Candidate {candidates[g_idx]:.4f}, Prob {probs[c_idx]:.4f}, EIG {winner_scores[c_idx]:.2e}")
            
        return best_indices
        
    loc.belief.select_max_information_gain = debug_select_max_information_gain
    
    # Run the experiment loop
    print("Starting simulation loop...")
    for step in range(1, 100):
        x = loc.next()
        obs = exp.measure(x, rng)
        loc.observe(obs)
        if step % 10 == 0:
            print(f"Step {step} done. Proposed: {x:.4f}")

if __name__ == "__main__":
    main()
