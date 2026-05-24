import sys
import json
from pathlib import Path

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nvision.cache.bridge import CacheBridge
from nvision.sim.combinations import CombinationGrid

def main():
    cache_dir = Path("artifacts/cache")
    bridge = CacheBridge(cache_dir)
    try:
        grid = CombinationGrid()
        # Find GaussianMixture combination
        combos = list(grid.iter(filter_strategy="GaussianMixture", filter_generator="NVCenter-lorentzian"))
        if not combos:
            print("No combos found")
            return
        
        combo = combos[0]
        cache = bridge.get_cache_for_category("NVCenter")
        
        # Get cached results using correct parameters for the new run
        results = cache.get_cached_combination(
            generator=combo.generator_name,
            noise=combo.noise_name,
            strategy=combo.strategy_name,
            repeats=1,
            seed=123,
            max_steps=100,
            timeout_s=1500,
        )
        
        if not results:
            print("No cached results found")
            return
            
        print(f"Loaded cached results: {len(results)} repeats")
        for i, (entries, main_result_row) in enumerate(results):
            print(f"\nRepeat {i+1}:")
            print(f"  Entries count: {len(entries)}")
            for j, entry in enumerate(entries):
                print(f"    Entry {j+1}: type={entry.get('type')}, path={entry.get('path')}")
                
    finally:
        bridge.close()

if __name__ == "__main__":
    main()
