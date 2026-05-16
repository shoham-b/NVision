from pathlib import Path
from nvision.cache import CacheBridge
from nvision.cli.render import _read_locator_combination_config

def dump_cache_configs():
    bridge = CacheBridge(Path("artifacts/cache"))
    stores = [("NVCenter", bridge.nv_center), ("Complementary", bridge.complementary)]
    
    print(f"{'Category':<15} {'Kind':<35} {'Gen':<25} {'Steps':<6} {'Repeats':<8}")
    print("-" * 100)
    
    for cat_name, store in stores:
        count = 0
        for key in store.backend:
            payload = store.backend.get(key)
            if isinstance(payload, dict) and "config" in payload:
                cfg = payload["config"]
                kind = cfg.get("kind", "-")
                gen = cfg.get("generator", "-")
                steps = cfg.get("max_steps", "-")
                repeats = cfg.get("repeats", "-")
                print(f"{cat_name:<15} {kind:<35} {gen:<25} {steps:<6} {repeats:<8}")
                count += 1
        print(f"Total for {cat_name}: {count}")

if __name__ == "__main__":
    dump_cache_configs()
