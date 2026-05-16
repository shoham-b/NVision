import sqlite3
import json
from pathlib import Path

def dump_shard():
    path = Path("artifacts/cache/nv_center_shard0001.db")
    if not path.exists():
        print("Shard not found")
        return
    
    conn = sqlite3.connect(path)
    cur = conn.execute("SELECT key, value FROM cache LIMIT 50")
    
    counts = {}
    total = 0
    for (value_str,) in conn.execute("SELECT value FROM cache"):
        total += 1
        try:
            val = json.loads(value_str)
            if isinstance(val, dict):
                kind = val.get("config", {}).get("kind")
                if not kind:
                    kind = val.get("type", "no-kind")
                counts[kind] = counts.get(kind, 0) + 1
            else:
                counts["non-dict"] = counts.get("non-dict", 0) + 1
        except:
            counts["invalid-json"] = counts.get("invalid-json", 0) + 1
            
    print(f"Total entries: {total}")
    print("Entry counts by kind/type:")
    for kind, count in counts.items():
        print(f"  {kind}: {count}")
    conn.close()

if __name__ == "__main__":
    dump_shard()
