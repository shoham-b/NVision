import sqlite3
import json
from pathlib import Path

def scan_all_db_files():
    cache_dir = Path("artifacts_old/cache")
    db_files = list(cache_dir.glob("*.db"))
    
    print(f"{'File':<40} {'Total Rows':<10} {'Kinds Found'}")
    print("-" * 80)
    
    for db_path in db_files:
        try:
            conn = sqlite3.connect(db_path)
            # Try to see if it has a 'cache' table or 'cache_index' table
            tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            
            row_count = 0
            kinds = set()
            
            if "cache" in tables:
                for (val_str,) in conn.execute("SELECT value FROM cache"):
                    row_count += 1
                    try:
                        val = json.loads(val_str)
                        if isinstance(val, dict):
                            kind = val.get("config", {}).get("kind")
                            if not kind:
                                kind = val.get("type", "no-kind")
                            kinds.add(kind)
                    except:
                        pass
            elif "cache_index" in tables:
                 row_count = conn.execute("SELECT count(*) FROM cache_index").fetchone()[0]
                 kinds.add("index")
            
            print(f"{db_path.name:<40} {row_count:<10} {', '.join(kinds)}")
            conn.close()
        except Exception as e:
            print(f"{db_path.name:<40} ERROR: {e}")

if __name__ == "__main__":
    scan_all_db_files()
