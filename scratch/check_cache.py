import sqlite3
import json
from pathlib import Path

def check_cache():
    db_path = Path("artifacts/cache/complementary_index.db")
    if not db_path.exists():
        print(f"DB not found: {db_path.absolute()}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # In ComplementaryCacheBackend, the index usually has a 'key' and 'value' (metadata)
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table_name, in tables:
            # The schema usually has 'metadata' or 'value' column
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            for row in rows:
                for col in row:
                    if isinstance(col, str) and "{" in col:
                        try:
                            meta = json.loads(col)
                            strategy = meta.get("strategy")
                            if strategy and "StudentsT" in strategy:
                                print(f"Found in {table_name}: {meta.get('generator')} | {meta.get('noise')} | {strategy}")
                        except:
                            pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_cache()
