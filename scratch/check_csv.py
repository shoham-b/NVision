import polars as pl
from pathlib import Path

def check_csv():
    path = Path("artifacts/locator_results.csv")
    if not path.exists():
        print("CSV not found")
        return
    
    df = pl.read_csv(path)
    print(f"Total rows: {len(df)}")
    counts = df.group_by(["generator", "noise", "strategy"]).agg(pl.col("attempt").count().alias("n")).sort("n", descending=True)
    print(counts.head(20))

if __name__ == "__main__":
    check_csv()
