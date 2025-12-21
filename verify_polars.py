import polars as pl

try:
    df = pl.DataFrame()
    print(f"Polars Version: {pl.__version__}")
    print(f"Has is_empty: {hasattr(df, 'is_empty')}")
    if hasattr(df, "is_empty"):
        print(f"is_empty result: {df.is_empty()}")
    print(f"Height: {df.height}")
except Exception as e:
    print(e)
