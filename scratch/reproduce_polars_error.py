import polars as pl

df_rows = [
    {"a": 1, "b": "foo"},
    {"a": 2.0, "b": "bar"}
]

try:
    df = pl.DataFrame(df_rows)
    print("Success")
    print(df)
except Exception as e:
    print(f"Failed: {e}")

df_rows_2 = [
    {"a": 1, "b": "foo"},
    {"a": 2.5, "b": "bar"}
]

try:
    df = pl.DataFrame(df_rows_2)
    print("Success 2")
    print(df)
except Exception as e:
    print(f"Failed 2: {e}")
