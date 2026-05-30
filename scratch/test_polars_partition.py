import polars as pl

df = pl.DataFrame({
    "generator": ["A", "A", "B", "B"],
    "noise": [1, 1, 2, 2],
    "strategy": ["X", "Y", "X", "Y"],
    "val": [10, 20, 30, 40]
})

partitions = df.partition_by(["generator", "noise", "strategy"], as_dict=True)
for k, v in partitions.items():
    print(k)
    print(v)
