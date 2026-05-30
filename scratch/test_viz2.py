import polars as pl

df_loc = pl.DataFrame({
    "generator": ["Gen1", "Gen1", "Gen2"],
    "noise": [0.1, 0.1, 0.2],
    "strategy": ["S1", "S1", "S2"],
    "metrics": [{"absolute_errors": {"x": 1}, "convergence_steps": {"x": 10}},
                {"absolute_errors": {"x": 2}, "convergence_steps": {"x": 20}},
                {"absolute_errors": {"x": 3}, "convergence_steps": {"x": 30}}]
})

partitions = df_loc.partition_by(["generator", "noise", "strategy"], as_dict=True)
for (gen, noise, strat), subset in partitions.items():
    print(gen, noise, strat)
    print(subset)
