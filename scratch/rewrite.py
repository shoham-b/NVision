with open("nvision/viz/__init__.py", "r") as f:
    code = f.read()

import re

search = """        # Iterate over unique combinations of generator, noise, and strategy
        unique = df_loc.select(["generator", "noise", "strategy"]).unique()
        for row in unique.iter_rows(named=True):
            gen = row["generator"]
            noise = row["noise"]
            strat = row["strategy"]
            # Retrieve metrics for this combo from the dataframe rows
            subset = df_loc.filter(
                (pl.col("generator") == gen)
                & (pl.col("noise") == noise)
                & (pl.col("strategy") == strat)
            )"""

replace = """        # Iterate over unique combinations of generator, noise, and strategy
        partitions = df_loc.partition_by(["generator", "noise", "strategy"], as_dict=True)
        for (gen, noise, strat), subset in partitions.items():"""

code = code.replace(search, replace)

with open("nvision/viz/__init__.py", "w") as f:
    f.write(code)
