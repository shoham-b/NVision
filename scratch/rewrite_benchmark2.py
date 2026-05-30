with open("debug/benchmark_run.py", "r") as f:
    code = f.read()

search = """    console.print(
        f"[bold blue]SBED Benchmark[/bold blue]: steps={steps}, particles={particles}, points_per_feature={points_per_feature}"
    )"""

replace = """    console.print(
        f"[bold blue]SBED Benchmark[/bold blue]: steps={steps}, "
        f"particles={particles}, points_per_feature={points_per_feature}"
    )"""

code = code.replace(search, replace)
with open("debug/benchmark_run.py", "w") as f:
    f.write(code)
