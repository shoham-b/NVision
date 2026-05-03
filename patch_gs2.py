with open("nvision/sim/locs/coarse/generic_sweep_locator.py", "r") as f:
    content = f.read()

content = content.replace("max_steps // 2", "int(config.max_steps) // 2")
content = content.replace("self._generate_sweep_points(max_steps)", "self._generate_sweep_points(int(config.max_steps))")

with open("nvision/sim/locs/coarse/generic_sweep_locator.py", "w") as f:
    f.write(content)
