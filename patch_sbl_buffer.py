with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "r") as f:
    content = f.read()

content = content.replace("self._sweep_buffer.observations", "self._sweep_buffer")

with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "w") as f:
    f.write(content)
