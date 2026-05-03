with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "r") as f:
    content = f.read()

content = content.replace("if self._sweep_buffer.count > 0:", "if len(self._sweep_buffer) > 0:")

with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "w") as f:
    f.write(content)
