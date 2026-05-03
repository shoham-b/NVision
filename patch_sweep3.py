with open("nvision/sim/locs/coarse/sweep_locator.py", "r") as f:
    content = f.read()

content = content.replace("ObservationHistory(max_steps)", "ObservationHistory(self.max_steps)")

with open("nvision/sim/locs/coarse/sweep_locator.py", "w") as f:
    f.write(content)
