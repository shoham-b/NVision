with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "r") as f:
    content = f.read()

content = content.replace("tuple(config.convergence.params) if config.convergence.params is not None else tuple(\\n            self.belief.model.parameter_names()\\n        )", "tuple(config.convergence.params) if config.convergence.params is not None else tuple(self.belief.model.parameter_names())")

with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "w") as f:
    f.write(content)
