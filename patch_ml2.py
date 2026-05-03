with open("nvision/sim/locs/bayesian/maximum_likelihood_locator.py", "r") as f:
    content = f.read()

s = """    def next(self) -> float:
        \"\"\"Propose next measurement with staged initial-sweep warm-start.\"\"\"
        self.step_count += 1

        if not self._staged_sobol.done():"""

# Wait, `MaximumLikelihoodLocator` in `nvision/sim/locs/bayesian/maximum_likelihood_locator.py` inherits from `SequentialBayesianLocator` and uses `_acquire`.
# Wait! MaximumLikelihoodLocator inherits from SequentialBayesianLocator but I only patched `__init__` and `create`.
