with open("nvision/sim/locs/bayesian/maximum_likelihood_locator.py", "r") as f:
    content = f.read()

s = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        config: LocatorConfig,
        scan_param: str | None = None,
    ) -> None:
        super().__init__(
            belief=belief,
            config=config,
            scan_param=scan_param,
        )"""

r = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        config: LocatorConfig,
        scan_param: str | None = None,
    ) -> None:
        super().__init__(
            belief=belief,
            config=config,
            scan_param=scan_param,
        )
        self._full_domain_lo = 0.0
        self._full_domain_hi = 1.0"""

# Actually, the base class SequentialBayesianLocator gets _full_domain_lo from belief
# Wait, why did it complain? `AttributeError: 'MaximumLikelihoodLocator' object has no attribute '_full_domain_lo'`
# Let's check `SequentialBayesianLocator.__init__` again
