with open('nvision/sim/locs/bayesian/maximum_likelihood_locator.py', 'r') as f:
    content = f.read()

search = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
    ) -> None:
        super().__init__(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
        )"""

replace = """    def __init__(
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

content = content.replace(search, replace)

search_create = """    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        **grid_config: object,
    ) -> MaximumLikelihoodLocator:"""

replace_create = """    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        **grid_config: object,
    ) -> MaximumLikelihoodLocator:"""

content = content.replace(search_create, replace_create)

search_return = """        return cls(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
        )"""

replace_return = """        return cls(
            belief=belief,
            config=config,
            scan_param=scan_param,
        )"""

content = content.replace(search_return, replace_return)
content = content.replace("from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator", "from nvision.models.locator import LocatorConfig\nfrom nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator")

with open('nvision/sim/locs/bayesian/maximum_likelihood_locator.py', 'w') as f:
    f.write(content)
