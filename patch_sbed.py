import sys

with open('nvision/sim/locs/bayesian/sbed_locator.py', 'r') as f:
    content = f.read()

# SBED has no convergence parameters or initial sweep steps to begin with, so we just check.
# Let's see its signature:
#     def __init__(
#         self,
#         belief,
#         config: LocatorConfig,
#         scan_param: str | None = None,
#         utility_method: str = "variance_approx",
#         n_candidates: int = 200,
#         n_draws: int = 100,
#     ) -> None:

# It seems SBED is already good because it doesn't take those extra args anyway.

# Let's check students_t
with open('nvision/sim/locs/bayesian/students_t_locator.py', 'r') as f:
    content = f.read()

search = """    def __init__(
        self,
        belief: StudentsTMixtureMarginalDistribution,
        config: LocatorConfig,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        df: float = 3.0,
    ) -> None:
        super().__init__(belief)
        self.belief: StudentsTMixtureMarginalDistribution = belief
        self.config = config
        self.max_steps = int(config.max_steps)
        self.convergence_threshold = config.convergence_threshold

        # Use first parameter as default scan parameter if none provided
        self._scan_param = scan_param or (
            self.belief.model.parameter_names()[0] if self.belief.model.parameter_names() else "peak_x"
        )
        self.initial_sweep_steps = initial_sweep_steps or 20
        self._initial_sweep_builder = initial_sweep_builder

        self.convergence_params = convergence_params or [self._scan_param]
        self.convergence_patience_steps = convergence_patience_steps
        self._convergence_streak = 0

        self.noise_std = config.noise_std
        self.df = max(1.0, float(df))"""

replace = """    def __init__(
        self,
        belief: StudentsTMixtureMarginalDistribution,
        config: LocatorConfig,
        scan_param: str | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        df: float = 3.0,
    ) -> None:
        super().__init__(belief)
        self.belief: StudentsTMixtureMarginalDistribution = belief
        self.config = config
        self.max_steps = int(config.max_steps)
        self.convergence_threshold = config.convergence.threshold

        # Use first parameter as default scan parameter if none provided
        self._scan_param = scan_param or (
            self.belief.model.parameter_names()[0] if self.belief.model.parameter_names() else "peak_x"
        )
        self.initial_sweep_steps = config.initial_sweep_steps if config.initial_sweep_steps is not None else 20
        self._initial_sweep_builder = initial_sweep_builder

        self.convergence_params = config.convergence.params or [self._scan_param]
        self.convergence_patience_steps = config.convergence.patience_steps
        self._convergence_streak = 0

        self.noise_std = config.noise_std if config.noise_std is not None else 0.05
        self.df = max(1.0, float(df))"""

content = content.replace(search, replace)

search_create = """    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        signal_model=None,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        df: float = 3.0,
        **grid_config: object,
    ) -> StudentsTLocator:"""

replace_create = """    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        signal_model=None,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        df: float = 3.0,
        **grid_config: object,
    ) -> StudentsTLocator:"""

content = content.replace(search_create, replace_create)

search_return = """        return cls(
            belief=belief,
            config=config,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            initial_sweep_builder=initial_sweep_builder,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            df=df,
        )"""

replace_return = """        return cls(
            belief=belief,
            config=config,
            scan_param=scan_param,
            initial_sweep_builder=initial_sweep_builder,
            df=df,
        )"""

content = content.replace(search_return, replace_return)

with open('nvision/sim/locs/bayesian/students_t_locator.py', 'w') as f:
    f.write(content)
