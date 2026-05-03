with open('nvision/sim/locs/bayesian/students_t_locator.py', 'r') as f:
    content = f.read()

search = """    def __init__(
        self,
        belief: StudentsTMixtureMarginalDistribution,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        df: float = 3.0,
    ) -> None:
        super().__init__(belief)
        self.belief: StudentsTMixtureMarginalDistribution = belief
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold

        # Use first parameter as default scan parameter if none provided
        self._scan_param = scan_param or (
            self.belief.model.parameter_names()[0] if self.belief.model.parameter_names() else "peak_x"
        )
        self.initial_sweep_steps = initial_sweep_steps or 20
        self._initial_sweep_builder = initial_sweep_builder

        self.convergence_params = convergence_params or [self._scan_param]
        self.convergence_patience_steps = convergence_patience_steps
        self._convergence_streak = 0

        self.noise_std = noise_std
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
        signal_model=None,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
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
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            initial_sweep_builder=initial_sweep_builder,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
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
content = content.replace("from nvision.models.locator import Locator", "from nvision.models.locator import Locator, LocatorConfig")

with open('nvision/sim/locs/bayesian/students_t_locator.py', 'w') as f:
    f.write(content)
