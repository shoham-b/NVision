import re

def update_file(filename, search, replace):
    with open(filename, 'r') as f:
        content = f.read()
    if search in content:
        content = content.replace(search, replace)
        # also add LocatorConfig import if needed
        if "from nvision.models.locator import LocatorConfig" not in content:
            if "from nvision.models.locator import Locator" in content:
                content = content.replace("from nvision.models.locator import Locator", "from nvision.models.locator import Locator, LocatorConfig")
            elif "from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator" in content:
                content = content.replace("from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator", "from nvision.models.locator import LocatorConfig\nfrom nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator")
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Updated {filename}")
    else:
        print(f"Search string not found in {filename}")

# sbed_locator.py
sbed_s = """    def __init__(
        self,
        belief,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        noise_std: float = 0.02,
        utility_method: str = "variance_approx",
        n_candidates: int = 200,
        n_draws: int = 100,
    ) -> None:
        super().__init__(belief, max_steps, convergence_threshold, scan_param, noise_std=noise_std)
        self.utility_method = utility_method
        self.n_candidates = int(n_candidates)
        self.n_draws = int(n_draws)

    @classmethod
    def create(
        cls,
        builder=None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds=None,
        noise_std: float | None = None,
        utility_method: str = "variance_approx",
        n_candidates: int = 200,
        n_draws: int = 100,
        **grid_config,
    ):
        if builder is None:
            raise ValueError(f"{cls.__name__} requires a 'builder' callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            noise_std=noise_std,
            utility_method=utility_method,
            n_candidates=n_candidates,
            n_draws=n_draws,
        )"""
sbed_r = """    def __init__(
        self,
        belief,
        config: LocatorConfig,
        scan_param: str | None = None,
        utility_method: str = "variance_approx",
        n_candidates: int = 200,
        n_draws: int = 100,
    ) -> None:
        super().__init__(belief, config=config, scan_param=scan_param)
        self.utility_method = utility_method
        self.n_candidates = int(n_candidates)
        self.n_draws = int(n_draws)

    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        builder=None,
        scan_param: str | None = None,
        parameter_bounds=None,
        utility_method: str = "variance_approx",
        n_candidates: int = 200,
        n_draws: int = 100,
        **grid_config,
    ):
        if builder is None:
            raise ValueError(f"{cls.__name__} requires a 'builder' callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            config=config,
            scan_param=scan_param,
            utility_method=utility_method,
            n_candidates=n_candidates,
            n_draws=n_draws,
        )"""

update_file("nvision/sim/locs/bayesian/sbed_locator.py", sbed_s, sbed_r)

# students_t_locator.py
t_s = """    def __init__(
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
        self.df = max(1.0, float(df))

        self.step_count = 0
        self.inference_step_count = 0

    @classmethod
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
    ) -> StudentsTLocator:
        # We enforce the parametric belief here, so we extract the model and use it
        model = signal_model
        if model is None:
            if builder is not None:
                # Create a dummy belief just to extract the model
                dummy_belief = builder(parameter_bounds, **grid_config)
                model = dummy_belief.model
            else:
                raise ValueError("StudentsTLocator requires either signal_model or a builder.")

        bounds = dict(parameter_bounds) if parameter_bounds else {}
        belief = StudentsTMixtureMarginalDistribution(
            model=model,
            _physical_param_bounds=bounds,
            dfs=np.array([df])
        )

        return cls(
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

t_r = """    def __init__(
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
        self.df = max(1.0, float(df))

        self.step_count = 0
        self.inference_step_count = 0

    @classmethod
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
    ) -> StudentsTLocator:
        # We enforce the parametric belief here, so we extract the model and use it
        model = signal_model
        if model is None:
            if builder is not None:
                # Create a dummy belief just to extract the model
                dummy_belief = builder(parameter_bounds, **grid_config)
                model = dummy_belief.model
            else:
                raise ValueError("StudentsTLocator requires either signal_model or a builder.")

        bounds = dict(parameter_bounds) if parameter_bounds else {}
        belief = StudentsTMixtureMarginalDistribution(
            model=model,
            _physical_param_bounds=bounds,
            dfs=np.array([df])
        )

        return cls(
            belief=belief,
            config=config,
            scan_param=scan_param,
            initial_sweep_builder=initial_sweep_builder,
            df=df,
        )"""

update_file("nvision/sim/locs/bayesian/students_t_locator.py", t_s, t_r)

# maximum_likelihood_locator.py
ml_s = """    def __init__(
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
        )

    @classmethod
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
    ) -> MaximumLikelihoodLocator:
        if builder is None:
            raise ValueError("MaximumLikelihoodLocator requires a builder callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief=belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
        )"""
ml_r = """    def __init__(
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

    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        **grid_config: object,
    ) -> MaximumLikelihoodLocator:
        if builder is None:
            raise ValueError("MaximumLikelihoodLocator requires a builder callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief=belief,
            config=config,
            scan_param=scan_param,
        )"""
update_file("nvision/sim/locs/bayesian/maximum_likelihood_locator.py", ml_s, ml_r)

# utility_sampling_locator.py
u_s = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        pickiness: float = 4.0,
        noise_std: float = 0.02,
        cost: float = 1.0,
        n_mc_samples: int = 64,
        n_candidates: int = 64,
    ) -> None:
        super().__init__(belief, max_steps, convergence_threshold, scan_param, noise_std=noise_std)
        self.pickiness = float(max(0.0, pickiness))
        self.noise_std = float(max(1e-9, noise_std))
        self.cost = float(max(1e-9, cost))
        self.n_mc_samples = int(max(8, n_mc_samples))
        self.n_candidates = int(max(8, n_candidates))

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution],
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        pickiness: float = 4.0,
        noise_std: float = 0.02,
        cost: float = 1.0,
        n_mc_samples: int = 64,
        n_candidates: int = 64,
        **grid_config: object,
    ) -> UtilitySamplingLocator:
        if builder is None:
            raise ValueError("UtilitySamplingLocator requires a builder callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            pickiness=pickiness,
            noise_std=noise_std,
            cost=cost,
            n_mc_samples=n_mc_samples,
            n_candidates=n_candidates,
        )"""

u_r = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        config: LocatorConfig,
        scan_param: str | None = None,
        pickiness: float = 4.0,
        cost: float = 1.0,
        n_mc_samples: int = 64,
        n_candidates: int = 64,
    ) -> None:
        super().__init__(belief, config=config, scan_param=scan_param)
        self.pickiness = float(max(0.0, pickiness))
        self.noise_std = float(max(1e-9, config.noise_std if config.noise_std is not None else 0.02))
        self.cost = float(max(1e-9, cost))
        self.n_mc_samples = int(max(8, n_mc_samples))
        self.n_candidates = int(max(8, n_candidates))

    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        builder: Callable[..., AbstractMarginalDistribution],
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        pickiness: float = 4.0,
        cost: float = 1.0,
        n_mc_samples: int = 64,
        n_candidates: int = 64,
        **grid_config: object,
    ) -> UtilitySamplingLocator:
        if builder is None:
            raise ValueError("UtilitySamplingLocator requires a builder callable.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            config=config,
            scan_param=scan_param,
            pickiness=pickiness,
            cost=cost,
            n_mc_samples=n_mc_samples,
            n_candidates=n_candidates,
        )"""
update_file("nvision/sim/locs/bayesian/utility_sampling_locator.py", u_s, u_r)
