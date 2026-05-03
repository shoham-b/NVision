import re

with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "r") as f:
    content = f.read()

content = content.replace("from nvision.models.locator import Locator", "from nvision.models.locator import Locator, LocatorConfig")

init_search = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int = 450,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
    ) -> None:
        super().__init__(belief)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.convergence_threshold = convergence_threshold
        # Total measurement count (includes initial sweep + Bayesian steps).
        self.step_count: int = 0
        # Bayesian acquisition count only (excludes initial sweep).
        self.inference_step_count: int = 0
        self._scan_param = scan_param or belief.model.parameter_names()[0]
        # Default convergence target is all model parameters.
        self._convergence_params: tuple[str, ...] = (
            tuple(convergence_params) if convergence_params is not None else tuple(self.belief.model.parameter_names())
        )
        self._convergence_patience_steps = max(1, int(convergence_patience_steps))
        self._convergence_streak = 0
        # Stored for noise-aware dip detection during sweep refocus.
        if noise_std is None or float(noise_std) <= 0:
            raise ValueError("noise_std must be a positive float; got %r" % noise_std)
        self._noise_std: float = float(noise_std)
        # Pre-computed maximum expected noise deviation for mid-sweep threshold.
        # When provided by the executor (via CompositeNoise.estimated_max_noise_deviation),
        # this is used directly as the dip threshold, bypassing the IQR fallback.
        self._noise_max_dev: float | None = (
            float(noise_max_dev) if (noise_max_dev is not None and noise_max_dev > 0) else None
        )
        # Physical max signal span (from signal spec's _signal_max_span bound).
        # Drives both sweep density and refocus window width directly.
        self._signal_max_span: float | None = (
            float(signal_max_span) if (signal_max_span is not None and signal_max_span > 0) else None
        )

        # --------------------------------------------------------------
        # Configuration for the initial coarse sweep
        # --------------------------------------------------------------
        if initial_sweep_steps is None:
            self._initial_sweep_steps_target = self.DEFAULT_INITIAL_SWEEP_STEPS
        else:
            self._initial_sweep_steps_target = max(0, int(initial_sweep_steps))"""

init_replace = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        config: LocatorConfig,
        scan_param: str | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
    ) -> None:
        super().__init__(belief)
        self.config = config
        self.max_steps = int(config.max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.convergence_threshold = config.convergence.threshold
        # Total measurement count (includes initial sweep + Bayesian steps).
        self.step_count: int = 0
        # Bayesian acquisition count only (excludes initial sweep).
        self.inference_step_count: int = 0
        self._scan_param = scan_param or belief.model.parameter_names()[0]
        # Default convergence target is all model parameters.
        self._convergence_params: tuple[str, ...] = (
            tuple(config.convergence.params) if config.convergence.params is not None else tuple(self.belief.model.parameter_names())
        )
        self._convergence_patience_steps = max(1, int(config.convergence.patience_steps))
        self._convergence_streak = 0
        # Stored for noise-aware dip detection during sweep refocus.
        if config.noise_std is None or float(config.noise_std) <= 0:
            raise ValueError("noise_std must be a positive float; got %r" % config.noise_std)
        self._noise_std: float = float(config.noise_std)
        # Pre-computed maximum expected noise deviation for mid-sweep threshold.
        # When provided by the executor (via CompositeNoise.estimated_max_noise_deviation),
        # this is used directly as the dip threshold, bypassing the IQR fallback.
        self._noise_max_dev: float | None = (
            float(noise_max_dev) if (noise_max_dev is not None and noise_max_dev > 0) else None
        )
        # Physical max signal span (from signal spec's _signal_max_span bound).
        # Drives both sweep density and refocus window width directly.
        self._signal_max_span: float | None = (
            float(signal_max_span) if (signal_max_span is not None and signal_max_span > 0) else None
        )

        # --------------------------------------------------------------
        # Configuration for the initial coarse sweep
        # --------------------------------------------------------------
        if config.initial_sweep_steps is None:
            self._initial_sweep_steps_target = self.DEFAULT_INITIAL_SWEEP_STEPS
        else:
            self._initial_sweep_steps_target = max(0, int(config.initial_sweep_steps))"""

content = content.replace(init_search, init_replace)

create_search = """    @classmethod
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
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        **grid_config: object,
    ) -> SequentialBayesianLocator:
        \"\"\"Generic factory for model-agnostic Bayesian locators.

        Subclasses for specific models (like NVCenterBayesianLocator) should
        override this to provide their own hard-coded belief setup.
        \"\"\"
        if builder is None:
            raise ValueError(
                f"{cls.__name__} requires a 'builder' callable to create the AbstractMarginalDistribution."
            )
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )"""

create_replace = """    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        **grid_config: object,
    ) -> SequentialBayesianLocator:
        \"\"\"Generic factory for model-agnostic Bayesian locators.

        Subclasses for specific models (like NVCenterBayesianLocator) should
        override this to provide their own hard-coded belief setup.
        \"\"\"
        if builder is None:
            raise ValueError(
                f"{cls.__name__} requires a 'builder' callable to create the AbstractMarginalDistribution."
            )
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            config=config,
            scan_param=scan_param,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )"""

content = content.replace(create_search, create_replace)

with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "w") as f:
    f.write(content)
