with open("nvision/sim/locs/coarse/sweep_locator.py", "r") as f:
    content = f.read()

s1 = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ):
        super().__init__(belief)
        # Signal model is independent of belief - used for sweep detection
        self.signal_model = signal_model
        self.max_steps = max_steps
        self.step_count = 0
        self.noise_std = noise_std
        self._noise_std = noise_std"""
r1 = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        config: LocatorConfig,
        signal_model: SignalModel,
        *,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ):
        super().__init__(belief)
        self.config = config
        # Signal model is independent of belief - used for sweep detection
        self.signal_model = signal_model
        self.max_steps = int(config.max_steps)
        self.step_count = 0
        noise_std = config.noise_std if config.noise_std is not None else 0.01
        self.noise_std = noise_std
        self._noise_std = noise_std"""
content = content.replace(s1, r1)
content = content.replace("from nvision.models.locator import Locator", "from nvision.models.locator import Locator, LocatorConfig")

with open("nvision/sim/locs/coarse/sweep_locator.py", "w") as f:
    f.write(content)

with open("nvision/sim/locs/coarse/sobol_locator.py", "r") as f:
    content = f.read()

content = content.replace("from nvision.models.locator import Locator", "from nvision.models.locator import Locator, LocatorConfig")
s2 = """    @classmethod
    def create(
        cls,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> SobolSweepLocator:"""
r2 = """    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        *,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        **kwargs: object,
    ) -> SobolSweepLocator:"""
content = content.replace(s2, r2)

s3 = """        return cls(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )"""
r3 = """        return cls(
            belief=belief,
            config=config,
            signal_model=signal_model,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )"""
content = content.replace(s3, r3)

s4 = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ):
        super().__init__(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )"""
r4 = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        config: LocatorConfig,
        signal_model: SignalModel,
        *,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ):
        super().__init__(
            belief=belief,
            config=config,
            signal_model=signal_model,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )"""
content = content.replace(s4, r4)

s5 = """    @classmethod
    def create(
        cls,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int = 300,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> StagedSobolSweepLocator:"""
r5 = """    @classmethod
    def create(
        cls,
        config: LocatorConfig,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        *,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        **kwargs: object,
    ) -> StagedSobolSweepLocator:"""
content = content.replace(s5, r5)

s6 = """        return cls(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
        )"""
r6 = """        return cls(
            belief=belief,
            config=config,
            signal_model=signal_model,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
        )"""
content = content.replace(s6, r6)

s7 = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
    ):
        super().__init__(belief)
        self.signal_model = signal_model
        self.max_steps = max_steps
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi
        self.noise_std = noise_std"""
r7 = """    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        config: LocatorConfig,
        signal_model: SignalModel,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        *,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
    ):
        super().__init__(belief)
        self.config = config
        self.signal_model = signal_model
        self.max_steps = int(config.max_steps)
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi
        self.noise_std = float(config.noise_std if config.noise_std is not None else 0.01)"""
content = content.replace(s7, r7)

with open("nvision/sim/locs/coarse/sobol_locator.py", "w") as f:
    f.write(content)
