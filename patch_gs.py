with open("nvision/sim/locs/coarse/generic_sweep_locator.py", "r") as f:
    content = f.read()

search_create = """    @classmethod
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
    ) -> GenericSweepLocator:"""

replace_create = """    @classmethod
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
    ) -> GenericSweepLocator:"""

content = content.replace(search_create, replace_create)

search_return = """        return cls(
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

replace_return = """        return cls(
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

content = content.replace(search_return, replace_return)

search_init = """    def __init__(
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

replace_init = """    def __init__(
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

content = content.replace(search_init, replace_init)

content = content.replace("from nvision.sim.locs.coarse.sweep_locator import SweepingLocator", "from nvision.models.locator import LocatorConfig\nfrom nvision.sim.locs.coarse.sweep_locator import SweepingLocator")

with open("nvision/sim/locs/coarse/generic_sweep_locator.py", "w") as f:
    f.write(content)
