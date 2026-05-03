with open("nvision/sim/locs/coarse/generic_sweep_locator.py", "r") as f:
    content = f.read()

search = """    def __init__(
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
replace = """    def __init__(
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

# I need to see what arguments SweepingLocator.__init__ actually takes
