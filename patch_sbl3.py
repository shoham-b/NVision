with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "r") as f:
    content = f.read()

s = """        self._noise_max_dev: float | None = (
            float(noise_max_dev) if (noise_max_dev is not None and noise_max_dev > 0) else None
        )
        # Physical max signal span (from signal spec's _signal_max_span bound).
        # Drives both sweep density and refocus window width directly.
        self._signal_max_span: float | None = (
            float(signal_max_span) if (signal_max_span is not None and signal_max_span > 0) else None
        )"""

r = """        self._noise_max_dev: float | None = (
            float(noise_max_dev) if (noise_max_dev is not None and noise_max_dev > 0) else None
        )
        # Physical max signal span (from signal spec's _signal_max_span bound).
        # Drives both sweep density and refocus window width directly.
        self._signal_max_span: float | None = (
            float(signal_max_span) if (signal_max_span is not None and signal_max_span > 0) else None
        )

        bounds = getattr(belief, "physical_param_bounds", None)
        if bounds is not None and self._scan_param in bounds:
            self._scan_lo, self._scan_hi = bounds[self._scan_param]
        else:
            self._scan_lo, self._scan_hi = 0.0, 1.0

        # Maintain the full physical domain mapping for the observer.
        # Returned ``x`` must stay normalized to this full range so ``measure()`` probes
        # the intended frequency, even when the belief is narrowed after the sweep.
        self._full_domain_lo, self._full_domain_hi = float(self._scan_lo), float(self._scan_hi)"""

if "self._scan_lo, self._scan_hi" not in content:
    content = content.replace(s, r)

s2 = """        # Create staged Sobol locator for initial sweep phase
        # belief is passed to satisfy Locator parent class
        # signal_model (belief.model) is used for sweep detection
        self._staged_sobol = StagedSobolSweepLocator(
            belief=self.belief,
            config=config,
            signal_model=self.belief.model,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            scan_param=self._scan_param,
        )"""

r2 = """        # Create staged Sobol locator for initial sweep phase
        # belief is passed to satisfy Locator parent class
        # signal_model (belief.model) is used for sweep detection
        self._staged_sobol = StagedSobolSweepLocator(
            belief=self.belief,
            config=config,
            signal_model=self.belief.model,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            scan_param=self._scan_param,
            domain_lo=self._full_domain_lo,
            domain_hi=self._full_domain_hi,
        )

        # Post-sweep interval in physical units where _acquire may search; starts at full scan.
        self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi

        # Initial sweep narrows parameter bounds physically without remapping to unit cube
        self._narrowed_param_bounds: dict[str, tuple[float, float]] = {}

        # Round-robin support for multi-dip narrowing
        self._per_dip_windows: list[tuple[float, float]] | None = None
        self._current_dip_window_idx: int = 0

        self._initial_sweep_completed_at_step: int = 0

        # Buffer for observations that arrive while the sweep locator is active
        self._sweep_buffer: list[Observation] = []"""

if "domain_lo=self._full_domain_lo" not in content:
    content = content.replace(s2, r2)

with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "w") as f:
    f.write(content)
