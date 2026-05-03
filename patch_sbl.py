import re

with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "r") as f:
    content = f.read()

s = """        if config.initial_sweep_steps is None:
            self._initial_sweep_steps_target = self.DEFAULT_INITIAL_SWEEP_STEPS
        else:
            self._initial_sweep_steps_target = max(0, int(config.initial_sweep_steps))"""

r = """        if config.initial_sweep_steps is None:
            self.initial_sweep_steps = self.DEFAULT_INITIAL_SWEEP_STEPS
        else:
            self.initial_sweep_steps = max(0, int(config.initial_sweep_steps))

        # Create staged Sobol locator for initial sweep phase
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

content = content.replace(s, r)
with open("nvision/sim/locs/bayesian/sequential_bayesian_locator.py", "w") as f:
    f.write(content)
