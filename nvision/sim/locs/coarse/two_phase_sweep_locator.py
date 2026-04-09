"""Two-phase locator: coarse Sobol search followed by finer sweep."""

from __future__ import annotations

from nvision.models.locator import Locator
from nvision.sim.locs.coarse.sobol_locator import SobolLocator
from nvision.sim.locs.coarse.sweep_locator import SimpleSweepLocator


class TwoPhaseSweepLocator(Locator):
    """Composite locator: coarse Sobol, then fine SimpleSweep."""

    def __init__(self, phase1: SobolLocator, phase2: SimpleSweepLocator):
        super().__init__(phase1.belief)
        self.phase1 = phase1
        self.phase2 = phase2
        self._in_phase1 = True

    @classmethod
    def create(
        cls,
        phase1_max_steps: int = 32,
        phase1_n_grid: int = 128,
        phase2_max_steps: int = 96,
        phase2_n_grid: int = 256,
        **kwargs,
    ) -> TwoPhaseSweepLocator:
        phase1 = SobolLocator.create(max_steps=phase1_max_steps, n_grid=phase1_n_grid)
        phase2 = SimpleSweepLocator.create(max_steps=phase2_max_steps, n_grid=phase2_n_grid)
        return cls(phase1=phase1, phase2=phase2)

    @property
    def _active(self) -> Locator:
        return self.phase1 if self._in_phase1 else self.phase2

    def next(self) -> float:
        return self._active.next()

    def observe(self, obs) -> None:
        self._active.observe(obs)
        # Keep top-level belief in sync with active phase
        self.belief = self._active.belief  # type: ignore[assignment]
        if self._in_phase1 and self.phase1.done():
            # For now, just switch; more advanced versions could restrict
            # phase2 domain around phase1.best_x.
            self._in_phase1 = False

    def done(self) -> bool:
        return (self._in_phase1 and self.phase1.done()) or (not self._in_phase1 and self.phase2.done())

    def result(self) -> dict[str, float]:
        return self._active.result()
