"""Demonstration of the new core architecture.

This example shows:
1. Creating a TrueSignal with known parameters
2. Creating a BeliefSignal with uniform prior
3. A simple sweep locator using the new architecture
4. Running with Observer to track convergence
5. Accessing trajectory data from RunResult
"""

import random

import numpy as np

from nvision import (
    BeliefSignal,
    CoreExperiment,
    Locator,
    LorentzianModel,
    Observer,
    ParameterWithPosterior,
    TrueSignal,
    run_loop,
)


class SimpleSweepLocator(Locator):
    """Simple grid sweep locator for demonstration."""

    def __init__(self, belief: BeliefSignal, max_steps: int = 50):
        super().__init__(belief)
        self.max_steps = max_steps
        self.step_count = 0
        self.grid_positions = np.linspace(0.0, 1.0, max_steps)

    @classmethod
    def create(cls, max_steps: int = 50, **kwargs):
        """Create fresh locator with uniform prior."""
        model = LorentzianModel()

        # Create uniform priors for each parameter
        belief = BeliefSignal(
            model=model,
            parameters=[
                ParameterWithPosterior(
                    name="frequency",
                    bounds=(0.2, 0.8),
                    grid=np.linspace(0.2, 0.8, 50),
                    posterior=np.ones(50) / 50,
                ),
                ParameterWithPosterior(
                    name="linewidth",
                    bounds=(0.01, 0.1),
                    grid=np.linspace(0.01, 0.1, 30),
                    posterior=np.ones(30) / 30,
                ),
                ParameterWithPosterior(
                    name="amplitude",
                    bounds=(0.1, 1.0),
                    grid=np.linspace(0.1, 1.0, 30),
                    posterior=np.ones(30) / 30,
                ),
                ParameterWithPosterior(
                    name="background",
                    bounds=(0.95, 1.05),
                    grid=np.linspace(0.95, 1.05, 20),
                    posterior=np.ones(20) / 20,
                ),
            ],
        )

        return cls(belief, max_steps)

    def next(self) -> float:
        """Return next grid position."""
        x = self.grid_positions[self.step_count]
        self.step_count += 1
        return x

    def done(self) -> bool:
        """Stop after max_steps."""
        return self.step_count >= self.max_steps

    def result(self) -> dict[str, float]:
        """Return final estimates."""
        return self.belief.estimates()


def main():
    """Run demonstration."""
    print("=" * 60)
    print("Core Architecture Demonstration")
    print("=" * 60)

    # 1. Create true signal
    print("\n1. Creating true signal...")
    model = LorentzianModel()
    true_parameters = [
        Parameter(name="frequency", bounds=(0.2, 0.8), value=0.5),
        Parameter(name="linewidth", bounds=(0.01, 0.1), value=0.05),
        Parameter(name="amplitude", bounds=(0.1, 1.0), value=0.5),
        Parameter(name="background", bounds=(0.95, 1.05), value=1.0),
    ]
    true_signal = TrueSignal(model=model, parameters=true_parameters)
    print(f"   True parameters: {[(p.name, p.value) for p in true_parameters]}")

    # 2. Create experiment
    print("\n2. Creating experiment...")
    experiment = CoreExperiment(
        true_signal=true_signal,
        noise=None,
        x_min=0.0,
        x_max=1.0,
    )

    # 3. Run with observer
    print("\n3. Running localization...")
    rng = random.Random(42)
    observer = Observer(true_signal, x_min=0.0, x_max=1.0)

    result = observer.watch(run_loop(SimpleSweepLocator, experiment, rng, max_steps=30))

    print(f"   Steps taken: {result.num_steps()}")
    print(f"   Final estimates: {result.final_estimates()}")

    # 4. Show convergence
    print("\n4. Convergence analysis:")
    for param_name in ["frequency", "linewidth", "amplitude"]:
        errors = result.error_trajectory(param_name)
        uncertainties = result.uncertainty_trajectory(param_name)
        print(f"\n   {param_name}:")
        print(f"      Initial error: {errors[0]:.6f}")
        print(f"      Final error:   {errors[-1]:.6f}")
        print(f"      Initial uncertainty: {uncertainties[0]:.6f}")
        print(f"      Final uncertainty:   {uncertainties[-1]:.6f}")

    # 5. Show entropy trajectory
    print("\n5. Entropy trajectory:")
    entropies = result.entropy_trajectory()
    print(f"   Initial entropy: {entropies[0]:.2f}")
    print(f"   Final entropy:   {entropies[-1]:.2f}")
    print(f"   Reduction:       {entropies[0] - entropies[-1]:.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
