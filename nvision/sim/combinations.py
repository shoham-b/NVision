"""Combination grid — the full (generator x noise x strategy) space.

``CombinationGrid`` is the single source of truth for which generators,
noises, and locator strategies exist and how they combine.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from nvision.belief.smc_marginal import (
    NVISION_SMC_A_PARAM,
    NVISION_SMC_ESS_THRESHOLD,
    NVISION_SMC_NUM_PARTICLES,
)
from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
from nvision.noises import OverFrequencyGaussianNoise
from nvision.sim import defaults as sim_defaults
from nvision.sim import presets as sim_presets
from nvision.sim.grid_enums import sweep_strategy_name
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.sim.locs.bayesian.sbed_drive_locator import DriveChoosingSBEDLocator
from nvision.sim.locs.bayesian.sbed_locator import (
    SequentialBayesianExperimentDesignLocator,
    SmartSBEDLocator,
    SmartSweepLocator,
)
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator


@dataclass(frozen=True, slots=True)
class Combination:
    """One (generator, noise, strategy) triple — the 'what' of a run."""

    generator_name: str
    generator: object
    noise_name: str
    noise: CompositeNoise | None
    strategy_name: str
    strategy: type | dict[str, Any]


# Sweep spacings studied, as grid points per prior FWHM (SWEEP_POINTS_PER_FWHM is the derived default).
SWEEP_SPACING_DIVISORS: tuple[int, ...] = (2, 3, 4)

_NV_SMC: dict[str, object] = {
    "builder": nv_center_smc_belief,
    "num_particles": NVISION_SMC_NUM_PARTICLES,
    "ess_threshold": NVISION_SMC_ESS_THRESHOLD,
    "a_param": NVISION_SMC_A_PARAM,
}

_GAUSS_RE = re.compile(r"^Gauss\(([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\)(?:\+Drift\(([A-Za-z0-9_-]+)\))?$")


def parse_gauss_sigma(name: str) -> float | None:
    """Extract the numeric sigma from a ``Gauss(sigma)`` (optionally ``+Drift(label)``) noise name.

    ``SNR(x)`` names have no sigma of their own: use :func:`noise_sigma_for` with the signal.

    Gives metrics/plots a numeric noise axis instead of the opaque preset string.
    """
    m = _GAUSS_RE.match(name)
    return float(m.group(1)) if m is not None else None


def parse_drift_label(name: str) -> str | None:
    """The drift scenario label from a ``Gauss(sigma)+Drift(label)`` or ``SNR(x)+Drift(label)`` name, else ``None``."""
    m = _GAUSS_RE.match(name) or _SNR_RE.match(name)
    return m.group(2) if m is not None else None


_SNR_RE = re.compile(r"^SNR\(([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\)(?:\+Drift\(([A-Za-z0-9_-]+)\))?$")


snr_noise_name = sim_presets.snr_noise_name


def parse_snr(name: str) -> float | None:
    """The SNR of an ``SNR(x)`` (optionally ``+Drift(label)``) noise name; ``None`` for any other name.

    A run is identified by its SNR, not by the noise sigma and contrast that make it: the sigma of an ``SNR(x)``
    noise is derived per signal (:func:`sigma_for_snr`), so the name keeps its meaning if the contrast constant moves.
    """
    m = _SNR_RE.match(name)
    return float(m.group(1)) if m is not None else None


def reference_contrast(generator: object) -> float:
    """The contrast an SNR is measured against: the signal's fixed ``contrast_total``.

    A generator that draws its contrast per repeat (or gets it from the drive) has none; its SNR is then nominal,
    against ``NVISION_SBED_SNR_CONTRAST``.
    """
    contrast = getattr(generator, "contrast_total", None)
    return float(contrast) if contrast is not None else float(sim_defaults.NVISION_SBED_SNR_CONTRAST)


def sigma_for_snr(generator: object, snr: float) -> float:
    """The per-shot Gaussian sigma of an SNR on this signal: ``reference_contrast / snr``."""
    if not snr > 0:
        raise ValueError(f"SNR must be positive, got {snr}")
    return reference_contrast(generator) / snr


def noise_sigma_for(generator: object, noise_name: str) -> float | None:
    """The numeric Gaussian sigma a noise name stands for on this signal (``None`` for names without one)."""
    snr = parse_snr(noise_name)
    return sigma_for_snr(generator, snr) if snr is not None else parse_gauss_sigma(noise_name)


def _snr_noise(generator: object, name: str) -> CompositeNoise | None:
    """The noise object of an ``SNR(x)[+Drift(label)]`` name on this signal; ``None`` if the name is not one."""
    m = _SNR_RE.match(name)
    if m is None:
        return None
    drift = None
    if m.group(2) is not None:
        drift = sim_presets.drift_scenario(m.group(2))
        if drift is None:
            return None
    sigma = sigma_for_snr(generator, float(m.group(1)))
    return CompositeNoise(
        over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(sigma)]),
        drift=drift,
    )


def _parse_noise(name: str) -> CompositeNoise | None:
    """Dynamically parse a ``Gauss(sigma)`` or ``Gauss(sigma)+Drift(label)`` noise descriptor.

    Makes ``run-single`` work for any sigma value even when
    :envvar:`NVISION_NOISE_MAX_GAUSS` caps the preset grid below the requested level.

    Args:
        name: A string like ``'Gauss(0.05)'`` or ``'Gauss(0.006)+Drift(warmup)'``.

    Returns:
        A :class:`CompositeNoise` for the parsed descriptor, or ``None`` if *name* does
        not match the expected pattern or names an unknown drift scenario.
    """
    m = _GAUSS_RE.match(name)
    if m is None:
        return None
    sigma = float(m.group(1))
    drift = None
    if m.group(2) is not None:
        drift = sim_presets.drift_scenario(m.group(2))
        if drift is None:
            return None
    return CompositeNoise(
        over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(sigma)]),
        drift=drift,
    )


def _drive_lineshape_for(generator_name: str) -> str | None:
    """Belief lineshape of a drive-controlled generator (see sim.presets.drive_controlled_generators), else None."""
    if generator_name.startswith("NVCenter-cw-drive"):
        return "cw_drive"
    if generator_name.startswith("NVCenter-pulsed-drive"):
        return "pulsed_drive"
    return None


def _strategy_matches(pattern: str, strat_name: str) -> bool:
    """Match strategy name against pattern (regex if valid, else substring)."""
    # Try regex first
    try:
        if pattern.startswith("^") or pattern.endswith("$") or any(c in pattern for c in ".*+?[]{}|()"):
            return bool(re.search(pattern, strat_name))
    except re.error:
        pass
    # Fall back to simple substring match
    return pattern in strat_name


class CombinationGrid:
    """Enumerates every (generator x noise x strategy) combination.

    Holds the full configuration grid in one place so the runner, CLI,
    and render code never duplicate the mapping logic.
    """

    def __init__(self, extra_generators: dict[str, object] | None = None) -> None:
        self._generators: dict[str, object] = dict(sim_presets.generators_basic())
        # Name -> generator lookup used only by resolve(), never by iter()/all_combinations().
        # Always includes the SBED run-groups' study grids so cache/metrics/render
        # tooling can resolve historical results by name regardless of how they were produced,
        # without expanding what plain (non-run-group) `nvision run` iterates over.
        self._resolve_generators: dict[str, object] = dict(self._generators)
        self._resolve_generators.update(dict(sim_presets.param_grid_generators()))
        self._resolve_generators.update(dict(sim_presets.param_grid_generators(variant="voigt")))
        self._resolve_generators.update(dict(sim_presets.saturation_voigt_param_grid_generators()))
        # Also registered here (a pre-existing gap this fixes): without it,
        # strategies_for() has no generator object to read the real hyperfine value
        # off of for this grid and has to fall back to parsing it out of the name.
        self._resolve_generators.update(dict(sim_presets.voigt_sigma_inhom_param_grid_generators()))
        self._resolve_generators.update(dict(sim_presets.voigt_snr_window_param_grid_generators()))
        self._resolve_generators.update(dict(sim_presets.drive_controlled_generators()))
        if extra_generators:
            self._resolve_generators.update(extra_generators)
        self._noises: dict[str, CompositeNoise | None] = dict(sim_presets.noises_single_each())

    @property
    def generators(self) -> dict[str, object]:
        return self._generators

    @property
    def noises(self) -> dict[str, CompositeNoise | None]:
        return self._noises

    @staticmethod
    def generator_category(name: str) -> str:
        if name.startswith("NVCenter-"):
            return "NVCenter"
        return "Unknown"

    def strategies_for(self, generator_name: str) -> list[tuple[str, Any]]:
        """Return the locator strategies appropriate for *generator_name*."""
        # Belief inference must match the generator's lineshape, or the locator
        # models the wrong signal shape entirely. nv_center_smc_belief defaults
        # to "lorentzian"; only study grids for other lineshapes need an override.
        # Scoped to the "-w"-suffixed width x contrast grid names specifically --
        # the bare "NVCenter-voigt" generator from generators_basic() has the same
        # gap but is intentionally left alone here (pre-existing, separate issue).
        nv_smc_config = dict(_NV_SMC)
        drive_lineshape = _drive_lineshape_for(generator_name)
        if drive_lineshape is not None:
            nv_smc_config["lineshape"] = drive_lineshape
        elif generator_name.startswith("NVCenter-saturation_voigt"):
            nv_smc_config["lineshape"] = "saturation_voigt"
        elif generator_name.startswith("NVCenter-voigt-fw"):
            # voigt_snr_window_param_grid_generators(): a fixed-isotope Voigt signal whose window is N FWHM.
            # The belief must resolve the same isotope as the signal, so it is read off the generator object.
            # infer_hyperfine stays False: the isotope is a grid coordinate here, not something to infer.
            resolved = self._resolve_generators.get(generator_name)
            if resolved is None:
                raise ValueError(f"unknown SNR-window generator {generator_name!r}")
            nv_smc_config["lineshape"] = "voigt"
            nv_smc_config["hyperfine"] = resolved.hyperfine
        elif generator_name.startswith("NVCenter-voigt-w"):
            nv_smc_config["lineshape"] = "voigt"
            # param_grid_generators(variant="voigt") (names WITHOUT a "-si" suffix)
            # explicitly draws a real ¹⁴N hyperfine triplet per Zeeman group
            # (hyperfine="n14", infer_hyperfine=True, see presets.py) to preserve that
            # grid's historical 6-dip behavior -- the belief must match. The separate
            # voigt_sigma_inhom_param_grid_generators() grid (names WITH a "-si"
            # suffix) leaves the generator at hyperfine="unresolved" (a plain Zeeman-only
            # 2-dip signal) -- forcing the triplet here for it as well was a real
            # belief/generator mismatch (the belief hunting for a hyperfine_shift/k_np hyperfine
            # substructure that the true signal never has), so it's excluded from this
            # override and falls through to nv_center_smc_belief's own
            # hyperfine="unresolved" default.
            if "-si" not in generator_name:
                nv_smc_config["hyperfine"] = "n14"
                nv_smc_config["infer_hyperfine"] = True
            else:
                # voigt_sigma_inhom_param_grid_generators() (registered in
                # self._resolve_generators above) may sweep hyperfine as its own
                # axis -- the belief must be told the true signal's actual value, or
                # nv_center_smc_belief falls back to its own "unresolved" default
                # regardless of what the signal really resolves, exactly the
                # generator/belief mismatch described above. Read straight off the
                # resolved generator object (its `hyperfine` field, see
                # NVCenterCoreGenerator) rather than re-derived from the name
                # string -- structurally correct for any name this grid produces,
                # including ones predating the hyperfine axis (whose generator
                # object already carries the "unresolved" it was actually built
                # with). infer_hyperfine stays False: the isotope is a fixed, known
                # grid coordinate here (like sigma_inhom/contrast), not something
                # the locator infers per repeat.
                resolved = self._resolve_generators.get(generator_name)
                hyperfine = getattr(resolved, "hyperfine", None)
                if hyperfine is not None:
                    nv_smc_config["hyperfine"] = hyperfine

        sbed_config = {"max_steps": sim_defaults.NVISION_SBED_MAX_STEPS, **nv_smc_config}

        def sweep(spacing_divisor: int) -> dict[str, Any]:
            # Sweeps infer with the same belief as SBED (same builder/prior), so their uncertainty
            # trajectories are comparable; the spacing is the prior FWHM / spacing_divisor.
            return {"class": GenericSweepLocator, "config": {**nv_smc_config, "spacing_divisor": spacing_divisor}}

        strats = [
            ("Bayesian-SBED", {"class": SequentialBayesianExperimentDesignLocator, "config": sbed_config}),
            ("Smart-SBED", {"class": SmartSBEDLocator, "config": sbed_config}),
            ("Smart-Sweep", {"class": SmartSweepLocator, "config": sbed_config}),
            # The sweep spacings under study, named by their points per FWHM; the analytically derived
            # default is SWEEP_POINTS_PER_FWHM.
            *((sweep_strategy_name(k), sweep(k)) for k in SWEEP_SPACING_DIVISORS),
        ]
        if drive_lineshape is not None:
            # Only drive-controlled generators have a drive to choose.
            strats.append(
                (
                    "Bayesian-SBED-Drive",
                    {
                        "class": DriveChoosingSBEDLocator,
                        "config": {"max_steps": sim_defaults.NVISION_SBED_MAX_STEPS, **nv_smc_config},
                    },
                )
            )

        return strats

    def __iter__(self) -> Iterator[Combination]:
        """Iterate all combinations (no filtering, no dedup)."""
        return self.all_combinations()

    def iter(
        self,
        filter_category: str | None = None,
        filter_strategy: str | None = None,
        filter_generator: str | None = None,
        filter_noise: str | None = None,
        filter_signal: str | None = None,
    ) -> Iterator[Combination]:
        """Yield every matching combination, deduplicating automatically.

        .. deprecated::
            Filtering via ``iter()`` is kept for backward compatibility.
            Prefer :meth:`all_combinations` or :meth:`resolve`.
        """
        seen: set[tuple[str, str, str]] = set()

        for gen_name, gen_obj in self._generators.items():
            if filter_category and self.generator_category(gen_name) != filter_category:
                continue
            if filter_generator is not None and gen_name != filter_generator:
                continue
            if filter_signal is not None and filter_signal not in gen_name:
                continue

            for strat_name, strat_obj in self.strategies_for(gen_name):
                if filter_strategy:
                    # Support comma-separated strategy filters with optional regex
                    # (e.g., "Sweep,SBED" matches both, "^Bayesian.*" uses regex)
                    patterns = [p.strip() for p in filter_strategy.split(",")]
                    if not any(_strategy_matches(p, strat_name) for p in patterns):
                        continue

                for noise_name, noise_obj in self._noises.items():
                    if filter_noise is not None:
                        noise_patterns = [p.strip() for p in filter_noise.split(",")]
                        if not any(noise_name.startswith(p) for p in noise_patterns):
                            continue
                    key = (gen_name, noise_name, strat_name)
                    if key in seen:
                        continue
                    seen.add(key)

                    yield Combination(
                        generator_name=gen_name,
                        generator=gen_obj,
                        noise_name=noise_name,
                        noise=noise_obj,
                        strategy_name=strat_name,
                        strategy=strat_obj,
                    )

    def all_combinations(self) -> Iterator[Combination]:
        """Yield every registered combination (full Cartesian product)."""
        seen: set[tuple[str, str, str]] = set()
        for gen_name, gen_obj in self._generators.items():
            for strat_name, strat_obj in self.strategies_for(gen_name):
                for noise_name, noise_obj in self._noises.items():
                    key = (gen_name, noise_name, strat_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield Combination(
                        generator_name=gen_name,
                        generator=gen_obj,
                        noise_name=noise_name,
                        noise=noise_obj,
                        strategy=strat_obj,
                        strategy_name=strat_name,
                    )

    def resolve(self, gen_name: str, noise_name: str, strat_name: str) -> Combination | None:
        """Resolve three preset names to a single :class:`Combination`.

        Returns ``None`` if the generator name or strategy name is not registered.

        For *noise_name*, the registered grid is checked first.  If the name is not
        found there (e.g. because :envvar:`NVISION_NOISE_MAX_GAUSS` caps the preset
        grid below the requested sigma), :func:`_parse_noise` is used to build the
        noise object on the fly from the descriptor string (e.g. ``'Gauss(0.15)'``).
        This ensures ``run-single`` works for any valid noise value regardless of
        the active grid configuration.
        """
        gen_obj = self._resolve_generators.get(gen_name)
        if gen_obj is None:
            return None
        noise_obj: CompositeNoise | None
        if parse_snr(noise_name) is not None:
            # An SNR noise has no sigma of its own: it comes from this signal's contrast.
            noise_obj = _snr_noise(gen_obj, noise_name)
            if noise_obj is None:
                return None
        elif noise_name in self._noises:
            noise_obj = self._noises[noise_name]
        else:
            # Build the noise on the fly for any descriptor not in the current grid.
            noise_obj = _parse_noise(noise_name)
            if noise_obj is None:
                return None
        for s_name, s_obj in self.strategies_for(gen_name):
            if s_name == strat_name:
                return Combination(
                    generator_name=gen_name,
                    generator=gen_obj,
                    noise_name=noise_name,
                    noise=noise_obj,
                    strategy_name=strat_name,
                    strategy=s_obj,
                )
        return None
