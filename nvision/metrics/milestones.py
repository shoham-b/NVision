"""Logic for detecting parameter-specific milestones in localization runs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from nvision.models.observer import RunResult
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
    NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD,
    PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS,
    param_convergence_bound_width,
)

# Raw saturation-Voigt parameters whose convergence is gated via the derived
# effective-HWHM / realized-contrast quantities instead of per-param thresholds.
# c_max is not included: it's a fixed constant (NV_SATURATION_C_MAX), not an
# inferred/estimated parameter, so it has no posterior uncertainty to gate on.
_SATURATION_VOIGT_RAW_PARAMS: tuple[str, ...] = ("saturation", "sigma_inhom")
# Inferred response of the sample to the microwave drive (drive-controlled models, see
# nvision/spectra/nv_center_drive.py). Nuisance parameters: excluded from the convergence targets.
DRIVE_RESPONSE_PARAMS: tuple[str, ...] = ("drive_amp_sat", "rabi_per_amp")


def saturation_voigt_derived_sigmas(
    estimates: Mapping[str, float], sigmas: Mapping[str, float]
) -> dict[str, float] | None:
    """Propagate raw saturation-Voigt per-param sigma-like values onto derived quantities.

    Maps ``sigmas`` entries (posterior uncertainties or CRLBs — both propagate the same
    way through the local linear Jacobian at *estimates*) for
    (``saturation``, ``sigma_inhom``) onto the derived effective-HWHM
    (keyed ``"hwhm"``) and realized-contrast (keyed ``"contrast_total"``) sigmas, so
    convergence semantics match the Lorentzian model's hwhm/contrast_total thresholds
    across lineshapes. The raw relative thresholds are physically inconsistent along
    the saturation axis (width scales as sqrt(1+s)), which is why derived gating is
    used instead. ``c_max`` is a fixed constant (:data:`~nvision.spectra.nv_center.
    NV_SATURATION_C_MAX`), so its contribution to ``sigma_c`` is exactly zero.

    Returns ``None`` unless the model exposes the saturation-Voigt parameter set.
    """
    if not all(k in estimates for k in _SATURATION_VOIGT_RAW_PARAMS):
        return None
    if "saturation" not in sigmas or "sigma_inhom" not in sigmas:
        return None
    from nvision.spectra.nv_center import (
        NV_SATURATION_C_MAX,
        saturation_voigt_effective_hwhm_and_unc,
        saturation_voigt_realized_contrast_and_unc,
    )

    s = float(estimates["saturation"])
    sigma_inhom = float(estimates["sigma_inhom"])
    _, sigma_omega = saturation_voigt_effective_hwhm_and_unc(
        s, sigma_inhom, float(sigmas["saturation"]), float(sigmas["sigma_inhom"])
    )
    _, sigma_c = saturation_voigt_realized_contrast_and_unc(s, NV_SATURATION_C_MAX, float(sigmas["saturation"]), 0.0)
    return {"hwhm": sigma_omega, "contrast_total": sigma_c}


def saturation_voigt_derived_bounds(
    phys_bounds: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Effective-HWHM / realized-contrast ranges implied by the raw parameter bounds.

    ``omega`` and ``C`` are monotone increasing in each of their inputs, so the range
    endpoints map directly. Used as the bound width for relative-threshold convergence
    of the derived quantities (keys match :func:`saturation_voigt_derived_sigmas`).
    ``c_max`` is the fixed :data:`~nvision.spectra.nv_center.NV_SATURATION_C_MAX`
    constant at both ends (it has no range — it isn't inferred).
    """
    from nvision.spectra.nv_center import (
        NV_SATURATION_C_MAX,
        saturation_voigt_effective_hwhm_and_unc,
        saturation_voigt_realized_contrast_and_unc,
    )

    s_lo, s_hi = phys_bounds.get("saturation", (0.0, 0.0))
    si_lo, si_hi = phys_bounds.get("sigma_inhom", (0.0, 0.0))
    omega_lo, _ = saturation_voigt_effective_hwhm_and_unc(s_lo, si_lo)
    omega_hi, _ = saturation_voigt_effective_hwhm_and_unc(s_hi, si_hi)
    c_lo, _ = saturation_voigt_realized_contrast_and_unc(s_lo, NV_SATURATION_C_MAX)
    c_hi, _ = saturation_voigt_realized_contrast_and_unc(s_hi, NV_SATURATION_C_MAX)
    return {"hwhm": (omega_lo, omega_hi), "contrast_total": (c_lo, c_hi)}


def target_params_converged(
    parameter_names: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    uncertainties: Mapping[str, float],
    estimates: Mapping[str, float],
    convergence_threshold: float,
) -> bool:
    """Whether every target parameter's uncertainty is under its ceiling at one moment of a run.

    The "all converged" criterion, shared by every locator (it is a measurement on the belief, not a property
    of how a locator chooses its next point). Convergence requires each target parameter's uncertainty to be
    below ``convergence_threshold`` as a fraction of its physical bound width (e.g. ``0.01`` = 1 %).
    Parameters in ``PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS`` use an absolute ceiling instead, and the overall
    (RMS) relative uncertainty must also be below the same threshold.

    ``parameter_names``: the model's parameters; ``bounds``, ``uncertainties``, ``estimates``: physical units,
    keyed by parameter name.
    """
    # The drive-response parameters describe the setup's coupling to the microwave field (a nuisance,
    # like an instrument calibration), not the spectrum being located; they never gate convergence.
    target_params = [name for name in parameter_names if name not in DRIVE_RESPONSE_PARAMS]

    # Saturation-Voigt: gate via derived effective-HWHM / realized-contrast instead
    # of the raw (saturation, sigma_inhom) params, whose relative thresholds are
    # physically inconsistent along the saturation axis.
    derived_sigmas: dict[str, float] | None = None
    if "saturation" in uncertainties and "sigma_inhom" in uncertainties:
        derived_sigmas = saturation_voigt_derived_sigmas(estimates, uncertainties)
    if derived_sigmas is not None:
        bounds = {**bounds, **saturation_voigt_derived_bounds(bounds)}

    check_items: list[tuple[str, float]] = [
        (name, float(uncertainties[name]))
        for name in target_params
        if name in uncertainties and not (derived_sigmas is not None and name in _SATURATION_VOIGT_RAW_PARAMS)
    ]
    if derived_sigmas is not None:
        check_items.extend(derived_sigmas.items())

    relative_uncertainties: dict[str, float] = {}
    for name, unc in check_items:
        if name == "zero_field_freq":
            # bound_width chosen so that unc / bound_width < threshold  <=>  unc < the absolute zero_field_freq
            # ceiling.
            bound_width = NVISION_ZERO_FIELD_FREQ_CONVERGENCE_THRESHOLD / convergence_threshold
        else:
            bound_width = param_convergence_bound_width(name, convergence_threshold, bounds)
        if bound_width <= 0:
            return False
        relative_uncertainties[name] = unc / bound_width

    if not relative_uncertainties:
        return False

    # Each individual parameter must be below threshold ...
    if not all(u < convergence_threshold for u in relative_uncertainties.values()):
        return False
    # ... and so must the overall (RMS) relative uncertainty.
    rms_uncertainty = float(np.sqrt(np.mean(np.array(list(relative_uncertainties.values())) ** 2)))
    return rms_uncertainty < convergence_threshold


def belief_target_params_converged(
    belief: Any,
    uncertainties: Mapping[str, float] | None = None,
    convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
) -> bool:
    """:func:`target_params_converged` on a belief (live or snapshotted); ``uncertainties`` default to its own.

    The belief's estimates are read only when the saturation-Voigt gating needs them.
    """
    if uncertainties is None:
        uncertainties = belief.uncertainty()
    needs_estimates = "saturation" in uncertainties and "sigma_inhom" in uncertainties
    return target_params_converged(
        belief.model.parameter_names(),
        belief.physical_param_bounds,
        uncertainties,
        belief.estimates() if needs_estimates else {},
        convergence_threshold,
    )


def detect_all_converged_step(
    run_result: RunResult, convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD
) -> int | None:
    """First step (1-indexed) at which every target parameter is converged, or None.

    Read off the recorded run's belief snapshots, so it is the same criterion for every locator.
    """
    for i, snapshot in enumerate(run_result.snapshots):
        if belief_target_params_converged(snapshot.belief, convergence_threshold=convergence_threshold):
            return i + 1
    return None


def convergence_threshold(
    run_result: RunResult,
    param: str,
    threshold: float = NVISION_CONVERGENCE_THRESHOLD,
    relative: bool = True,
) -> float:
    """Uncertainty ceiling (physical units) below which *param* counts as converged.

    The parameter's absolute ceiling when it has one, else ``threshold`` as a fraction of its prior
    bound width (``relative=True``), else ``threshold`` itself.
    """
    if param in PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS:
        return PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS[param]
    if relative:
        bounds = run_result.snapshots[0].belief.physical_param_bounds.get(param)
        if bounds:
            lo, hi = bounds
            return threshold * (hi - lo)
    return threshold


def detect_milestone(
    run_result: RunResult,
    param: str,
    threshold: float = NVISION_CONVERGENCE_THRESHOLD,
    relative: bool = True,
) -> int | None:
    """Find the first step where a parameter's uncertainty drops below a threshold.

    Args:
        run_result: Full trajectory of the run.
        param: Parameter name (e.g., 'zero_field_freq').
        threshold: Uncertainty threshold.
        relative: If True, threshold is relative to initial parameter range.

    Returns:
        Step index (0-indexed) or None if never converged.
    """
    if not run_result.snapshots:
        return None

    threshold = convergence_threshold(run_result, param, threshold, relative)

    for i, snapshot in enumerate(run_result.snapshots):
        uncert = snapshot.belief.uncertainty().get(param)
        if uncert is not None and uncert < threshold:
            return i

    return None


def default_shift_param(run_result: RunResult) -> str:
    """Shift parameter for the shift-error metrics: ``zeeman_shift`` when the model has it, else ``hyperfine_shift``.

    The default NV model is Zeeman-split (parameter ``zeeman_shift``); hyperfine models
    expose ``hyperfine_shift``. Hardcoding ``hyperfine_shift`` silently yields all-NaN shift metrics for Zeeman runs.
    """
    try:
        params = run_result.true_signal.parameter_values()
    except Exception:
        return "hyperfine_shift"
    return "zeeman_shift" if "zeeman_shift" in params else "hyperfine_shift"


_PRIMARY_PARAM_PREFERENCE = ("zeeman_shift", "hyperfine_shift", "zero_field_freq")


def resolve_primary_param(available_params: Iterable[str]) -> str | None:
    """The parameter whose convergence defines the locator's primary milestone.

    Prefers the model's splitting parameter (the actual free/scientific-interest
    quantity once zero_field_freq is fixed by default -- see ``with_fixed_zero_field_freq`` in
    nvision/spectra/nv_center.py), falling back to ``"zero_field_freq"`` itself for legacy
    free-zero_field_freq configurations. ``None`` if the model has neither (milestone
    tracking stays permanently unset, matching historical behavior for such models).
    """
    available = set(available_params)
    for candidate in _PRIMARY_PARAM_PREFERENCE:
        if candidate in available:
            return candidate
    return None


def extract_milestone_metrics(
    run_result: RunResult,
    step_idx: int,
    primary_param: str = "zero_field_freq",
    shift_param: str = "hyperfine_shift",
    override_estimates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract estimates and errors at a specific step milestone.

    ``override_estimates``, when given, is merged on top of the belief's raw
    estimates before ``primary_param``/``shift_param`` are read. Pass
    ``run_result.fit_mode_estimates`` here for the final-state milestone: sweep
    locators (``GenericSweepLocator``) defer their belief update to a batch flush
    that can collapse to a garbage marginal (see ``executor.py``), so the actual
    least-squares fit is the only trustworthy estimate at that step.

    Returns a dictionary of metrics at that step.
    """
    if step_idx >= len(run_result.snapshots):
        return {}

    snapshot = run_result.snapshots[step_idx]
    estimates = snapshot.belief.estimates()
    if override_estimates:
        estimates = {**estimates, **override_estimates}
    uncertainties = snapshot.belief.reported_uncertainty()

    param_values = run_result.true_signal.parameter_values()
    true_primary = param_values.get(primary_param, math.nan)
    true_shift = param_values.get(shift_param, math.nan)

    est_primary = estimates.get(primary_param, math.nan)
    est_shift = estimates.get(shift_param, math.nan)

    # Calculate overall uncertainty (mean of all parameters)
    overall_uncert = float(sum(uncertainties.values()) / len(uncertainties)) if uncertainties else math.nan

    return {
        "step": step_idx + 1,  # 1-indexed for display
        "est_primary": est_primary,
        "est_shift": est_shift,
        "err_primary": abs(est_primary - true_primary) if not math.isnan(est_primary) else math.nan,
        "err_shift": abs(est_shift - true_shift) if not math.isnan(est_shift) else math.nan,
        "uncert_primary": uncertainties.get(primary_param, math.nan),
        "overall_uncert": overall_uncert,
    }


def calculate_all_converged_metrics(
    run_result: RunResult,
    all_converged_step: int | None,
    primary_param: str = "zero_field_freq",
    shift_param: str | None = None,
) -> dict[str, Any]:
    """Error/uncertainty of the primary parameter at the all-converged milestone.

    ``all_converged_step`` (1-indexed) comes from :func:`detect_all_converged_step` on the recorded run.
    """
    if shift_param is None:
        shift_param = default_shift_param(run_result)

    step_idx = all_converged_step - 1 if all_converged_step is not None else -1
    if step_idx < 0 or step_idx >= len(run_result.snapshots):
        return {
            "err_primary_at_all_converged": None,
            "uncert_primary_at_all_converged": None,
        }

    ms = extract_milestone_metrics(run_result, step_idx, primary_param, shift_param)
    return {
        "err_primary_at_all_converged": ms["err_primary"],
        "uncert_primary_at_all_converged": ms["uncert_primary"],
    }


def calculate_zeeman_metrics(
    run_result: RunResult,
    threshold: float = NVISION_CONVERGENCE_THRESHOLD,
    primary_param: str = "zero_field_freq",
    shift_param: str | None = None,
) -> dict[str, Any]:
    """Compare the primary-parameter milestone to the final state.

    ``shift_param`` defaults to the model's splitting parameter (``zeeman_shift`` when
    present, else ``hyperfine_shift``). Returns aggregated metrics for the repeat.
    """
    if shift_param is None:
        shift_param = default_shift_param(run_result)

    # 1. Primary-parameter milestone
    primary_idx = detect_milestone(run_result, primary_param, threshold)

    metrics: dict[str, Any] = {}

    if primary_idx is not None:
        ms = extract_milestone_metrics(run_result, primary_idx, primary_param, shift_param)
        metrics.update(
            {
                "steps_to_primary": ms["step"],
                "err_primary_at_milestone": ms["err_primary"],
                "err_shift_at_milestone": ms["err_shift"],
                "primary_at_milestone": ms["est_primary"],
                "shift_at_milestone": ms["est_shift"],
                "uncert_primary_at_milestone": ms["uncert_primary"],
                "overall_uncert_at_milestone": ms["overall_uncert"],
            }
        )
    else:
        metrics.update(
            {
                "steps_to_primary": None,
                "err_primary_at_milestone": None,
                "err_shift_at_milestone": None,
                "primary_at_milestone": None,
                "shift_at_milestone": None,
                "uncert_primary_at_milestone": None,
                "overall_uncert_at_milestone": None,
            }
        )

    # 2. Final state
    final_idx = len(run_result.snapshots) - 1
    if final_idx >= 0:
        fs = extract_milestone_metrics(
            run_result, final_idx, primary_param, shift_param, override_estimates=run_result.fit_mode_estimates
        )
        metrics.update(
            {
                "final_err_primary": fs["err_primary"],
                "final_err_shift": fs["err_shift"],
                "final_overall_uncert": fs["overall_uncert"],
                "final_steps": fs["step"],
            }
        )

        # 3. Deltas
        if primary_idx is not None:
            metrics["err_primary_diff"] = metrics["err_primary_at_milestone"] - fs["err_primary"]
            metrics["err_shift_diff"] = metrics["err_shift_at_milestone"] - fs["err_shift"]

    return metrics
