# Microwave Drive Control (CW and pulsed ODMR)

Most NVision generators describe a spectrum at one fixed, implicit microwave drive: the locator
only chooses *where* to measure (`drive_freq`). In a real ODMR setup the experimenter also sets the
**microwave drive amplitude**, and it is a real design lever:

* **Search.** A strong drive broadens the dip (power broadening in CW, Rabi broadening in pulsed), so
  while the dip's location is still unknown a probe is much more likely to land on it.
* **Precision.** Once the dip is found, a weak drive gives a narrow line with a steep slope, so each
  measurement pins down the dip position more tightly.

The **drive-controlled** generators make the drive a per-measurement setting, and the
**`Bayesian-SBED-Drive`** locator chooses it together with the drive frequency at every step.

## Coordinates and objects

| Object | Meaning |
|---|---|
| `DriveSetting(drive_amp_unit, pulse_duration_s=None)` (`nvision/spectra/drive.py`) | The drive of one measurement. `drive_amp_unit` is the field amplitude as a fraction of the source maximum, in `[DRIVE_AMP_MIN_UNIT = 0.05, 1]`. `pulse_duration_s` is set for pulsed ODMR only. |
| `Observation.drive` | The drive the measurement was **actually taken at**, always recorded, including the baseline drive. |
| `TrueSignal.baseline_drive` | The drive used by every locator that does not choose one (`Locator.next_drive()` returns `None`). It is an explicit experiment setting, not a fallback. |
| `SignalModel.controls_drive` | True for drive-controlled models. They **cannot be evaluated directly**: use `model.at_drive(drive)` (a fixed-drive view, an ordinary frequency-only model over the same parameters) or `compute_vectorized_many_driven` (many measurements, each at its own drive). |

## Physics

### CW ODMR: `NVCenter-cw-drive` (`NVCenterCWDriveModel`)

The drive amplitude $a$ sets the saturation parameter (power ∝ field²):

$$ s = \left(\frac{a}{a_\mathrm{sat}}\right)^2 $$

The existing saturation-Voigt physics then follows. The homogeneous HWHM is
$\Gamma_0\sqrt{1+s}$ (power broadening) and the contrast is $c_\max\, s/(1+s)$. Both are evaluated
by the saturation-Voigt kernels. The inferred parameter is `drive_amp_sat` ($a_\mathrm{sat}$), with
bounds `[0.05, 0.5]`. These bounds put $s(a=1)$ in the range 4–400, always past the CW slope optimum
$s \approx 2$, so "always drive at maximum" is never optimal and the amplitude is a real trade-off.
`sigma_inhom` (inhomogeneous width) and `zeeman_shift` are inferred as usual.

### Pulsed ODMR: `NVCenter-pulsed-drive` (`NVCenterPulsedDriveModel`)

A single square pulse at Rabi frequency $\Omega = \kappa\, a$ ($\kappa$ = `rabi_per_amp`, in Hz at
full amplitude) and duration $t$. The spin-flip probability at detuning $\Delta$ (Hz) is

$$ P(\Delta) = \frac{\Omega^2}{\Omega^2+\Delta^2}\,\sin^2\!\left(\pi t \sqrt{\Omega^2+\Delta^2}\right) $$

$P$ is averaged over a Gaussian inhomogeneous detuning spread (`sigma_inhom`, 7-node
Gauss–Hermite), and the signal is $1 - c_\mathrm{total}\sum_\text{lines} w\,\bar P$. The inferred
parameters are `rabi_per_amp`, `contrast_total`, `sigma_inhom` and `zeeman_shift`. The line HWHM is
$\approx 0.8\,\Omega$, so the width scales with the amplitude.

**The pulse is calibrated from the belief, not from an oracle.** The locator sets
$t = 1/(2\,\Omega_\mathrm{est})$ with $\Omega_\mathrm{est} = \hat\kappa\, a$, where $\hat\kappa$ is
the current posterior estimate. The truth uses the real $\Omega$, so early pulses are
mis-calibrated, which lowers the contrast, and improve as $\kappa$ is learned. Fixed-drive baselines
use a π-pulse calibrated from the repeat's *prior mean* $\kappa$: a rough calibration, so they get no
oracle advantage either.

**Aliasing.** On resonance the depth is $\sin^2(\tfrac{\pi}{2}\, r)$ with $r = \kappa/\hat\kappa$.
This is periodic: $r$ and $2-r$ give the same depth, and $r$ is the same at every amplitude, so
varying the amplitude cannot break the alias through the contrast alone. The `rabi_per_amp`
posterior may therefore be multimodal, and it is handled as follows:

* The SMC carries `rabi_per_amp` as an ordinary particle dimension, so its weighted particles can
  hold several modes between resamples. **The resampling step is not mode-aware, though**: Liu–West
  shrinks every particle toward the *global* mean (`a = 0.98`) and nudges with the *global*
  covariance, which drags particles into the gap between aliased modes (see Limitations).
* The calibration uses the posterior's **dominant mode** (`dominant_mode` in
  `sbed_drive_locator.py`), never the weighted mean. The mean between two aliased modes is a
  meaningless Rabi frequency.
* The off-resonance lineshape is *not* symmetric under $r \leftrightarrow 2-r$, and the width scales
  with $\Omega$. EIG therefore picks detunings that separate the branches.

The drive-response parameters (`drive_amp_sat`, `rabi_per_amp`) are nuisance parameters. They are
excluded from the convergence targets (`DRIVE_RESPONSE_PARAMS`); `zeeman_shift` stays primary.

## The `Bayesian-SBED-Drive` locator

`DriveChoosingSBEDLocator` (`nvision/sim/locs/bayesian/sbed_drive_locator.py`) subclasses
Bayesian-SBED and changes only the acquisition. Each step it scores the joint grid of
*drive candidates × in-focus drive-frequency candidates* by EIG
(`SMCMarginalDistribution.expected_information_gain_driven`, which returns shape
`(n_drives, n_freqs)`) and picks one pair with the usual chunked softmax. Drive candidates are
rebuilt once per belief epoch (each resample), from the posterior rather than a hand-tuned grid:

* **CW:** 7 amplitudes spanning $s \approx 0.3$–$30$ over the posterior's 10–90% `drive_amp_sat`
  range ($a = a_\mathrm{sat}\sqrt{s}$), plus full amplitude for search.
* **Pulsed:** 8 amplitudes log-spaced over `[0.05, 1]`, each with the π-pulse from the current
  `rabi_per_amp` mode.

The prediction matrix is cached for the epoch, exactly like the frequency-only EIG. Exploration
steps draw a uniform drive frequency and a log-uniform amplitude.

Every other locator supports both modes unchanged. CW vs pulsed is a property of the generator's
model, and a locator that does not choose the drive measures at `baseline_drive`; its belief is built
on the model's fixed-drive view (`nv_center_smc_belief(..., fixed_drive=...)`). The step budget is
identical for `Bayesian-SBED` and `Bayesian-SBED-Drive`.

## Running

```bash
uv run nv run-single NVCenter-cw-drive "Gauss(0.01)" Bayesian-SBED-Drive --repeats 1 --runners 1 --dry-run
uv run nv groups drive-control --repeats 5
```

The `drive-control` run group compares `Bayesian-SBED` (fixed baseline drive) with
`Bayesian-SBED-Drive` on both generators over the SBED noise grid. In the results UI the measurement
points of a drive-choosing run are coloured by **drive amplitude**, so the high-then-low pattern is
visible. The dense true-signal and belief curves are drawn at the baseline drive.

## Limitations (v1)

* **Resampling is unimodal.** Liu–West shrinkage toward the global mean, and nudges from the global
  covariance, smear an aliased (multimodal) `rabi_per_amp` posterior. This is a plausible
  contributor to confidently wrong pulsed runs at strong Rabi coupling. In those runs the very wide
  full-amplitude Rabi line, with its side lobes, is fitted with the wrong `zeeman_shift` and a
  shifted coupling.
* The plots' oracle CRLB uses the baseline drive. For the drive-choosing locator that is the bound a
  *fixed-drive* experiment could reach, not its own.
* The drive constants (prior bounds, baseline amplitude, Rabi range) are not part of
  `PHYSICS_CONFIG_FINGERPRINT`. Changing them does not invalidate cached drive-controlled results:
  delete those combinations' cache entries (or re-run with `--no-cache`) after such a change.

* Hyperfine structure is the fixed isotope geometry; `infer_hyperfine` is not supported.
* The saturated CW contrast `c_max` stays a fixed constant. With a controllable drive it becomes
  identifiable, which is a possible follow-up.
* Dip detection (`find_dips`) runs on raw observations taken at mixed amplitudes, so dips are
  detected with mixed depths. It only shapes candidate placement and the focus window.
* MATLAB replay has no drive axis: a drive-choosing locator is rejected there.
