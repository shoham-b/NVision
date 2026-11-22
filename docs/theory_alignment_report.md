# Theory alignment report (draft)

Status: draft, evidence being collected. See the final version for verdicts.

## Item 1. Lab refine falls to the sweep rate in wide windows

The lab runs are the lab's own MATLAB `BayesianESR`, not NVision's SBED. After detection, its focus window is locked
in 95/91/62 % of the steps at W = 5/10/20 MHz, and in 0 % at 50 and 100 MHz (median over runs, Nov 2 set). Placement
efficiency is 0.40–0.45 on locked steps at every window, and about the uniform-over-window value on unlocked steps.
Script: `debug/lab_placement_mechanism.py`.

## Item 2. Lab claimed sigma

- Nov 2 set (full 300 steps, C = 0.14, FWHM 1.82 MHz): the line jumps by 0.5–1 MHz between runs minutes apart.
  Consecutive-pair z is near calibrated (median |r| 0.63–1.16). `debug/lab_means_vs_time.py`,
  `debug/lab_segment_calibration.py`.
- runs_to_convergence set (stop at sigma < 1 MHz): configured C = 0.14, FWHM 3.5 MHz, but the per-run free fit gives
  C ~ 0.065. That is about 2.2x too much contrast, so I is about 4.6x too large and sigma about 2.1x too small.
  `debug/lab_pooled_lineshape.py`.

## Item 4. Calibration at 10–50 Gamma

End-of-run SBED 2-sigma coverage is 0.95/0.97/0.97 at spans 10/20/50 (250 runs each).
`debug/article_grid_calibration.py`.

## Item 5. Sweep rate

Measured sweep placement over 8 pi Gamma / (27 W_drive) is 0.86 at span 1.6, 0.95 at 5 and 0.97–1.01 at 10 and
above. Against the prior range W = span x Gamma, the gap is about span / (span + 1).
`debug/sweep_q_finite_window.py`.
