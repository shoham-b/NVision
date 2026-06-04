"""Benchmark: graph creation and storage pipeline timing.

Measures each stage for the three main output types:
  - Scan plot (Plotly figure: true signal + measurement scatter)
  - Posterior data (plots_data custom JSON: particle history)
  - Plotly comparison figure (box plot)

Run with:
    uv run python scripts/bench_write.py
"""

from __future__ import annotations

import gzip
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _timer(label: str, fn, *args, repeats: int = 5, **kwargs):
    """Run fn(*args, **kwargs) `repeats` times, print per-step timing."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    best = min(times)
    avg = sum(times) / len(times)
    print(f"  {label:<42}  best={best * 1000:7.1f} ms  avg={avg * 1000:7.1f} ms")
    return result


# ---------------------------------------------------------------------------
# Scan plot benchmark  (Plotly figure -> .json.gz)
# ---------------------------------------------------------------------------


def _make_scan_figure(n_dense: int = 5000, n_meas: int = 200) -> go.Figure:
    """Approximate a real scan plot: dense true-signal line + measurement scatter."""
    x_dense = np.linspace(2.6e9, 3.1e9, n_dense)
    y_true = 1.0 - 0.3 * np.exp(-((x_dense - 2.87e9) ** 2) / (2 * (5e6) ** 2))

    m_x = RNG.uniform(2.6e9, 3.1e9, n_meas)
    m_y = y_true[np.searchsorted(x_dense, m_x).clip(0, n_dense - 1)] + RNG.normal(0, 0.01, n_meas)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_dense.tolist(), y=y_true.tolist(), mode="lines", name="true signal"))
    fig.add_trace(
        go.Scatter(
            x=m_x.tolist(),
            y=m_y.tolist(),
            mode="markers",
            name="measurements (inference)",
            marker=dict(size=6, color=list(range(n_meas)), colorscale="Viridis"),
        )
    )
    fig.update_layout(template="plotly_white", title="Scan")
    return fig


def bench_scan(tmp: Path):
    from nvision.viz._f32_json import fig_to_f32_json, write_plotly_gz

    print("\n=== Scan plot (5 000-pt signal + 200 measurements) ===")

    fig = _timer("Build Plotly figure", _make_scan_figure)

    # Baseline: plain fig.to_json() + write text
    def _plain_write(fig):
        p = tmp / "scan_plain.json"
        p.write_text(fig.to_json(), encoding="utf-8")
        return p.stat().st_size

    # New: Float32 JSON string
    def _f32_str(fig):
        return fig_to_f32_json(fig)

    # New: Float32 + gzip write
    def _gz_write(fig):
        p = tmp / "scan.json.gz"
        write_plotly_gz(fig, p)
        return p.stat().st_size

    plain_sz = _timer("Plain fig.to_json() + write", _plain_write, fig)
    f32_s = _timer("Float32 JSON encode (str only)", _f32_str, fig)
    gz_sz = _timer("Float32 + gzip write (.json.gz)", _gz_write, fig)
    print(f"  File size: plain={plain_sz // 1024} KB  gz={gz_sz // 1024} KB  ({plain_sz / gz_sz:.1f}x smaller)")


# ---------------------------------------------------------------------------
# Posterior data benchmark  (plots_data -> .json.gz)
# ---------------------------------------------------------------------------


def _make_posterior_inputs(n_frames: int, n_particles: int, n_params: int = 3):
    """Synthetic anim_all dict matching plots_data.write_posterior_data input."""
    param_names = ["frequency", "split", "linewidth"][:n_params]
    anim_all = {}
    for p in param_names:
        hist = []
        for _ in range(n_frames):
            vals = RNG.normal(0.5, 0.05, n_particles)
            w = RNG.dirichlet(np.ones(n_particles))
            hist.append(np.column_stack([vals, w]))
        anim_all[p] = (hist, np.linspace(0, 1, 2))
    return anim_all, param_names


def bench_posterior(tmp: Path):
    from nvision.runner.plots_data import write_posterior_data

    configs = [
        ("300 frames × 60 particles  [new default]", 300, 60),
        ("1000 frames × 200 particles [old default]", 1000, 200),
    ]

    for label, n_frames, n_particles in configs:
        print(f"\n=== Posterior data ({label}) ===")
        anim_all, _ = _make_posterior_inputs(n_frames, n_particles)

        # Old: raw float64 JSON to plain file
        def _old_write(anim_all):
            payload = {
                "schema": "posterior_v1",
                "param_names": ["frequency", "split", "linewidth"],
                "steps": [
                    {
                        p: {
                            "type": "particles",
                            "values": anim_all[p][0][i][:, 0].tolist(),
                            "weights": anim_all[p][0][i][:, 1].tolist(),
                        }
                        for p in anim_all
                    }
                    for i in range(n_frames)
                ],
            }
            p = tmp / "old_posterior.json"
            p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            return p.stat().st_size

        def _new_write(anim_all):
            p = tmp / "new_posterior.json.gz"
            write_posterior_data(
                anim_all,
                p,
                physical_bounds={
                    "frequency": (2.6e9, 3.1e9),
                    "split": (3e6, 8.5e6),
                    "linewidth": (200e3, 5e6),
                },
            )
            return p.stat().st_size

        old_sz = _timer("Old: float64 JSON + write       ", _old_write, anim_all)
        new_sz = _timer("New: Float32 + gzip write       ", _new_write, anim_all)
        print(f"  File size: old={old_sz // 1024} KB  new={new_sz // 1024} KB  ({old_sz / new_sz:.1f}x smaller)")


# ---------------------------------------------------------------------------
# Comparison plot benchmark (box plot Plotly figure)
# ---------------------------------------------------------------------------


def _make_comparison_figure(n_strategies: int = 5, n_repeats: int = 100) -> go.Figure:
    fig = go.Figure()
    for i in range(n_strategies):
        vals = RNG.normal(loc=i * 0.1, scale=0.05, size=n_repeats)
        fig.add_trace(go.Box(y=vals.tolist(), name=f"strategy_{i}"))
    fig.update_layout(template="plotly_white", title="Comparison")
    return fig


def bench_comparison(tmp: Path):
    from nvision.viz._f32_json import write_plotly_gz

    print("\n=== Comparison plot (5 strategies × 100 repeats box plot) ===")
    fig = _timer("Build Plotly figure", _make_comparison_figure)

    def _plain(fig):
        p = tmp / "comp_plain.json"
        p.write_text(fig.to_json(), encoding="utf-8")
        return p.stat().st_size

    def _gz(fig):
        p = tmp / "comp.json.gz"
        write_plotly_gz(fig, p)
        return p.stat().st_size

    plain_sz = _timer("Plain fig.to_json() + write", _plain, fig)
    gz_sz = _timer("Float32 + gzip write (.json.gz)", _gz, fig)
    print(f"  File size: plain={plain_sz // 1024} KB  gz={gz_sz // 1024} KB  ({plain_sz / gz_sz:.1f}x smaller)")


# ---------------------------------------------------------------------------
# Stage breakdown: where does the time go?
# ---------------------------------------------------------------------------


def bench_stage_breakdown(tmp: Path):
    from nvision.viz._f32_json import _encode_arrays

    print("\n=== Stage breakdown for scan figure ===")
    fig = _make_scan_figure()

    _timer("1. fig.to_plotly_json()", lambda: fig.to_plotly_json())
    d = fig.to_plotly_json()
    _timer("2. _encode_arrays(dict)", lambda: _encode_arrays(d))
    encoded = _encode_arrays(d)
    _timer("3. json.dumps(encoded)", lambda: json.dumps(encoded, separators=(",", ":")))
    s = json.dumps(encoded, separators=(",", ":")).encode("utf-8")
    _timer("4. gzip.compress(level=1)", lambda: gzip.compress(s, compresslevel=1))
    _timer("4b. gzip.compress(level=6)", lambda: gzip.compress(s, compresslevel=6))
    gz1 = gzip.compress(s, compresslevel=1)
    gz6 = gzip.compress(s, compresslevel=6)
    print(f"  Level-1 size: {len(gz1) // 1024} KB   Level-6 size: {len(gz6) // 1024} KB")

    _timer("5. json.dumps plain (no F32)", lambda: json.dumps(fig.to_plotly_json(), separators=(",", ":")))
    plain_s = json.dumps(fig.to_plotly_json(), separators=(",", ":")).encode("utf-8")
    _timer("5b. gzip.compress plain lvl1", lambda: gzip.compress(plain_s, compresslevel=1))
    gz_plain = gzip.compress(plain_s, compresslevel=1)
    print(f"  Plain+gzip-1 size: {len(gz_plain) // 1024} KB  vs F32+gzip-1: {len(gz1) // 1024} KB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print(f"Python {sys.version.split()[0]}  |  NumPy {np.__version__}")
    print("Each timing = best of 5 runs\n")

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        bench_scan(t)
        bench_posterior(t)
        bench_comparison(t)
        bench_stage_breakdown(t)
