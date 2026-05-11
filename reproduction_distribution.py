
import numpy as np
import plotly.graph_objects as go
import polars as pl
from pathlib import Path

def reproduction():
    # 200 points coarse sweep (spread out)
    x_coarse = np.linspace(2.8675, 2.8685, 200)
    # 500 points at resonance 1 (very narrow)
    x_res1 = np.random.normal(2.8692, 0.00005, 500)
    # 300 points at resonance 2 (very narrow)
    x_res2 = np.random.normal(2.8702, 0.00005, 300)

    x_meas = np.concatenate([x_coarse, x_res1, x_res2])

    history = pl.DataFrame({"x": x_meas})

    xs_dense = np.linspace(2.867, 2.872, 5000)
    ys_dense = np.ones_like(xs_dense)
    ys_dense[np.abs(xs_dense - 2.8692) < 0.0002] = 0.8
    ys_dense[np.abs(xs_dense - 2.8702) < 0.0002] = 0.6

    # Original logic
    n_bins = max(24, min(80, int(np.sqrt(x_meas.size) * 8)))
    counts, edges = np.histogram(x_meas, bins=n_bins, range=(float(xs_dense.min()), float(xs_dense.max())))

    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts.astype(float) / max(1.0, float(counts.max()))

    kernel = np.array([0.2, 0.6, 0.2], dtype=float)
    density_smooth = np.convolve(density, kernel, mode="same")

    y_min = float(min(ys_dense))
    y_max = float(max(ys_dense))
    y_span = max(1e-9, y_max - y_min)
    y_band_base = y_min + 0.02 * y_span
    y_band_height = 0.18 * y_span
    y_curve = y_band_base + density_smooth * y_band_height

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_dense, y=ys_dense, name="true signal", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=centers, y=y_curve, name="measurement distribution", fill="tozeroy", fillcolor="rgba(111,66,193,0.2)", line=dict(color="purple")))

    # Proposed logic (sqrt scaling + more bins + more smoothing)
    n_bins_new = max(40, min(200, int(np.sqrt(x_meas.size) * 10)))
    counts_new, edges_new = np.histogram(x_meas, bins=n_bins_new, range=(float(xs_dense.min()), float(xs_dense.max())))
    centers_new = 0.5 * (edges_new[:-1] + edges_new[1:])

    # Sqrt scaling
    density_new = np.sqrt(counts_new.astype(float)) / max(1.0, np.sqrt(float(counts_new.max())))

    # Larger smoothing kernel for 200 bins
    kernel_size = 5
    kernel_new = np.ones(kernel_size) / kernel_size
    density_smooth_new = np.convolve(density_new, kernel_new, mode="same")

    y_curve_new = y_band_base + density_smooth_new * y_band_height
    fig.add_trace(go.Scatter(x=centers_new, y=y_curve_new, name="measurement distribution (new)", fill="tozeroy", fillcolor="rgba(255,0,0,0.2)", line=dict(color="red")))

    fig.write_html("reproduction_distribution_v2.html")
    print(f"n_bins={n_bins}, n_bins_new={n_bins_new}")
    print(f"Max count: {counts.max()}")

if __name__ == "__main__":
    reproduction()
