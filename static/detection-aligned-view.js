// Plots of runs aligned at their own dip detection: ρ against van Trees restarted at detection and the posterior
// negentropy (Simulation page, static/article-view.js), and the lab's claimed rate and placement against the window (Lab
// page, static/lab-view.js). Data: /api/detection-aligned and /api/lab-detection-aligned
// (nvision/metrics/detection_aligned.py, nvision/metrics/lab_detection_aligned.py; every number is computed there).
// Needs ensurePlotly from plotly-utils.js.

const DA_REF_COLOR = '#94a3b8';   // reference lines (rho = 1, the Gaussian threshold), as the bound lines elsewhere

const daFmt = (v, digits = 3) => (v == null || !Number.isFinite(v) ? '–' : Number(v).toPrecision(digits));
const daEsc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const daShort = (name) => name.replace('SimpleSweep-optimal-x', 'Sweep k*×').replace('SimpleSweep-optimal', 'Sweep k*');
const daCell = 'padding:3px 10px; text-align:right; font-family:ui-monospace,monospace;';
const daTable = (head, rows, note) => `
    <table style="border-collapse:collapse; font-size:0.85em; background:#fff;">
        <thead><tr style="color:#475569; text-align:right;">${head.map((h, i) => `<th style="padding:3px 10px; ${i ? '' : 'text-align:left;'}">${h}</th>`).join('')}</tr></thead>
        <tbody>${rows.join('')}</tbody></table>
    ${note ? `<p style="margin:0.4em 0 0; font-size:0.8em; color:#64748b;">${note}</p>` : ''}`;

function daLayout(xTitle, yTitle, xType, yType) {
    return {
        template: 'plotly_white',
        margin: { t: 30, b: 55, l: 70, r: 15 },
        hovermode: 'closest',
        legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1, font: { size: 10 } },
        xaxis: { title: xTitle, type: xType, gridcolor: '#f1f5f9' },
        yaxis: { title: yTitle, type: yType, gridcolor: '#f1f5f9' },
    };
}

const daHLine = (y, dash = 'dot') => ({
    type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: y, y1: y, line: { color: DA_REF_COLOR, width: 1.5, dash },
});

// The lag axis is logarithmic, so lag 0 (the detection itself) is left out of the plots and listed in the table.
const daPositive = (lags, ys) => {
    const x = [], y = [];
    lags.forEach((k, i) => { if (k > 0 && ys[i] != null) { x.push(k); y.push(ys[i]); } });
    return { x, y };
};

// ── Simulation ──────────────────────────────────────────────────────────────────────────────────────────

function daPlotRho(div, locs, colorOf) {
    const traces = [];
    for (const [name, loc] of locs) {
        if (loc.rho_j_tau) {
            traces.push({ ...daPositive(loc.lags, loc.rho_j_tau), type: 'scatter', mode: 'lines', name: `${daShort(name)} · J_τ (bound)`,
                line: { color: colorOf(name), width: 2.5 }, legendgroup: name });
        }
        if (loc.rho_j_d) {
            traces.push({ ...daPositive(loc.lags, loc.rho_j_d), type: 'scatter', mode: 'lines', name: `${daShort(name)} · J_d (model)`,
                line: { color: colorOf(name), width: 1.5, dash: 'dot' }, legendgroup: name });
        }
    }
    const layout = daLayout('measurements since the run\'s own detection, k', 'ρ(k) = mean bound / mean squared error', 'log', 'log');
    layout.shapes = [daHLine(1, 'dash')];
    Plotly.newPlot(div, traces, layout, { responsive: true });
}

function daPlotNegentropy(div, locs, colorOf, threshold) {
    const traces = [];
    for (const [name, loc] of locs) {
        const q = loc.gaussian.negentropy_by_lag;
        const c = colorOf(name);
        const hi = daPositive(loc.lags, q.q75), lo = daPositive(loc.lags, q.q25);
        traces.push(
            { x: hi.x, y: hi.y, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, legendgroup: name },
            { x: lo.x, y: lo.y, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: `${c}26`, hoverinfo: 'skip', showlegend: false, legendgroup: name },
            { ...daPositive(loc.lags, q.median), mode: 'lines', name: daShort(name), line: { color: c, width: 2.5 }, legendgroup: name },
        );
        const onset = loc.gaussian.onset_lag_median;
        if (onset != null && onset > 0) {
            traces.push({ x: [onset], y: [threshold], mode: 'markers', name: `${daShort(name)} onset (median)`,
                marker: { symbol: 'diamond', size: 11, color: c }, legendgroup: name, showlegend: false,
                hovertemplate: `${daShort(name)}: Gaussian from k = ${onset} (median)<extra></extra>` });
        }
    }
    const layout = daLayout('measurements since the run\'s own detection, k', 'posterior negentropy (nats)', 'log', 'log');
    layout.shapes = [daHLine(threshold)];
    Plotly.newPlot(div, traces, layout, { responsive: true });
}

// The information rate a sweep over W gets per measurement, in units of I_max (Eq. sweep-info): 8πΓ / (27 W).
const daSweepQ = (fwhm, window) => (8 * Math.PI * fwhm) / (27 * window);

function daPlotLabRates(div, wins) {
    const ok = wins.filter((w) => w.claimed_rate && w.claimed_rate.q_hat != null);
    const placed = wins.filter((w) => w.placement && w.placement.efficiency_median != null);
    const xs = wins.map((w) => w.window_mhz);
    const lo = Math.min(...xs), hi = Math.max(...xs);
    const grid = Array.from({ length: 40 }, (_, i) => lo * (hi / lo) ** (i / 39));
    const fwhm = wins[0].model_fwhm_mhz;
    const traces = [
        { x: ok.map((w) => w.window_mhz), y: ok.map((w) => w.claimed_rate.q_hat), mode: 'lines+markers', name: 'claimed rate q̂ after detection',
            line: { color: '#2563eb', width: 2.5 }, marker: { size: 9 } },
        { x: placed.map((w) => w.window_mhz), y: placed.map((w) => w.placement.efficiency_median), mode: 'lines+markers', name: 'placement efficiency ⟨I(d)/I_max⟩',
            line: { color: '#e05c00', width: 2.5 }, marker: { size: 9, symbol: 'square' } },
        { x: grid, y: grid.map((w) => Math.min(1, daSweepQ(fwhm, w))), mode: 'lines', name: `a sweep over W: 8πΓ/(27W), Γ = ${daFmt(fwhm, 3)} MHz`,
            line: { color: DA_REF_COLOR, width: 2, dash: 'dash' } },
    ];
    const layout = daLayout('drive_freq window W (MHz)', 'information per measurement / I_max', 'log', 'log');
    layout.shapes = [daHLine(1)];
    Plotly.newPlot(div, traces, layout, { responsive: true });
}

function daLabTable(wins) {
    const rows = wins.map((w) => {
        const p = w.placement || {};
        const last = w.lags ? String(w.lags[w.lags.length - 1]) : null;
        const tail = (obj) => (obj && last != null ? obj[last] ?? Object.values(obj).at(-1) : null);
        return `<tr>
            <td style="padding:3px 10px; font-weight:600; text-align:left;">${w.window_mhz}</td>
            <td style="${daCell}">${w.n_detected}/${w.n_runs}</td>
            <td style="${daCell}">${daFmt(w.model_fwhm_mhz, 3)}</td>
            <td style="${daCell}">${daFmt(w.snr, 3)}</td>
            <td style="${daCell}">${daFmt(w.i_max_per_mhz2, 3)}</td>
            <td style="${daCell}">${w.detected_step_median == null ? '–' : `${daFmt(w.detected_step_median, 3)} (${daFmt(w.detected_step_q25, 3)}–${daFmt(w.detected_step_q75, 3)})`}</td>
            <td style="${daCell}">${w.max_lag ?? '–'}</td>
            <td style="${daCell}">${daFmt(w.claimed_rate && w.claimed_rate.q_hat, 2)}</td>
            <td style="${daCell}">${daFmt(p.efficiency_median, 2)}</td>
            <td style="${daCell}">${p.far_share_median == null ? '–' : `${Math.round(100 * p.far_share_median)} %`}</td>
            <td style="${daCell}">${daFmt(daSweepQ(w.model_fwhm_mhz, w.window_mhz), 2)}</td>
            <td style="${daCell}">${daFmt(tail(w.rmse_at), 2)}</td>
            <td style="${daCell}">${daFmt(tail(w.rms_claimed_at), 2)}</td>
            <td style="padding:3px 10px; font-size:0.9em; color:#92400e; text-align:left;">${daEsc(w.unavailable || w.j_d_unavailable || '')}</td>
        </tr>`;
    });
    return daTable(
        ['W (MHz)', 'detected', 'model Γ (MHz)', 'SNR', 'I_max (MHz⁻²)', 'detection step, median (IQR)', 'common lags', 'q̂',
            'placement', '> 3 HWHM off', 'sweep 8πΓ/27W', 'rms error at the last lag (MHz)', 'rms claimed σ there (MHz)', 'note'],
        rows,
        'Detection is each run\'s first steep, lasting drop of its claimed σ. q̂ is the slope of 1/⟨σ²⟩ per measurement in units of I_max over the second half of the lags. Placement is ⟨I(d)/I_max⟩ of the measurements after detection, about each run\'s own final estimate. The error is against the lab reference (the median final mean of the converged runs), so it cannot fall below the drift of the line between recordings (0.4–0.6 MHz).',
    );
}

