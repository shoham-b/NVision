// Lab view: calibration of the lab's own MATLAB Bayesian ESR runs, from /api/lab-study
// (served only when `nv serve --lab-dir` was given). Called by app.js showScanViewPanels('lab').
// Needs ensurePlotly from plotly-utils.js.
// Names of the labs the server found (empty without `nv serve --lab-dir`); app.js lists them as Studies.
window.NVISION_LABS = [];
window.labsReady = fetch(`${window.NVISION_ASSET_PREFIX || ''}api/lab-studies`, { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : []))
    .then((names) => { window.NVISION_LABS = names; })
    .catch(() => {});

async function renderLabStudy(container, name) {
    const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? '–' : Number(x).toFixed(d));
    const body = container.querySelector('#lab-body');
    let data;
    try {
        const resp = await fetch(`${window.NVISION_ASSET_PREFIX || ''}api/lab-study?name=${encodeURIComponent(name)}`, { cache: 'no-store' });
        if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);
        data = await resp.json();
    } catch (err) {
        body.innerHTML = `<p style="color:#64748b;">Lab "${name}" failed to load: ${err.message}.</p>`;
        return;
    }
    const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const colors = ['#1e90ff', '#e05c00', '#00a878'];
    const cols = [
        ['Window (MHz)', (r) => r.window_mhz],
        ['Converged', (r) => `${r.n_converged}/${r.n_runs}`],
        ['Steps, median (IQR)', (r) => `${fmt(r.steps_median, 0)} (${fmt(r.steps_q25, 0)}–${fmt(r.steps_q75, 0)})`],
        ['Claimed σ (MHz)', (r) => fmt(r.claimed_sigma_mhz)],
        ['Across-run std (MHz)', (r) => fmt(r.across_run_std_mhz)],
        ['Spread ÷ claimed σ', (r) => fmt(r.overconfidence)],
        ['Within 1σ (nominal 0.68)', (r) => fmt(r.within_1sigma)],
        ['Within 2σ (nominal 0.95)', (r) => fmt(r.within_2sigma)],
        ['95% interval hit', (r) => fmt(r.ci95_hit)],
    ];
    const cell = 'border:1px solid #e2e8f0; padding:0.35em 0.6em; text-align:right;';
    body.innerHTML = `
        <p style="margin:0 0 0.5em; color:#475569; font-size:0.85em;">Lab: <b>${name}</b> · reference resonance ${fmt(data.reference_mhz, 2)} MHz (median of the converged runs' final means).</p>
        <table style="border-collapse:collapse; width:100%; font-size:0.9em; background:#fff;">
            <thead><tr>${cols.map((c) => `<th style="${cell} background:#f8fafc; color:#475569;">${c[0]}</th>`).join('')}</tr></thead>
            <tbody>${data.windows.map((r) => `<tr>${cols.map((c) => `<td style="${cell} font-family:ui-monospace,monospace;">${c[1](r)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table>
        <p style="color:#475569; font-size:0.85em;">No ground truth exists for real data: error is measured against the reference, so the across-run std also contains any real drift of the resonance between runs. Read "Spread ÷ claimed σ" as an upper bound on miscalibration: the runs disagree with each other by that many times the σ each one claims.</p>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:1em;">
            <div><h3 style="margin:0 0 0.3em; font-size:1em;">Claimed σ vs across-run spread <span tabindex="0" class="help-icon-visible" title="Solid: median self-reported posterior std (band: quartiles). Dashed: how far the runs' posterior means scatter from each other at that step (std across runs; not an error against a known truth). Dashed above solid means the runs disagree more than they claim. Both include only runs still measuring at that step.">?</span></h3><div id="lab-sigma" style="height:420px"></div></div>
            <div><h3 style="margin:0 0 0.3em; font-size:1em;">Steps to convergence <span tabindex="0" class="help-icon-visible" title="Median number of measurements until the run's own stop rule fired, with the interquartile range, per drive_freq window.">?</span></h3><div id="lab-steps" style="height:420px"></div></div>
        </div>`;
    await ensurePlotly();
    const traces = [];
    data.windows.forEach((r, i) => {
        const c = data.curves[String(r.window_mhz)];
        const color = colors[i % colors.length];
        const name = `${r.window_mhz} MHz`;
        traces.push({ x: c.step, y: c.claimed_q75, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, legendgroup: name });
        traces.push({ x: c.step, y: c.claimed_q25, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: color + '22', hoverinfo: 'skip', showlegend: false, legendgroup: name });
        traces.push({ x: c.step, y: c.claimed_median, mode: 'lines', line: { color, width: 2 }, name: `${name} claimed`, legendgroup: name });
        traces.push({ x: c.actual_step, y: c.across_run_std, mode: 'lines', line: { color, width: 2, dash: 'dash' }, name: `${name} spread`, legendgroup: name });
    });
    Plotly.newPlot('lab-sigma', traces, {
        margin: { t: 10, r: 10, b: 50, l: 60 }, hovermode: 'x unified',
        xaxis: { title: 'measurement step', type: 'log', gridcolor: '#eeeeee' },
        yaxis: { title: 'σ (MHz)', type: 'log', gridcolor: '#eeeeee' },
        legend: { orientation: 'h', y: -0.25 },
    }, { responsive: true });
    const w = data.windows;
    Plotly.newPlot('lab-steps', [{
        x: w.map((r) => `${r.window_mhz} MHz`), y: w.map((r) => r.steps_median), type: 'bar', marker: { color: colors[0] },
        error_y: { type: 'data', symmetric: false, array: w.map((r) => r.steps_q75 - r.steps_median), arrayminus: w.map((r) => r.steps_median - r.steps_q25) },
    }], {
        margin: { t: 10, r: 10, b: 50, l: 60 },
        xaxis: { title: 'drive_freq window' }, yaxis: { title: 'measurements to converge', gridcolor: '#eeeeee' },
    }, { responsive: true });
}
