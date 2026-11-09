// Lab view: calibration of the lab's own MATLAB Bayesian ESR runs, from /api/lab-study
// (served only when `nv serve --lab-dir` was given). Called by app.js showScanViewPanels('lab').
// Needs ensurePlotly from plotly-utils.js.
// Names of the labs the server found (empty without `nv serve --lab-dir`); app.js lists them as Studies.
window.NVISION_LABS = [];
window.labsReady = fetch(`${window.NVISION_ASSET_PREFIX || ''}api/lab-studies`, { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : []))
    .then((names) => { window.NVISION_LABS = names; })
    .catch(() => {});

// The study name the server lists once raw sweeps were processed (nvision.metrics.lab_reference.SWEEPS_LAB_NAME).
const LAB_SWEEPS_NAME = 'raw sweeps (SBED)';
const labApi = (path) => `${window.NVISION_ASSET_PREFIX || ''}api/${path}`;

// Sync bar shown on every lab page: the sources added with `nv matlab`, the last background sync
// (`nv serve` starts one on launch), its failures, and a "Sync now" button.
async function renderLabSync(bar) {
    let state;
    try {
        const resp = await fetch(labApi('lab-sync'), { cache: 'no-store' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        state = await resp.json();
    } catch (err) {
        bar.innerHTML = '';
        return;
    }
    if (!state.sources.length) {
        bar.innerHTML = '<p style="margin:0 0 0.8em; color:#475569; font-size:0.85em;">No lab source added yet: run <code>nv matlab &lt;Drive folder link&gt;</code> to add and sync it; run <code>nv matlab</code> again for later changes.</p>';
        return;
    }
    const st = state.status;
    const when = (iso) => (iso ? new Date(iso).toLocaleString() : '–');
    let line;
    if (state.running) {
        line = '<b>Syncing…</b> new and changed files are being processed.';
    } else if (!st) {
        line = 'Not synced yet.';
    } else {
        const failed = st.sources.flatMap((s) => [...(s.error ? [`${s.label}: ${s.error}`] : []), ...s.failures.map((f) => `${s.label}/${f}`)]);
        const outcome = st.state === 'done' ? 'all files processed'
            : st.state === 'interrupted' ? '<span style="color:#b91c1c;">interrupted before it finished (press Sync now)</span>'
            : `<span style="color:#b91c1c;">${failed.length} problem(s)</span>`;
        line = `Last sync ${when(st.finished || st.started)} · ${outcome}`;
        if (failed.length) line += `<ul style="margin:0.3em 0 0; color:#b91c1c;">${failed.map((f) => `<li style="font-family:ui-monospace,monospace;">${f}</li>`).join('')}</ul>`;
    }
    bar.innerHTML = `
        <div style="display:flex; flex-wrap:wrap; gap:0.6em; align-items:baseline; margin:0 0 0.8em; color:#475569; font-size:0.85em;">
            <span>Sources: <b style="font-family:ui-monospace,monospace;">${state.sources.join(', ')}</b> · ${line}</span>
            <button class="link-button" id="lab-sync-now" ${state.running ? 'disabled' : ''}>Sync now</button>
            <span id="lab-sync-done"></span>
        </div>`;
    bar.querySelector('#lab-sync-now').onclick = async () => {
        await fetch(labApi('lab-sync'), { method: 'POST' });
        renderLabSync(bar);
    };
    if (state.running) {
        bar.dataset.wasRunning = '1';
        setTimeout(() => renderLabSync(bar), 3000);
    } else if (bar.dataset.wasRunning) {
        // The sync just finished: the generator lists and studies were read at page load.
        bar.querySelector('#lab-sync-done').innerHTML = 'Sync finished. <button class="link-button" onclick="location.reload()">Reload page</button> to see new runs.';
    }
}


// sigma(n) = 1/sqrt(J0 + I*max(n - n0, 0)): J0 = 1/variance of the initial window (NOT fitted; a uniform window of width W has
// variance W^2/12, so J0 = 12/W^2 in 1/MHz^2), I = information per measurement (1/MHz^2) once the search is over, n0 = step
// at which information starts to accumulate.
const windowJ0 = (windowMhz) => 12 / (windowMhz * windowMhz);
const infoSigma = (n, f) => 1 / Math.sqrt(f.J0 + f.I * Math.max(n - f.n0, 0));

// Fit I and n0 of infoSigma to a curve (steps n >= 1, sigma > 0) with J0 fixed: grid over n0, golden-section search on log I for
// each, minimising sum (log sigma_k - log infoSigma(n_k))^2 with every step weighted equally.
function fitInfoCurve(steps, sigma, J0) {
    const stride = Math.max(1, Math.ceil(steps.length / 100));  // uniform subsample keeps the equal weight per step
    const pts = steps.map((n, k) => [n, sigma[k]]).filter(([n, s], k) => n > 0 && s > 0 && k % stride === 0);
    const last = pts[pts.length - 1][0];
    let overall = null;
    for (let n0 = 0; n0 <= last / 2; n0 += Math.max(1, Math.round(last / 60))) {
        const cost = (li) => pts.reduce((acc, [n, s]) => acc + (Math.log(s) + 0.5 * Math.log(J0 + Math.exp(li) * Math.max(n - n0, 0))) ** 2, 0);
        let lo = Math.log(J0) - 20, hi = Math.log(J0) + 10;
        const g = (Math.sqrt(5) - 1) / 2;
        for (let it = 0; it < 80; it++) {
            const m1 = hi - g * (hi - lo), m2 = lo + g * (hi - lo);
            if (cost(m1) < cost(m2)) hi = m2; else lo = m1;
        }
        const li = (lo + hi) / 2, f = cost(li);
        if (overall === null || f < overall.f) overall = { f, J0, I: Math.exp(li), n0 };
    }
    return overall;
}

async function renderLabStudy(container, name) {
    const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? '–' : Number(x).toFixed(d));
    let bar = container.querySelector('#lab-sync');
    if (!bar) {
        bar = document.createElement('div');
        bar.id = 'lab-sync';
        container.querySelector('#lab-body').before(bar);
    }
    renderLabSync(bar);
    const body = container.querySelector('#lab-body');
    if (name === LAB_SWEEPS_NAME) return renderLabSweeps(body, fmt);
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
    // Focus-window check (from the runs' recorded focus candidates): found, covering the resonance, how early.
    const pct = (x) => `${Math.round(100 * x)}%`;
    const step = (x) => (x === null ? '–' : Math.round(x));
    const focusCols = [
        ['Window (MHz)', (r) => r.window_mhz],
        ['Runs', (r) => r.n_runs],
        ['Window found', (r) => pct(r.found_rate)],
        ['Found at step', (r) => step(r.found_step)],
        ['Covers resonance', (r) => pct(r.cover_rate)],
        ['… from step', (r) => step(r.cover_step)],
        ['Covers to the end', (r) => pct(r.stable_rate)],
        ['… from step', (r) => step(r.stable_step)],
        ['Median steps to converge', (r) => step(data.windows.find((w) => w.window_mhz === r.window_mhz)?.steps_median ?? null)],
        ['Final window locked', (r) => pct(r.locked_rate)],
        ['Final window width (MHz)', (r) => fmt(r.final_width_mhz, 1)],
    ];
    const focusTable = data.focus && data.focus.length ? `
        <h3 style="margin:1em 0 0.3em; font-size:1em;">Focus window <span tabindex="0" class="help-icon-visible" title="From the focus candidates the runs recorded at every step. 'Found': at least one candidate window exists (not before the minimum focus steps). 'Covers resonance': some candidate contains the reference resonance; 'to the end': it does at every later step too. Compare these steps with the median steps to converge: a window found after the run has converged was no help. 'Locked': the run narrowed to a single final window; otherwise the final window is the whole scan range.">?</span></h3>
        <table style="border-collapse:collapse; width:100%; font-size:0.9em; background:#fff; margin-bottom:1em;">
            <thead><tr>${focusCols.map((c) => `<th style="${cell} background:#f8fafc; color:#475569;">${c[0]}</th>`).join('')}</tr></thead>
            <tbody>${data.focus.map((r) => `<tr>${focusCols.map((c) => `<td style="${cell} font-family:ui-monospace,monospace;">${c[1](r)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table>` : '';
    body.innerHTML = `
        <p style="margin:0 0 0.5em; color:#475569; font-size:0.85em;">Lab: <b>${name}</b> · reference resonance ${fmt(data.reference_mhz, 2)} MHz (median of the converged runs' final means).</p>
        <table style="border-collapse:collapse; width:100%; font-size:0.9em; background:#fff;">
            <thead><tr>${cols.map((c) => `<th style="${cell} background:#f8fafc; color:#475569;">${c[0]}</th>`).join('')}</tr></thead>
            <tbody>${data.windows.map((r) => `<tr>${cols.map((c) => `<td style="${cell} font-family:ui-monospace,monospace;">${c[1](r)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table>
        <p style="color:#475569; font-size:0.85em;">No ground truth exists for real data: error is measured against the reference, so the across-run std also contains any real drift of the resonance between runs. Read "Spread ÷ claimed σ" as an upper bound on miscalibration: the runs disagree with each other by that many times the σ each one claims.</p>
        ${focusTable}
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:1em;">
            <div>${labPlotHead('Claimed σ vs across-run spread', "Solid: median self-reported posterior std (band: quartiles). Dashed: how far the runs' posterior means scatter from each other at that step (std across runs; not an error against a known truth). Dashed above solid means the runs disagree more than they claim. Both include only runs still measuring at that step.", 'lab-spread')}<div id="lab-spread" style="height:420px"></div></div>
            <div>${labPlotHead('Claimed σ vs fit', 'Solid: median self-reported posterior std. Dotted: whole-curve fit σ = 1/√(J0 + I·(n − n0)) with J0 = 12/W² fixed from the initial window of width W (not fitted), I the information per measurement and n0 the step where information starts accruing.', 'lab-fit')}<div id="lab-fit" style="height:420px"></div></div>
            <div><h3 style="margin:0 0 0.3em; font-size:1em;">Steps to convergence <span tabindex="0" class="help-icon-visible" title="Median number of measurements until the run's own stop rule fired, with the interquartile range, per drive_freq window.">?</span></h3><div id="lab-steps" style="height:420px"></div></div>
        </div>`;
    await ensurePlotly();
    // Every legend entry gets its own colour (window x series), so no two lines share a hue.
    const palette = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#db2777', '#0891b2', '#65a30d', '#475569'];
    let ci = 0;
    const next = () => palette[ci++ % palette.length];
    const spreadTraces = [], fitTraces = [];
    data.windows.forEach((r) => {
        const c = data.curves[String(r.window_mhz)];
        const name = `${r.window_mhz} MHz`;
        const claimedColor = next();
        const spreadColor = claimedColor, fitColor = claimedColor; // one colour per window; solid = claimed, dashed = spread, dotted = fit
        const band = (group) => [
            { x: c.step, y: c.claimed_q75, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, legendgroup: group },
            { x: c.step, y: c.claimed_q25, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: claimedColor + '26', hoverinfo: 'skip', showlegend: false, legendgroup: group },
        ];
        // Whole-curve fit of the claimed median to sigma = 1/sqrt(J0 + I*max(n - n0, 0)) (see infoSigma): J0 = 12/W^2 fixed from the initial window (not fitted), I = information
        // per measurement (1/MHz^2). A scale a in a/sqrt(...) would be degenerate with J0 and I, so a = 1. Least squares in
        // log sigma, every step weighted equally so the end of the curve is matched.
        const fit = fitInfoCurve(c.step, c.claimed_median, windowJ0(r.window_mhz));
        const claimed = (group) => ({ x: c.step, y: c.claimed_median, mode: 'lines', line: { color: claimedColor, width: 2.5 }, name: `${name} claimed σ`, legendgroup: group });
        spreadTraces.push(...band(name + 's'), claimed(name + 's'),
            { x: c.actual_step, y: c.across_run_std, mode: 'lines', line: { color: spreadColor, width: 2.5, dash: 'dash' }, name: `${name} across-run spread`, legendgroup: name + 's' });
        fitTraces.push(...band(name + 'f'), claimed(name + 'f'),
            { x: c.step, y: c.step.map((n) => infoSigma(n, fit)), mode: 'lines', line: { color: fitColor, width: 3, dash: 'dot' }, name: `${name} fit (I=${fit.I.toPrecision(2)}/meas, n0=${fit.n0})`, legendgroup: name + 'f' });
    });
    // The legend sits under the axis title and needs room for every entry: size the bottom margin and the plot height from the
    // number of legend rows (spread: two entries per row, fit: one long entry per row), so it never overlaps the plot or the axis title.
    const sigmaPlot = (id, traces, perRow) => {
        const rows = Math.ceil(traces.filter((t) => t.showlegend !== false).length / perRow);
        const bottom = 70 + 24 * rows, plotH = 330;
        const el = document.getElementById(id);
        el.dataset.rows = rows;
        el.style.height = `${plotH + bottom + 10}px`;
        Plotly.newPlot(id, traces, {
            margin: { t: 10, r: 10, b: bottom, l: 60 }, hovermode: 'x unified', font: { size: 13 },
            xaxis: { title: { text: 'measurement step', standoff: 8 }, type: 'log', gridcolor: '#eeeeee' },
            yaxis: { title: 'σ (MHz)', type: 'log', gridcolor: '#eeeeee' },
            legend: { orientation: 'h', yanchor: 'top', y: -(60 / plotH) },
        }, { responsive: true, displaylogo: false });
    };
    sigmaPlot('lab-spread', spreadTraces, 2);
    sigmaPlot('lab-fit', fitTraces, 1);
    wireLabDownloads('lab-spread', 'claimed-sigma-vs-spread', 'Claimed σ vs across-run spread');
    wireLabDownloads('lab-fit', 'claimed-sigma-vs-fit', 'Claimed σ vs fit');
    const w = data.windows;
    Plotly.newPlot('lab-steps', [{
        x: w.map((r) => `${r.window_mhz} MHz`), y: w.map((r) => r.steps_median), type: 'bar', marker: { color: colors[0] },
        error_y: { type: 'data', symmetric: false, array: w.map((r) => r.steps_q75 - r.steps_median), arrayminus: w.map((r) => r.steps_median - r.steps_q25) },
    }], {
        margin: { t: 10, r: 10, b: 50, l: 60 },
        xaxis: { title: 'drive_freq window' }, yaxis: { title: 'measurements to converge', gridcolor: '#eeeeee' },
    }, { responsive: true });
}

// Heading + help icon + PNG/SVG download buttons for one lab plot (wired by wireLabDownloads).
function labPlotHead(title, help, id) {
    const btn = 'font-size:0.8em; padding:0.15em 0.7em; margin-left:0.4em; border:1px solid #cbd5e1; border-radius:999px; background:#fff; color:#334155; cursor:pointer;';
    return `<h3 style="margin:0 0 0.3em; font-size:1em; display:flex; align-items:center; gap:0.3em;">${title} <span tabindex="0" class="help-icon-visible" title="${help.replace(/"/g, '&quot;')}">?</span>
        <span style="margin-left:auto; font-weight:normal;"><button type="button" data-dl="${id}" data-fmt="png" style="${btn}">Download PNG</button><button type="button" data-dl="${id}" data-fmt="svg" style="${btn}">SVG</button></span></h3>`;
}

// Downloads the plot as a clean, titled, white-background figure. The export is smaller (1100 px wide, rendered at 3x) with bigger
// fonts, so text is large relative to the graph; margins and legend position are rescaled for the bigger text, then restored.
function wireLabDownloads(id, filename, title) {
    document.querySelectorAll(`button[data-dl="${id}"]`).forEach((b) => {
        b.onclick = async () => {
            const gd = document.getElementById(id);
            const k = 20 / gd.layout.font.size;  // text scale-up for the export
            const rows = Number(gd.dataset.rows), plotH = 380, bottom = Math.round(k * (70 + 24 * rows)), top = 70;
            const orig = { 'font.size': gd.layout.font.size, 'margin.t': gd.layout.margin.t, 'margin.b': gd.layout.margin.b, 'margin.l': gd.layout.margin.l, 'legend.y': gd.layout.legend.y, 'title.text': '', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' };
            await Plotly.relayout(gd, { 'font.size': 20, 'title.text': title, 'title.font.size': 26, 'title.x': 0.02, 'margin.t': top, 'margin.b': bottom, 'margin.l': 80, 'legend.y': -(k * 60 / plotH), paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff' });
            try {
                await Plotly.downloadImage(gd, { format: b.dataset.fmt, filename, width: 1100, height: top + plotH + bottom, scale: 3 });
            } finally {
                await Plotly.relayout(gd, orig);
            }
        };
    });
}

// One Plotly graph of claimed σ (MHz) against measurement step for the files picked in #lab-sigma-files, for the
// parameter picked in the radio group. Per file: SBED on all sweeps (solid), SBED on the even sweeps (dashed), and the
// odd-sweep fit's σ and the lab-averaged-sweep fit's σ as horizontal lines. Legend clicks toggle each line.
function renderLabSigmaGraph(files) {
    const picker = document.getElementById('lab-sigma-files');
    const colors = ['#1e90ff', '#e05c00', '#00a878', '#8b5cf6', '#d946ef', '#0891b2', '#ca8a04', '#be123c'];
    const draw = () => {
        const param = document.querySelector('input[name="lab-sigma-param"]:checked').value;
        const picked = [...picker.selectedOptions].map((o) => files[Number(o.value)]);
        const traces = [];
        picked.forEach((f, k) => {
            const c = f.curves, color = colors[k % colors.length], label = picked.length > 1 ? `${f.file} · ` : '';
            const hline = (sigma, name, dash) => sigma == null ? [] : [{
                x: [c.steps[0], c.steps[c.steps.length - 1]], y: [sigma, sigma], mode: 'lines', line: { color, width: 2, dash },
                name: `${label}${name} (${sigma.toPrecision(2)} MHz)`, legendgroup: f.id,
            }];
            traces.push(
                { x: c.steps, y: c[`all_${param}`], mode: 'lines', line: { color, width: 2 }, name: `${label}SBED, all sweeps`, legendgroup: f.id },
                { x: c.steps, y: c[`even_${param}`], mode: 'lines', line: { color, width: 2, dash: 'dash' }, name: `${label}SBED, even sweeps`, legendgroup: f.id },
                ...hline(c[`ref_${param}`], 'odd-sweep fit σ', 'dot'),
                ...hline(c[`lab_${param}`], 'lab averaged sweep fit σ', 'dashdot'),
            );
        });
        Plotly.react('lab-sweeps-sigma', traces, {
            margin: { t: 10, r: 10, b: 50, l: 60 }, hovermode: 'x unified',
            xaxis: { title: 'measurement step', type: 'log', gridcolor: '#eeeeee' },
            yaxis: { title: `claimed σ of ${param} (MHz)`, type: 'log', gridcolor: '#eeeeee' },
            legend: { orientation: 'h', y: -0.25 },
        }, { responsive: true });
    };
    picker.onchange = draw;
    document.querySelectorAll('input[name="lab-sigma-param"]').forEach((el) => { el.onchange = draw; });
    draw();
}

// Raw ESR sweeps run through SBED (from /api/lab-sweeps): per folder, SBED on all the data, and SBED on the even
// sweeps checked against a fit of the odd ones (z). See nvision/metrics/lab_reference.py.
async function renderLabSweeps(body, fmt) {
    let data;
    try {
        const resp = await fetch(labApi('lab-sweeps'), { cache: 'no-store' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        data = await resp.json();
    } catch (err) {
        body.innerHTML = `<p style="color:#64748b;">Raw sweeps failed to load: ${err.message}.</p>`;
        return;
    }
    const cell = 'border:1px solid #e2e8f0; padding:0.35em 0.6em; text-align:right;';
    const mono = 'font-family:ui-monospace,monospace;';
    const pm = (v, s, d = 3) => (v == null ? '–' : `${fmt(v, d)} ± ${fmt(s, d)}`);
    const zCell = (z, atBound) => (atBound ? '<span style="color:#b45309;" title="The reference fit sits on the prior bound of this parameter (for zeeman_shift: 0, the doublet is not resolved), so it has no usable σ_ref and no z is formed. Not counted in the shares.">ref on bound</span>'
        : z == null ? '–' : `<span style="${Math.abs(z) > 2 ? 'color:#b91c1c; font-weight:bold;' : ''}">${z >= 0 ? '+' : ''}${fmt(z)}</span>`);
    const cols = [
        ['File', 'Recording file name.', (r) => `<span style="${mono} text-align:left;">${r.file}</span>`],
        ['Sweeps', 'Complete passes of the instrument over every drive_freq point recorded in the file (shot columns).', (r) => r.n_sweeps],
        ['drive_freq window (MHz)', 'Width of the recorded drive_freq range.', (r) => fmt(r.window_mhz, 0)],
        ['Steps', 'SBED measurements on all the data (budget 300).', (r) => r.steps],
        ['Converged at', 'Step from which the claimed σ of zeeman_shift stayed under the stop ceiling; – if it never did.', (r) => r.converged_step ?? '–'],
        ['zeeman_shift, all data (MHz)', 'SBED on every sweep: estimate ± claimed σ.', (r) => pm(r.zeeman_mhz, r.zeeman_sigma_mhz)],
        ['zeeman_shift, even sweeps (MHz)', 'SBED given only the even sweeps: estimate ± claimed σ. This is the run the z checks.', (r) => pm(r.holdout_zeeman_mhz, r.holdout_zeeman_sigma_mhz)],
        ['Reference, odd sweeps (MHz)', 'Least-squares fit of the odd sweeps, which SBED never saw, with the same model and bounds: value ± σ_ref.', (r) => (r.ref_zeeman_at_bound ? `${fmt(r.ref_zeeman_mhz, 3)} (on bound)` : pm(r.ref_zeeman_mhz, r.ref_zeeman_sigma_mhz))],
        ['z zeeman_shift', '(even-sweep estimate − reference) / √(σ² + σ_ref²). Honest σ: |z| ≤ 1 for about 68% of files, ≤ 2 for 95%. Bold red: |z| > 2.', (r) => zCell(r.z_zeeman, r.ref_zeeman_at_bound)],
        ['z zero_field_freq', 'The same check for the dip centre zero_field_freq.', (r) => zCell(r.z_center, r.ref_center_at_bound)],
        ['Reference χ²/dof', `Goodness of the reference fit. Above ${data.chi2_dof_warn} the Lorentzian model does not describe the data, so σ_ref is too small and |z| too large.`, (r) => `<span style="${r.ref_chi2_dof > data.chi2_dof_warn ? 'color:#b45309; font-weight:bold;' : ''}">${fmt(r.ref_chi2_dof)}</span>`],
        ['Shot reuse', 'Draws per distinct recorded shot in the even-sweep run. SBED draws shots with replacement and counts each draw as new evidence, so well above 1 makes its σ too small.', (r) => fmt(r.shot_reuse, 1)],
    ];
    const head = cols.map((c) => `<th style="${cell} background:#f8fafc; color:#475569;">${c[0]} <span tabindex="0" class="help-icon-visible" title="${c[1]}">?</span></th>`).join('');
    const share = (c) => (c.n ? `${Math.round(c.within_1 * c.n)}/${c.n} within 1σ (nominal 68%) · ${Math.round(c.within_2 * c.n)}/${c.n} within 2σ (95%) · median |z| ${fmt(c.median_abs)}` : 'no z');
    const groups = data.groups.map((g, i) => `
        <h3 style="margin:1.2em 0 0.3em; font-size:1em;"><span style="${mono}">${g.folder}</span> <span style="color:#64748b; font-weight:normal;">· source ${g.label} · ${g.rows.length} file(s)</span></h3>
        <p style="margin:0 0 0.4em; color:#475569; font-size:0.85em;">zeeman_shift: ${share(g.coverage.z_zeeman)}<br>zero_field_freq: ${share(g.coverage.z_center)}</p>
        <div style="overflow-x:auto;"><table style="border-collapse:collapse; width:100%; font-size:0.85em; background:#fff;">
            <thead><tr>${head}</tr></thead>
            <tbody>${g.rows.map((r) => `<tr>${cols.map((c) => `<td style="${cell} ${mono}">${c[2](r)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table></div>`).join('');
    // One entry per file, in table order, keyed for the file picker; files from old records have no sigma_curves.
    const sigmaFiles = data.groups.flatMap((g) => g.rows.map((r) => ({ id: `${g.label}/${g.folder}/${r.file}`, folder: g.folder, file: r.file, curves: r.sigma_curves })));
    const fileOptions = sigmaFiles.map((f, k) => `<option value="${k}" ${f.curves ? '' : 'disabled'} ${k === sigmaFiles.findIndex((x) => x.curves) ? 'selected' : ''}>${f.folder} / ${f.file}${f.curves ? '' : ' (sync again)'}</option>`).join('');
    const failed = data.failed.length
        ? `<h3 style="margin:1.2em 0 0.3em; font-size:1em; color:#b91c1c;">Failed files (${data.failed.length})</h3><ul style="color:#b91c1c; font-size:0.85em;">${data.failed.map((f) => `<li><span style="${mono}">${f.label}/${f.file}</span>: ${f.error}</li>`).join('')}</ul>`
        : '';
    body.innerHTML = `
        <p style="margin:0 0 0.5em; color:#475569; font-size:0.85em;">Raw ESR recordings run through SBED. Real data has no ground truth, so each file is split by sweep: SBED gets the even sweeps and the odd ones are fitted as an independent reference. One file gives one independent z, so the shares are over the files of one folder (one experiment), never pooled across folders.</p>
        <div><h3 style="margin:0 0 0.3em; font-size:1em;">z of zeeman_shift per file <span tabindex="0" class="help-icon-visible" title="Each point is one file: (SBED estimate on the even sweeps − reference fit of the odd sweeps) / combined σ. Shaded: |z| ≤ 1 and ≤ 2. If SBED's claimed σ is honest, about 68% of the points sit in the inner band and 95% in the outer one.">?</span></h3><div id="lab-sweeps-z" style="height:360px"></div></div>
        <div><h3 style="margin:1em 0 0.3em; font-size:1em;">Claimed σ vs measurement step <span tabindex="0" class="help-icon-visible" title="SBED's claimed σ after each measurement, against two yardsticks drawn as horizontal lines: the σ of a fit to the odd sweeps (never seen by SBED) and the σ of a fit to the lab's own averaged sweep (all shots; SBED draws from the same shots, so it is not independent). Click a legend entry to hide or show that line; double-click to isolate it. Select several files to overlay them.">?</span></h3>
            <div style="display:flex; flex-wrap:wrap; gap:1em; align-items:flex-start; margin:0 0 0.4em; font-size:0.85em; color:#475569;">
                <label>Files (Ctrl/Shift-click for several)<br><select id="lab-sigma-files" multiple size="6" style="min-width:22em; font-family:ui-monospace,monospace;">${fileOptions}</select></label>
                <div><div>Parameter</div>
                    <label><input type="radio" name="lab-sigma-param" value="zeeman_shift" checked> zeeman_shift</label><br>
                    <label><input type="radio" name="lab-sigma-param" value="zero_field_freq"> zero_field_freq</label></div>
            </div>
            <div id="lab-sweeps-sigma" style="height:420px"></div></div>
        ${groups}${failed}`;
    await ensurePlotly();
    renderLabSigmaGraph(sigmaFiles);
    const colors = ['#1e90ff', '#e05c00', '#00a878', '#8b5cf6', '#d946ef', '#0891b2'];
    const traces = data.groups.map((g, i) => ({
        x: g.rows.map(() => `${g.folder} (${g.label})`), y: g.rows.map((r) => r.z_zeeman), text: g.rows.map((r) => r.file),
        mode: 'markers', type: 'scatter', name: g.folder, marker: { color: colors[i % colors.length], size: 9 },
        hovertemplate: '%{text}<br>z = %{y:.2f}<extra></extra>',
    }));
    const band = (h, opacity) => ({ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: -h, y1: h, fillcolor: '#94a3b8', opacity, line: { width: 0 }, layer: 'below' });
    Plotly.newPlot('lab-sweeps-z', traces, {
        margin: { t: 10, r: 10, b: 60, l: 60 }, showlegend: false, shapes: [band(2, 0.12), band(1, 0.18)],
        xaxis: { title: 'folder (source)' }, yaxis: { title: 'z of zeeman_shift', zeroline: true, gridcolor: '#eeeeee' },
    }, { responsive: true });
}
