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
        bar.innerHTML = '<p style="margin:0 0 0.8em; color:#475569; font-size:0.85em;">No lab source added yet: run <code>nv matlab &lt;Drive folder link&gt;</code> once, then <code>nv serve</code> keeps it in sync.</p>';
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
    body.innerHTML = `
        <p style="margin:0 0 0.5em; color:#475569; font-size:0.85em;">Lab: <b>${name}</b> · reference resonance ${fmt(data.reference_mhz, 2)} MHz (median of the converged runs' final means).</p>
        <table style="border-collapse:collapse; width:100%; font-size:0.9em; background:#fff;">
            <thead><tr>${cols.map((c) => `<th style="${cell} background:#f8fafc; color:#475569;">${c[0]}</th>`).join('')}</tr></thead>
            <tbody>${data.windows.map((r) => `<tr>${cols.map((c) => `<td style="${cell} font-family:ui-monospace,monospace;">${c[1](r)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table>
        <p style="color:#475569; font-size:0.85em;">No ground truth exists for real data: error is measured against the reference, so the across-run std also contains any real drift of the resonance between runs. Read "Spread ÷ claimed σ" as an upper bound on miscalibration: the runs disagree with each other by that many times the σ each one claims.</p>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:1em;">
            <div><h3 style="margin:0 0 0.3em; font-size:1em;">Claimed σ vs across-run spread <span tabindex="0" class="help-icon-visible" title="Solid: median self-reported posterior std (band: quartiles). Dashed: how far the runs' posterior means scatter from each other at that step (std across runs; not an error against a known truth). Dashed above solid means the runs disagree more than they claim. Both include only runs still measuring at that step.">?</span> <label style="font-weight:normal; font-size:0.85em; margin-left:0.8em;"><input type="checkbox" id="lab-show-spread" checked> show spread (dashed)</label> <label style="font-weight:normal; font-size:0.85em; margin-left:0.8em;"><input type="checkbox" id="lab-show-guide" checked> −1/2 fit where reached (dotted)</label></h3><div id="lab-sigma" style="height:420px"></div></div>
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
        // Log-log fit of the claimed median over a step range [lo, last]; slope is -1/2 once the posterior is asymptotic.
        const last = c.step[c.step.length - 1];
        const fit = (lo) => {
            const ks = c.step.map((s, k) => k).filter((k) => c.step[k] >= lo && c.claimed_median[k] > 0);
            const lx = ks.map((k) => Math.log(c.step[k])), ly = ks.map((k) => Math.log(c.claimed_median[k]));
            const mx = lx.reduce((s, v) => s + v, 0) / lx.length, my = ly.reduce((s, v) => s + v, 0) / ly.length;
            const slope = lx.reduce((s, v, k) => s + (v - mx) * (ly[k] - my), 0) / lx.reduce((s, v) => s + (v - mx) ** 2, 0);
            return { slope, a: Math.exp(ly.reduce((s, v, k) => s + v + 0.5 * lx[k], 0) / lx.length) };  // a: sigma = a / sqrt(step)
        };
        // Longest end-anchored range (at least a factor 2.5 in steps) whose free slope is within 0.1 of -1/2.
        let lo = Math.abs(fit(last / 10).slope + 0.5) < 0.1 ? c.step.find((s) => s >= last / 10) : null;  // the last decade first
        for (let k = 0; lo === null && k < c.step.length; k++) {
            if (c.step[k] * 2.5 > last) break;
            // The second half of the range must also sit near -1/2, so a curve that has flattened again is not accepted.
            if (Math.abs(fit(c.step[k]).slope + 0.5) < 0.1 && Math.abs(fit(Math.sqrt(c.step[k] * last)).slope + 0.5) < 0.15) { lo = c.step[k]; break; }
        }
        const tailFit = fit(last / 10);
        const label = lo === null ? `slope ${tailFit.slope.toFixed(2)}, not at -1/2 yet` : `slope ${fit(lo).slope.toFixed(2)} from step ${lo}`;
        traces.push({ x: c.step, y: c.claimed_median, mode: 'lines', line: { color, width: 2 }, name: `${name} claimed (${label})`, legendgroup: name });
        if (lo !== null) {
            const f = fit(lo), xs = [lo, last];
            traces.push({ x: xs, y: xs.map((s) => f.a / Math.sqrt(s)), mode: 'lines', line: { color, width: 3, dash: 'dot' }, hoverinfo: 'skip', showlegend: false, legendgroup: name, meta: 'guide' });
        }
        traces.push({ x: c.actual_step, y: c.across_run_std, mode: 'lines', line: { color, width: 2, dash: 'dash' }, name: `${name} spread`, legendgroup: name, meta: 'spread' });
    });
    Plotly.newPlot('lab-sigma', traces, {
        margin: { t: 10, r: 10, b: 50, l: 60 }, hovermode: 'x unified',
        xaxis: { title: 'measurement step', type: 'log', gridcolor: '#eeeeee' },
        yaxis: { title: 'σ (MHz)', type: 'log', gridcolor: '#eeeeee' },
        legend: { orientation: 'h', y: -0.25 },
    }, { responsive: true });
    // Hiding the dashed traces lets the y axis autorange onto the claimed curves, so their late -1/2 slope is readable.
    document.getElementById('lab-show-spread').addEventListener('change', (e) => {
        const idx = traces.map((t, k) => (t.meta === 'spread' ? k : -1)).filter((k) => k >= 0);
        Plotly.restyle('lab-sigma', { visible: e.target.checked }, idx);
        Plotly.relayout('lab-sigma', { 'yaxis.autorange': true });
    });
    document.getElementById('lab-show-guide').addEventListener('change', (e) => {
        const idx = traces.map((t, k) => (t.meta === 'guide' ? k : -1)).filter((k) => k >= 0);
        Plotly.restyle('lab-sigma', { visible: e.target.checked }, idx);
    });
    const w = data.windows;
    Plotly.newPlot('lab-steps', [{
        x: w.map((r) => `${r.window_mhz} MHz`), y: w.map((r) => r.steps_median), type: 'bar', marker: { color: colors[0] },
        error_y: { type: 'data', symmetric: false, array: w.map((r) => r.steps_q75 - r.steps_median), arrayminus: w.map((r) => r.steps_median - r.steps_q25) },
    }], {
        margin: { t: 10, r: 10, b: 50, l: 60 },
        xaxis: { title: 'drive_freq window' }, yaxis: { title: 'measurements to converge', gridcolor: '#eeeeee' },
    }, { responsive: true });
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
    const failed = data.failed.length
        ? `<h3 style="margin:1.2em 0 0.3em; font-size:1em; color:#b91c1c;">Failed files (${data.failed.length})</h3><ul style="color:#b91c1c; font-size:0.85em;">${data.failed.map((f) => `<li><span style="${mono}">${f.label}/${f.file}</span>: ${f.error}</li>`).join('')}</ul>`
        : '';
    body.innerHTML = `
        <p style="margin:0 0 0.5em; color:#475569; font-size:0.85em;">Raw ESR recordings run through SBED. Real data has no ground truth, so each file is split by sweep: SBED gets the even sweeps and the odd ones are fitted as an independent reference. One file gives one independent z, so the shares are over the files of one folder (one experiment), never pooled across folders.</p>
        <div><h3 style="margin:0 0 0.3em; font-size:1em;">z of zeeman_shift per file <span tabindex="0" class="help-icon-visible" title="Each point is one file: (SBED estimate on the even sweeps − reference fit of the odd sweeps) / combined σ. Shaded: |z| ≤ 1 and ≤ 2. If SBED's claimed σ is honest, about 68% of the points sit in the inner band and 95% in the outer one.">?</span></h3><div id="lab-sweeps-z" style="height:360px"></div></div>
        ${groups}${failed}`;
    await ensurePlotly();
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
