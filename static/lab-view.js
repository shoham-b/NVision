// Lab page: the article's lab claims (search cost, information after detection, calibration) from /api/lab-study and
// /api/lab-detection-aligned, plus the per-run inspection. Shown when a lab is picked as the Study (app.js).
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


// A design-system colour token, read from styles.css (:root), for Plotly traces.
const labToken = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();

// Eq. search-cost: before the dip is found, a measurement learns about it only within about a linewidth Γ (FWHM), so
// the window holds W/Γ places to check, each needing about 1/(k SNR²) measurements: n_search ≳ (8/π) (W/Γ) / (k SNR²).
// k = 1 shot per lab measurement; SNR = C/σ of the run's own model and noise (lab_detection_aligned.analyse_window).
const labSearchCost = (w) => (8 / Math.PI) * (w.window_mhz / w.model_fwhm_mhz) / (w.snr * w.snr);

function labCard(title, help, claim, id, height = 400) {
    return `<div class="chart-card">
        <h3>${title} <span tabindex="0" class="help-icon-visible" title="${help.replace(/"/g, '&quot;')}">?</span></h3>
        <p class="chart-claim">${claim}</p>
        <div id="${id}" style="height:${height}px; width:100%;"></div>
    </div>`;
}

function labLayout(xTitle, yTitle) {
    return {
        template: 'plotly_white',
        margin: { t: 30, b: 55, l: 70, r: 15 },
        hovermode: 'closest',
        legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1, font: { size: 10 } },
        xaxis: { title: xTitle, type: 'log', gridcolor: labToken('chart-grid') },
        yaxis: { title: yTitle, type: 'log', gridcolor: labToken('chart-grid') },
    };
}

// Fig. lab_search: the median detection step of each window against Eq. search-cost.
function labPlotSearch(div, wins) {
    const det = wins.filter((w) => w.detected_step_median != null);
    const x = det.map((w) => w.window_mhz / w.model_fwhm_mhz);
    const traces = [
        {
            x, y: det.map((w) => w.detected_step_median), mode: 'lines+markers', name: 'median detection step (IQR)',
            line: { color: labToken('highlight-1'), width: 2.5 }, marker: { size: 9 },
            error_y: {
                type: 'data', symmetric: false,
                array: det.map((w) => w.detected_step_q75 - w.detected_step_median),
                arrayminus: det.map((w) => w.detected_step_median - w.detected_step_q25),
            },
            customdata: det.map((w) => [w.window_mhz, w.snr, w.detected_step_median / labSearchCost(w), `${w.n_detected}/${w.n_runs}`]),
            hovertemplate: 'W = %{customdata[0]} MHz, W/Γ = %{x:.3g}<br>detection step %{y}<br>SNR %{customdata[1]:.2f}<br>'
                + '%{customdata[2]:.3g} × Eq. search-cost<br>detected %{customdata[3]}<extra></extra>',
        },
        {
            x, y: det.map(labSearchCost), mode: 'lines+markers', name: 'Eq. search-cost (8/π)(W/Γ)/SNR²',
            line: { color: labToken('slate-700'), width: 2, dash: 'dash' }, marker: { size: 6, symbol: 'diamond' },
        },
    ];
    Plotly.newPlot(div, traces, labLayout('window W / Γ', 'measurements to detection'), { responsive: true });
    return det.map((w) => w.detected_step_median / labSearchCost(w));
}

// Calibration: the spread of the runs' final estimates against the σ they claim, one point per window.
function labPlotCalibration(div, windows) {
    const ok = windows.filter((r) => r.claimed_sigma_mhz != null && r.across_run_std_mhz != null);
    const vals = ok.flatMap((r) => [r.claimed_sigma_mhz, r.across_run_std_mhz]);
    const lo = Math.min(...vals) / 1.5, hi = Math.max(...vals) * 1.5;
    const traces = [
        {
            x: ok.map((r) => r.claimed_sigma_mhz), y: ok.map((r) => r.across_run_std_mhz), mode: 'markers+text',
            text: ok.map((r) => `${r.window_mhz} MHz`), textposition: 'top center', name: 'one window',
            marker: { size: 11, color: labToken('highlight-1') },
            customdata: ok.map((r) => [r.window_mhz, r.overconfidence, r.within_1sigma, r.within_2sigma, `${r.n_converged}/${r.n_runs}`]),
            hovertemplate: 'W = %{customdata[0]} MHz (%{customdata[4]} converged)<br>claimed σ %{x:.3g} MHz<br>across-run std %{y:.3g} MHz<br>'
                + 'spread ÷ claimed %{customdata[1]:.2f}<br>within 1σ %{customdata[2]:.2f}, within 2σ %{customdata[3]:.2f}<extra></extra>',
        },
        { x: [lo, hi], y: [lo, hi], mode: 'lines', name: 'calibrated: spread = claimed σ', line: { color: labToken('slate-500'), dash: 'dash', width: 1.5 } },
    ];
    const layout = labLayout('median claimed σ at the end of the run (MHz)', 'std of the final estimates across runs (MHz)');
    layout.xaxis.range = [Math.log10(lo), Math.log10(hi)];
    layout.yaxis.range = [Math.log10(lo), Math.log10(hi)];
    Plotly.newPlot(div, traces, layout, { responsive: true });
    return ok.map((r) => r.overconfidence).filter((v) => v != null);
}

const labMedian = (xs) => {
    const v = xs.filter((x) => x != null && Number.isFinite(x)).sort((a, b) => a - b);
    if (!v.length) return null;
    const m = Math.floor(v.length / 2);
    return v.length % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
};

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
        body.innerHTML = `<p class="empty-state">Lab "${name}" failed to load: ${err.message}.</p>`;
        return;
    }
    const cols = [
        ['Window (MHz)', (r) => r.window_mhz],
        ['Converged', (r) => `${r.n_converged}/${r.n_runs}`],
        ['Claimed σ (MHz)', (r) => fmt(r.claimed_sigma_mhz)],
        ['Across-run std (MHz)', (r) => fmt(r.across_run_std_mhz)],
        ['Spread ÷ claimed σ', (r) => fmt(r.overconfidence)],
        ['Within 1σ (nominal 0.68)', (r) => fmt(r.within_1sigma)],
        ['Within 2σ (nominal 0.95)', (r) => fmt(r.within_2sigma)],
    ];
    const cell = 'border:1px solid var(--slate-200); padding:0.35em 0.6em; text-align:right;';
    body.innerHTML = `
        <p style="margin:0 0 0.5em; color:var(--slate-600); font-size:0.85em;">Lab: <b>${name}</b> · reference resonance <span class="claim-value">${fmt(data.reference_mhz, 2)} MHz</span> (median of the converged runs' final means). Each run is read from its own detection, the first steep, lasting drop of its claimed σ.</p>
        <div class="bayes-tab-bar" role="tablist" aria-label="Lab study views">
            <button type="button" class="bayes-tab-button is-active" role="tab" aria-selected="true" data-lab-tab="claims">Claims</button>
            <button type="button" class="bayes-tab-button" role="tab" aria-selected="false" data-lab-tab="per-run">Per Run</button>
        </div>
        <div class="bayes-tab-panel" role="tabpanel" data-lab-pane="per-run">
            <div id="lab-per-run"></div>
            <div id="lab-replay" style="margin:1.2em 0;"></div>
        </div>
        <div class="bayes-tab-panel is-active" role="tabpanel" data-lab-pane="claims">
            <h3 class="group-label">Search + localize</h3>
            <div class="chart-pair">
                ${labCard('Search cost vs window',
                    'Each point is one drive_freq window: the median step at which the runs detected the dip (bars: interquartile range), against Eq. search-cost, (8/π)(W/Γ)/(k SNR²) with k = 1 shot, Γ the FWHM of the line the runs assume and SNR = C/σ of their own model and noise. The equation is a lower bound for any strategy at low SNR: before the dip is found no strategy knows where to look.',
                    '<span id="lab-search-claim">Article: the lab search cost grows linearly in W at about 25 × Eq. search-cost.</span>', 'lab-search')}
                ${labCard('Information per measurement after detection',
                    "Blue: the claimed information rate q̂ after each run's own detection, the slope of 1/⟨σ²⟩ per measurement in units of I_max of the lab line, over the second half of the lags. Orange: the placement efficiency, the mean I(d)/I_max of where the runs measured after detection, about each run's own final estimate. Dashed: the rate a sweep over the whole window gets, 8πΓ/(27W) (Eq. sweep-info). Measuring every time on the steepest slope reaches 1 (dotted).",
                    'Article: after detection q̂ and the placement are near 0.4–0.5 in narrow windows and fall to the sweep rate above about 25 Γ.', 'da-lab-rates')}
            </div>
            <h3 class="group-label">Calibration</h3>
            <div class="chart-pair">
                ${labCard('Claimed σ vs repeated-run scatter',
                    "One point per window: the median σ the runs claimed at their end against how far their final estimates scatter from each other. No ground truth exists for lab data, so the scatter also holds any drift of the line between recordings (0.4–0.6 MHz): read 'spread ÷ claimed σ' as an upper bound on the overconfidence. Calibrated runs sit on the dashed line.",
                    '<span id="lab-calib-claim">A calibrated σ matches the scatter of repeated runs.</span>', 'lab-calibration')}
                <div class="chart-card" style="overflow-x:auto;">
                    <h3>Calibration per window <span tabindex="0" class="help-icon-visible" title="Within kσ: share of runs whose final estimate lies within k claimed σ of the reference. Against the nominal 0.68 and 0.95 of a calibrated Gaussian posterior.">?</span></h3>
                    <table style="border-collapse:collapse; width:100%; font-size:0.9em; background:var(--surface);">
                        <thead><tr>${cols.map((c) => `<th style="${cell} background:var(--slate-50); color:var(--slate-600);">${c[0]}</th>`).join('')}</tr></thead>
                        <tbody>${data.windows.map((r) => `<tr>${cols.map((c) => `<td style="${cell} font-family:ui-monospace,monospace;">${c[1](r)}</td>`).join('')}</tr>`).join('')}</tbody>
                    </table>
                </div>
            </div>
            <div id="lab-da-table" style="overflow-x:auto; margin-top:1em;"><p class="empty-state">Aligning the runs at their own detection (reads every run file)…</p></div>
            <details style="margin-top:1em;">
                <summary style="cursor:pointer;"><h3 class="group-label" style="display:inline;">Claimed σ over the run, per window</h3></summary>
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:1em; margin-top:0.6em;">
                    <div>${labPlotHead('Claimed σ vs spread and error', "Solid: median self-reported posterior std (band: quartiles). Dashed: how far the runs' posterior means scatter from each other at that step (std across runs). Dotted: the rms error against the reference.", 'lab-spread')}<div id="lab-spread"></div></div>
                    <div>${labPlotHead('Claimed σ vs the elimination model', "Solid: median self-reported posterior std. Dotted: the article's mean curve (elimination search over the window W, then van Trees from each detection with J_d and I_max), fitted to the median at import with W fixed. Dash-dot: van Trees fitted to the runs aligned at their own detection, drawn from the median detection step. Folders imported before the model was stored show the claimed σ only (re-import them with nv matlab).", 'lab-fit')}<div id="lab-fit"></div></div>
                </div>
            </details>
        </div>`;
    let perRunDrawn = false, curvesDrawn = false;
    body.querySelectorAll('[data-lab-tab]').forEach((btn) => btn.addEventListener('click', () => {
        const tab = btn.dataset.labTab;
        body.querySelectorAll('[data-lab-tab]').forEach((b) => {
            b.classList.toggle('is-active', b === btn);
            b.setAttribute('aria-selected', String(b === btn));
        });
        body.querySelectorAll('[data-lab-pane]').forEach((p) => p.classList.toggle('is-active', p.dataset.labPane === tab));
        // Plotly lays out a hidden pane at zero width: draw Per Run the first time it is shown.
        if (tab === 'per-run' && !perRunDrawn) {
            perRunDrawn = true;
            renderLabPerRun(body.querySelector('#lab-per-run'), data);
            renderLabReplay(body.querySelector('#lab-replay'), name, data.reference_mhz);
        }
        body.querySelectorAll(`[data-lab-pane="${tab}"] .js-plotly-plot`).forEach((gd) => Plotly.Plots.resize(gd));
    }));
    await ensurePlotly();
    const over = labPlotCalibration(document.getElementById('lab-calibration'), data.windows);
    if (over.length) {
        document.getElementById('lab-calib-claim').innerHTML = `Spread ÷ claimed σ, median over windows: <span class="claim-value">${fmt(labMedian(over))}</span> (1 = calibrated; above 1 = overconfident, or line drift).`;
    }
    const details = body.querySelector('details');
    details.addEventListener('toggle', () => {
        if (details.open && !curvesDrawn) {
            curvesDrawn = true;
            labPlotCurves(data);
        }
    });
    // The detection-aligned numbers read every run file (about a minute): drawn when they arrive.
    let wins;
    try {
        const resp = await fetch(labApi(`lab-detection-aligned?name=${encodeURIComponent(name)}`), { cache: 'no-store' });
        if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);
        wins = await resp.json();
    } catch (err) {
        body.querySelector('#lab-da-table').innerHTML = `<p class="notice-amber">Detection-aligned analysis failed: ${err.message}</p>`;
        return;
    }
    const ratios = labPlotSearch(document.getElementById('lab-search'), wins);
    if (ratios.length) {
        document.getElementById('lab-search-claim').innerHTML = `Article: the lab search cost grows linearly in W at about 25 × Eq. search-cost. Here: <span class="claim-value">${ratios.map((r) => r.toFixed(0)).join(', ')} ×</span> over the windows.`;
    }
    daPlotLabRates(document.getElementById('da-lab-rates'), wins);
    body.querySelector('#lab-da-table').innerHTML = daLabTable(wins);
}

// Claimed σ against the run, per window: against the across-run spread and error, and against the article's mean curve
// fitted at import (c.search_fit). Folders without a stored fit show the claimed σ only: no fit is made up here.
function labPlotCurves(data) {
    const palette = ['highlight-1', 'highlight-6', 'highlight-3', 'highlight-2', 'highlight-4', 'highlight-5', 'highlight-7', 'highlight-8'].map(labToken);
    const spreadTraces = [], fitTraces = [];
    data.windows.forEach((r, j) => {
        const c = data.curves[String(r.window_mhz)];
        const name = `${r.window_mhz} MHz`;
        const color = palette[j % palette.length];
        const band = (group) => [
            { x: c.step, y: c.claimed_q75, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, legendgroup: group },
            { x: c.step, y: c.claimed_q25, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: color + '26', hoverinfo: 'skip', showlegend: false, legendgroup: group },
        ];
        const claimed = (group) => ({ x: c.step, y: c.claimed_median, mode: 'lines', line: { color, width: 2.5 }, name: `${name} claimed σ`, legendgroup: group });
        spreadTraces.push(...band(name + 's'), claimed(name + 's'),
            { x: c.actual_step, y: c.across_run_std, mode: 'lines', line: { color, width: 2.5, dash: 'dash' }, name: `${name} across-run spread`, legendgroup: name + 's' });
        if (c.rms_err_mhz) {
            spreadTraces.push({ x: c.rms_err_mhz.map((_, k) => k + 1), y: c.rms_err_mhz, mode: 'lines', line: { color, width: 2.5, dash: 'dot' }, name: `${name} rms error vs reference`, legendgroup: name + 's' });
        }
        fitTraces.push(...band(name + 'f'), claimed(name + 'f'));
        const sf = c.search_fit;
        if (sf) {
            fitTraces.push({ x: c.step, y: sf.model, mode: 'lines', line: { color, width: 3, dash: 'dot' }, name: `${name} mean curve (v=${sf.v_mhz_per_step.toPrecision(2)} MHz/meas, J_d=${sf.j_d_per_mhz2.toPrecision(2)}, I=${sf.i_max_per_mhz2.toPrecision(2)})`, legendgroup: name + 'f' });
            const det = sf.detection;
            if (det && det.n_detected !== undefined) {
                fitTraces.push({
                    x: det.steps, y: det.model, mode: 'lines+markers', line: { color, width: 2, dash: 'dashdot' },
                    marker: { size: det.steps.map((n, i) => (i === 0 ? 11 : 0)), symbol: 'diamond', color },
                    name: `${name} van Trees from each run's detection, median n_d=${det.step}, ${det.n_detected}/${det.n_runs} runs`,
                    legendgroup: name + 'f',
                });
            }
        }
    });
    const sigmaPlot = (id, traces, filename, title) => {
        const rows = traces.filter((t) => t.showlegend !== false).length;
        const bottom = 70 + 24 * rows, plotH = 330;
        const el = document.getElementById(id);
        el.dataset.rows = rows;
        el.style.height = `${plotH + bottom + 10}px`;
        Plotly.newPlot(id, traces, {
            margin: { t: 10, r: 10, b: bottom, l: 60 }, hovermode: 'x unified', font: { size: 13 },
            xaxis: { title: { text: 'measurement step', standoff: 8 }, type: 'log', gridcolor: labToken('chart-grid') },
            yaxis: { title: 'σ (MHz)', type: 'log', gridcolor: labToken('chart-grid') },
            legend: { orientation: 'h', yanchor: 'top', y: -(60 / plotH) },
        }, {
            responsive: true, displaylogo: false,
            modeBarButtonsToRemove: ['toImage'],
            modeBarButtonsToAdd: [{ name: 'Download figure (PNG)', icon: Plotly.Icons.camera, click: () => labExport(id, filename, title, 'png') }],
        });
        wireLabDownloads(id, filename, title);
    };
    sigmaPlot('lab-spread', spreadTraces, 'claimed-sigma-vs-spread', 'Claimed σ vs spread and error');
    sigmaPlot('lab-fit', fitTraces, 'claimed-sigma-vs-model', 'Claimed σ vs the elimination model');
}

// Per Run tab: one lab run's claimed σ against measurement step, picked by window and run, over its window's median and
// quartiles. data.curves[w].runs is null for folders imported before per-run histories were saved; those windows have no
// per-run view (no fallback to the aggregate).
function renderLabPerRun(pane, data) {
    const windows = data.windows.map((r) => r.window_mhz);
    const runsOf = (w) => data.curves[String(w)].runs;
    const available = windows.filter((w) => runsOf(w) !== null);
    if (!available.length) {
        pane.innerHTML = `<div class="empty-state" role="status" style="padding:2em; text-align:center; color:#666; background:#fafafa; border:1px dashed #ccc; border-radius:8px; margin:1em 0;">
            No per-run claimed σ histories in this lab study: it was imported before they were saved. Re-sync its source with <code>nv matlab</code> to add them.</div>`;
        return;
    }
    // Detection step, as in nvision/metrics/lab_search_fit.py detection_step (keep the two in sync): the step (1-based)
    // right after the run's first steep, lasting one-step drop of claimed σ. Steep: ln σ falls by at least
    // DETECTION_STEEP_FRACTION of the run's steepest one-step fall. Lasting: σ never later climbs back above the geometric
    // mean of its values before and after the drop.
    const DETECTION_STEEP_FRACTION = 0.7;
    const priorOf = (w) => data.curves[String(w)].prior_sigma_mhz;
    const detectionStep = (r) => {
        const ls = r.std_mhz.map(Math.log);
        const drops = ls.slice(1).map((v, k) => v - ls[k]);
        const steepest = Math.min(...drops);
        if (!(steepest < 0)) return null;
        for (let k = 0; k < drops.length; k++) {
            if (drops[k] > DETECTION_STEEP_FRACTION * steepest) continue;
            const mid = 0.5 * (ls[k] + ls[k + 1]);
            if (ls.slice(k + 1).every((v) => v <= mid)) return k + 2;
        }
        return null;
    };
    // The lab names runs Run_<W>MHz_<n>: list them in natural order (2 before 10), not the lexical order of run_number.
    const ordered = (w) => [...runsOf(w)].sort((a, b) => a.file.localeCompare(b.file, undefined, { numeric: true }));
    pane.innerHTML = `
        <div style="display:flex; flex-wrap:wrap; align-items:center; gap:0.8em; margin:0 0 0.6em;">
            <label class="control-label" for="lab-run-window">Window:</label>
            <select class="control-select" id="lab-run-window" aria-label="drive_freq window">
                ${windows.map((w) => `<option value="${w}"${runsOf(w) === null ? ' disabled' : ''}>${w} MHz${runsOf(w) === null ? ' (no per-run data)' : ''}</option>`).join('')}
            </select>
            <div aria-label="Lab run navigation" class="repeat-picker" role="group">
                <button aria-label="Previous run" class="control-button" id="lab-run-prev" type="button">Previous</button>
                <select aria-label="Lab run" class="control-select" id="lab-run"></select>
                <button aria-label="Next run" class="control-button" id="lab-run-next" type="button">Next</button>
            </div>
        </div>
        <p id="lab-run-summary" style="margin:0 0 0.5em; color:#475569; font-size:0.85em;"></p>
        ${labPlotHead('Claimed σ and error of one run', "Solid blue: this run's self-reported posterior std after each measurement. Red: its actual error |posterior mean − reference resonance| (the reference is the median final mean of the study's converged runs); red above blue means the run claims more precision than it has. Grey: the median and quartiles over all runs of the window, as on the Aggregated tab. Dotted: the prior σ = W/√12 of the drive_freq window. The diamond marks this run's detection: the step right after its first steep, lasting drop of claimed σ (ln σ falls by at least 0.7 of the run's steepest one-step fall, and σ never climbs back above the geometric mean of its values before and after); the circle the step at which the precision stop fired (replayed for full-budget runs).", 'lab-run-sigma')}
        <div id="lab-run-sigma" style="height:520px"></div>`;
    const winSel = pane.querySelector('#lab-run-window');
    const runSel = pane.querySelector('#lab-run');
    const fillRuns = () => {
        const w = Number(winSel.value);
        runSel.innerHTML = ordered(w).map((r, i) => {
            const det = detectionStep(r);
            return `<option value="${i}">${r.file} · detection at ${det ?? '–'} · stop ${r.converged ? r.steps_to_convergence : 'none'}</option>`;
        }).join('');
    };
    const draw = () => {
        const w = Number(winSel.value);
        const r = ordered(w)[Number(runSel.value)];
        const c = data.curves[String(w)];
        const steps = r.std_mhz.map((_, k) => k + 1);
        const prior = priorOf(w), det = detectionStep(r);
        const xEnd = Math.max(steps.length, c.step[c.step.length - 1]);
        const runColor = '#2563eb', grey = '#94a3b8';
        const hline = (y, name) => ({ x: [1, xEnd], y: [y, y], mode: 'lines', line: { color: '#64748b', width: 1.5, dash: 'dot' }, name, hoverinfo: 'skip' });
        const at = (n, symbol, name) => ({ x: [n], y: [r.std_mhz[n - 1]], mode: 'markers', marker: { size: 12, symbol, color: runColor, line: { color: '#fff', width: 1.5 } }, name });
        const traces = [
            { x: c.step, y: c.claimed_q75, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false, legendgroup: 'win' },
            { x: c.step, y: c.claimed_q25, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: grey + '40', name: `${w} MHz quartiles (all runs)`, hoverinfo: 'skip', legendgroup: 'win' },
            { x: c.step, y: c.claimed_median, mode: 'lines', line: { color: grey, width: 2 }, name: `${w} MHz median`, legendgroup: 'win' },
            hline(prior, `prior σ = W/√12 (${prior.toPrecision(3)} MHz)`),
            { x: steps, y: r.std_mhz, mode: 'lines', line: { color: runColor, width: 2.5 }, name: `${r.file} claimed σ` },
            { x: steps, y: r.err_mhz.map(Math.abs), mode: 'lines', line: { color: '#dc2626', width: 1.8 }, name: `${r.file} |mean − reference|` },
        ];
        if (det !== null) traces.push(at(det, 'diamond', `detection (first steep, lasting drop) at step ${det}`));
        if (r.converged) traces.push(at(r.steps_to_convergence, 'circle', `precision stop at step ${r.steps_to_convergence}`));
        pane.querySelector('#lab-run-summary').textContent = `${r.file} (run ${Number(runSel.value) + 1} of ${runsOf(w).length} in the ${w} MHz window): ${steps.length} measurements; detection (first steep, lasting drop of claimed σ) at step ${det ?? '– (none)'}; precision stop ${r.converged ? `at step ${r.steps_to_convergence}` : 'never fired'}; final error ${Math.abs(r.err_mhz[r.err_mhz.length - 1]).toPrecision(2)} MHz against claimed σ ${r.std_mhz[r.std_mhz.length - 1].toPrecision(2)} MHz.`;
        // Legend under the axis title, two entries per row, sized as on the Aggregated tab (labExport reads dataset.rows).
        const rows = Math.ceil(traces.filter((t) => t.showlegend !== false).length / 2);
        const bottom = 70 + 24 * rows, plotH = 380;
        const el = document.getElementById('lab-run-sigma');
        el.dataset.rows = rows;
        el.style.height = `${plotH + bottom + 10}px`;
        const filename = `claimed-sigma-${r.file}`, title = `Claimed σ and error of ${r.file}`;
        Plotly.react(el, traces, {
            margin: { t: 10, r: 10, b: bottom, l: 60 }, hovermode: 'x unified', font: { size: 13 },
            xaxis: { title: { text: 'measurement step', standoff: 8 }, type: 'log', gridcolor: '#eeeeee' },
            yaxis: { title: 'σ (MHz)', type: 'log', gridcolor: '#eeeeee' },
            legend: { orientation: 'h', yanchor: 'top', y: -(60 / plotH) },
        }, {
            responsive: true, displaylogo: false,
            modeBarButtonsToRemove: ['toImage'],
            modeBarButtonsToAdd: [{ name: 'Download figure (PNG)', icon: Plotly.Icons.camera, click: () => labExport('lab-run-sigma', filename, title, 'png') }],
        });
        wireLabDownloads('lab-run-sigma', filename, title);
        pane.querySelector('#lab-run-prev').disabled = Number(runSel.value) === 0;
        pane.querySelector('#lab-run-next').disabled = Number(runSel.value) === runSel.options.length - 1;
    };
    winSel.value = String(available[0]);
    fillRuns();
    winSel.addEventListener('change', () => { fillRuns(); draw(); });
    runSel.addEventListener('change', draw);
    pane.querySelector('#lab-run-prev').addEventListener('click', () => { runSel.value = String(Number(runSel.value) - 1); draw(); });
    pane.querySelector('#lab-run-next').addEventListener('click', () => { runSel.value = String(Number(runSel.value) + 1); draw(); });
    draw();
}

// Live replay of one lab run (from /api/lab-runs and /api/lab-run), laid out like the simulation's Run view: a sticky timeline
// (Play, slider, speed) drives the measured-signal plot (samples coloured by step, focus windows shaded) and, below it, the
// posterior evolution and parameter convergence. The lab saves the posterior of the resonance centre as a mean and a std per
// step, so the posterior panel draws that normal approximation, not the lab's particles.
async function renderLabReplay(host, name, referenceMhz) {
    const api = (p) => `${window.NVISION_ASSET_PREFIX || ''}api/${p}`;
    let files = [];
    try {
        const r = await fetch(api(`lab-runs?name=${encodeURIComponent(name)}`), { cache: 'no-store' });
        if (r.ok) files = await r.json();
    } catch (err) { /* no run files: the panel stays hidden */ }
    if (!files.length) { host.innerHTML = ''; return; }
    host.innerHTML = `
        <h2 style="margin:0.6em 0 0.3em; font-size:1.2em;">Run replay <span tabindex="0" class="help-icon-visible" title="One lab run, step by step. Press Play or drag the slider: the plots show the run as it was after that many measurements.">?</span></h2>
        <div class="sticky-selectors" style="top:0;">
            <div class="control-row"><span class="control-label">Run:</span>
                <select id="lab-replay-file" class="control-select" style="max-width:30em;">${files.map((f) => `<option value="${f}">${f}</option>`).join('')}</select></div>
            <div class="global-timeline-controls">
                <button id="lab-replay-play" class="timeline-btn" type="button"><svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block; vertical-align:middle; margin-right:4px;"><path d="M8 5v14l11-7z"/></svg><span>Play</span></button>
                <div class="timeline-slider-container">
                    <div style="position:relative; flex:1; display:flex; align-items:center; margin-right:8px;"><input type="range" id="lab-replay-step" class="timeline-slider" min="1" max="1" value="1" style="width:100%; margin:0;"></div>
                    <span id="lab-replay-label" class="timeline-label">Step: --</span>
                </div>
                <select id="lab-replay-speed" class="speed-select">
                    <option value="500">Speed: 0.5x</option><option value="200">Speed: 1x</option><option value="100" selected>Speed: 2x</option><option value="50">Speed: 4x</option>
                </select>
            </div>
        </div>
        <div id="lab-replay-info" style="font-family:ui-monospace,monospace; font-size:0.85em; color:#334155; margin:0.5em 0 0.3em;"></div>
        <div id="lab-replay-signal" style="width:100%; height:480px;"></div>
        <h3 style="margin:1em 0 0.3em; font-size:1em;">Posterior evolution <span tabindex="0" class="help-icon-visible" title="Normal approximation of the posterior of the resonance centre after every step (from the posterior mean and std the lab saved), up to the shown step; the line marks the shown step. Brighter is more probable; the colour of each row is scaled to its own peak so the late, narrow posteriors stay visible.">?</span></h3>
        <div id="lab-replay-posterior" style="width:100%; height:360px;"></div>
        <h3 style="margin:1em 0 0.3em; font-size:1em;">Parameter convergence <span tabindex="0" class="help-icon-visible" title="Posterior std of the resonance centre against measurement step; the dot is the shown step. Dotted: the width of the focus window.">?</span></h3>
        <div id="lab-replay-conv" style="width:100%; height:300px;"></div>`;
    const $ = (id) => host.querySelector(`#${id}`);
    const cfg = { responsive: true, displaylogo: false };
    let tr = null, timer = null, heat = null;
    const setPlaying = (on) => {
        if (!on) { clearInterval(timer); timer = null; }
        $('lab-replay-play').querySelector('span').textContent = on ? 'Pause' : 'Play';
    };
    // Posterior heatmap grid (rows: steps, columns: drive_freq), built once per run.
    const buildHeat = () => {
        const nx = 400, [a, b] = tr.scan_range_mhz;
        const xs = Array.from({ length: nx }, (_, i) => a + ((b - a) * i) / (nx - 1));
        const z = tr.mean_mhz.map((m, i) => {
            const sd = Math.max(tr.std_mhz[i], 1e-6), row = xs.map((x) => Math.exp(-0.5 * ((x - m) / sd) ** 2));
            const peak = Math.max(...row);
            return row.map((v) => v / peak);
        });
        heat = { xs, z };
    };
    const draw = (k) => {
        const last = k - 1, x = tr.step.slice(0, k), f = tr.drive_freq_mhz.slice(0, k), sig = tr.signal.slice(0, k);
        const mean = tr.mean_mhz[last], sd = tr.std_mhz[last];
        $('lab-replay-label').textContent = `Step: ${k}/${tr.n_steps}`;
        $('lab-replay-step').value = k;
        const cands = tr.candidates ? tr.candidates[last] || [] : null;
        const lo = tr.window_lo_mhz[last], hi = tr.window_hi_mhz[last], scanW = tr.scan_range_mhz[1] - tr.scan_range_mhz[0];
        $('lab-replay-info').textContent = `measured ${f[last].toFixed(2)} MHz${tr.exploration && tr.exploration[last] ? ' (exploration)' : ''} · posterior ${mean.toFixed(2)} ± ${sd.toFixed(2)} MHz`
            + (cands ? ` · ${cands.length} focus candidate window(s)${cands.length ? `, narrowest ${Math.min(...cands.map(([p, q]) => q - p)).toFixed(1)} MHz` : ''}` : ` · focus window ${(hi - lo).toFixed(1)} MHz (${(100 * (hi - lo) / scanW).toFixed(0)}% of scan)${tr.locked[last] ? ', locked' : ''}`);

        // Focus windows, as in the simulation view: one candidate is the settled window (solid), several are pale boxes.
        const windows = cands !== null ? cands : (tr.locked[last] ? [[lo, hi]] : []);
        const shapes = windows.map(([p, q]) => ({ type: 'rect', xref: 'x', yref: 'paper', x0: p, x1: q, y0: 0, y1: 1, line: { width: 0 }, layer: 'below',
            fillcolor: windows.length === 1 ? 'rgba(245,158,11,0.22)' : 'rgba(234,88,12,0.10)' }));
        const annotations = windows.length === 1 ? [{ text: 'focus window', x: windows[0][0], xref: 'x', y: 1, yref: 'paper', yanchor: 'bottom', xanchor: 'left', showarrow: false, font: { size: 11 } }] : [];
        shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: mean - sd, x1: mean + sd, y0: 0, y1: 1, line: { width: 0 }, fillcolor: 'rgba(37,99,235,0.15)', layer: 'below' });
        if (referenceMhz != null) {
            shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: referenceMhz, x1: referenceMhz, y0: 0, y1: 1, line: { width: 1.5, color: '#16a34a', dash: 'dash' } });
            annotations.push({ text: 'reference resonance', x: referenceMhz, xref: 'x', y: 1, yref: 'paper', yanchor: 'bottom', xanchor: 'center', showarrow: false, font: { size: 11, color: '#16a34a' } });
        }
        // Dense x grid, with extra points around the centre so a narrow dip stays smooth on a wide scan.
        const curveX = [...Array.from({ length: 600 }, (_, i) => tr.scan_range_mhz[0] + ((tr.scan_range_mhz[1] - tr.scan_range_mhz[0]) * i) / 599),
            ...Array.from({ length: 201 }, (_, i) => mean + ((i - 100) * tr.model.fwhm_mhz) / 20)].sort((p, q) => p - q);
        Plotly.react('lab-replay-signal', [
            { x: f, y: sig, mode: 'markers', name: 'measurements', marker: { size: 7, color: x, colorscale: 'Viridis', cmin: 1, cmax: tr.n_steps, colorbar: { title: { text: 'step' } } },
              hovertemplate: 'step %{marker.color}<br>%{x:.2f} MHz<br>signal %{y:.4g}<extra></extra>' },
            { x: [f[last]], y: [sig[last]], mode: 'markers', showlegend: false, hoverinfo: 'skip', marker: { size: 15, color: 'rgba(0,0,0,0)', line: { color: '#dc2626', width: 2 } } },
            { x: [mean, mean], y: [Math.min(...tr.signal), Math.max(...tr.signal)], mode: 'lines', name: 'posterior mean', line: { color: '#2563eb', width: 2 }, hoverinfo: 'skip' },
            { x: curveX, y: curveX.map((v) => tr.model.baseline * (1 - tr.model.contrast / (1 + ((v - mean) / (tr.model.fwhm_mhz / 2)) ** 2))), mode: 'lines', name: 'most likely signal',
              line: { color: '#dc2626', width: 2.5 }, hovertemplate: '%{x:.2f} MHz<br>%{y:.4f}<extra>most likely signal</extra>' },
        ], {
            margin: { t: 24, r: 10, b: 50, l: 70 }, font: { size: 13 }, shapes, annotations,
            xaxis: { title: 'drive_freq (MHz)', range: tr.scan_range_mhz, gridcolor: '#eeeeee' },
            yaxis: { title: 'signal', gridcolor: '#eeeeee' }, legend: { orientation: 'h', y: -0.2 },
        }, cfg);

        Plotly.react('lab-replay-posterior', [
            { type: 'heatmap', x: heat.xs, y: tr.step.slice(0, k), z: heat.z.slice(0, k), colorscale: 'Viridis', showscale: false, hoverinfo: 'skip' },
            { x: [mean - 2 * sd, mean + 2 * sd], y: [k, k], mode: 'lines', line: { color: '#dc2626', width: 2 }, showlegend: false, hoverinfo: 'skip' },
        ], {
            margin: { t: 10, r: 10, b: 50, l: 70 }, font: { size: 13 },
            xaxis: { title: 'drive_freq (MHz)', range: tr.scan_range_mhz }, yaxis: { title: 'measurement step', range: [tr.n_steps, 1] },
        }, cfg);

        Plotly.react('lab-replay-conv', [
            { x: tr.step, y: tr.std_mhz, mode: 'lines', name: 'posterior std', line: { color: '#cbd5e1', width: 1.5 }, hoverinfo: 'skip' },
            { x, y: tr.std_mhz.slice(0, k), mode: 'lines', name: 'posterior std (so far)', line: { color: '#2563eb', width: 2.5 } },
            { x: tr.step, y: tr.window_hi_mhz.map((h, i) => h - tr.window_lo_mhz[i]), mode: 'lines', name: 'focus window width', line: { color: '#f59e0b', width: 1.5, dash: 'dot' }, hoverinfo: 'skip' },
            { x: [k], y: [sd], mode: 'markers', showlegend: false, hoverinfo: 'skip', marker: { size: 11, color: '#dc2626' } },
        ], {
            margin: { t: 10, r: 10, b: 60, l: 70 }, font: { size: 13 },
            xaxis: { title: 'measurement step', range: [0, tr.n_steps + 1], gridcolor: '#eeeeee' },
            yaxis: { title: 'MHz', type: 'log', gridcolor: '#eeeeee' }, legend: { orientation: 'h', y: -0.3 },
        }, cfg);
    };
    const load = async () => {
        setPlaying(false);
        const r = await fetch(api(`lab-run?name=${encodeURIComponent(name)}&file=${encodeURIComponent($('lab-replay-file').value)}`), { cache: 'no-store' });
        if (!r.ok) { $('lab-replay-info').textContent = `Failed to load run: ${(await r.json()).detail || r.status}`; return; }
        tr = await r.json();
        buildHeat();
        $('lab-replay-step').max = tr.n_steps;
        draw(tr.n_steps);
    };
    $('lab-replay-file').onchange = load;
    $('lab-replay-step').oninput = (e) => { setPlaying(false); draw(Number(e.target.value)); };
    $('lab-replay-play').onclick = () => {
        if (timer) return setPlaying(false);
        let k = Number($('lab-replay-step').value);
        if (k >= tr.n_steps) k = 0;
        setPlaying(true);
        // The tick delay is read from the speed select every frame so changing it mid-play takes effect.
        const tick = () => {
            if (!timer) return;
            k = Math.min(tr.n_steps, k + 1);
            draw(k);
            if (k >= tr.n_steps) return setPlaying(false);
            timer = setTimeout(tick, Number($('lab-replay-speed').value));
        };
        timer = setTimeout(tick, 0);
    };
    load();
}

// Heading + help icon + PNG/SVG download buttons for one lab plot (wired by wireLabDownloads).
function labPlotHead(title, help, id) {
    const btn = 'font-size:0.8em; padding:0.15em 0.7em; margin-left:0.4em; border:1px solid #cbd5e1; border-radius:999px; background:#fff; color:#334155; cursor:pointer;';
    return `<h3 style="margin:0 0 0.3em; font-size:1em; display:flex; align-items:center; gap:0.3em;">${title} <span tabindex="0" class="help-icon-visible" title="${help.replace(/"/g, '&quot;')}">?</span>
        <span style="margin-left:auto; font-weight:normal;"><button type="button" data-dl="${id}" data-fmt="png" style="${btn}">Download PNG</button><button type="button" data-dl="${id}" data-fmt="svg" style="${btn}">SVG</button></span></h3>`;
}

// Downloads the plot as a clean, titled, white-background figure. The export is compact (plot area about 5:4, rendered at 3x) with bigger
// fonts, so text is large relative to the graph; margins and legend position are rescaled for the bigger text, then restored.
async function labExport(id, filename, title, fmt) {
    {
        {
            const gd = document.getElementById(id);
            const k = 20 / gd.layout.font.size;  // text scale-up for the export
            const rows = Number(gd.dataset.rows), plotW = 800, plotH = 640, bottom = Math.round(k * (70 + 24 * rows)), top = 70;
            const orig = { 'font.size': gd.layout.font.size, 'margin.t': gd.layout.margin.t, 'margin.b': gd.layout.margin.b, 'margin.l': gd.layout.margin.l, 'legend.y': gd.layout.legend.y, 'title.text': '', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' };
            await Plotly.relayout(gd, { 'font.size': 20, 'title.text': title, 'title.font.size': 26, 'title.x': 0.02, 'margin.t': top, 'margin.b': bottom, 'margin.l': 80, 'legend.y': -(k * 60 / plotH), paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff' });
            try {
                await Plotly.downloadImage(gd, { format: fmt, filename, width: 80 + plotW + 10, height: top + plotH + bottom, scale: 3 });
            } finally {
                await Plotly.relayout(gd, orig);
            }
        }
    }
}

function wireLabDownloads(id, filename, title) {
    document.querySelectorAll(`button[data-dl="${id}"]`).forEach((b) => {
        b.onclick = () => labExport(id, filename, title, b.dataset.fmt);
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
                ...(c[`searchfit_${param}`] ? [{
                    x: c.steps, y: c[`searchfit_${param}`], mode: 'lines', line: { color, width: 2, dash: 'longdash' },
                    name: `${label}search + van Trees fit (v ${c[`searchfit_${param}_params`].v_mhz_per_step.toPrecision(2)} MHz/meas, J_d ${c[`searchfit_${param}_params`].j_d_per_mhz2.toPrecision(2)}/MHz², I ${c[`searchfit_${param}_params`].i_max_per_mhz2.toPrecision(2)}/MHz², rms ln ${c[`searchfit_${param}_params`].rms_log.toPrecision(2)})`,
                    legendgroup: f.id,
                }] : []),
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
