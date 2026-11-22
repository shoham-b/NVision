// Runs aligned at their own dip detection: the localize ratio against van Trees restarted at detection, and the onset of
// a Gaussian posterior (Stages tab, Localize and Refine scopes), and the lab's claimed refine rate and placement (Lab tab,
// "From detection"). Data: /api/detection-aligned and /api/lab-detection-aligned (nvision/metrics/detection_aligned.py,
// nvision/metrics/lab_detection_aligned.py; every number is computed there, this file only draws).
// Needs ensurePlotly from plotly-utils.js.

const DA_MAX_USE_CASES = 12;
const DA_LOCATORS = ['Bayesian-SBED', 'SimpleSweep-optimal-x0.5', 'SimpleSweep-optimal', 'SimpleSweep-optimal-x2'];
const DA_REF_COLOR = '#94a3b8';   // reference lines (rho = 1, the Gaussian threshold), as the bound lines elsewhere

const daFmt = (v, digits = 3) => (v == null || !Number.isFinite(v) ? '–' : Number(v).toPrecision(digits));
const daEsc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const daShort = (name) => name.replace('SimpleSweep-optimal-x', 'Sweep k*×').replace('SimpleSweep-optimal', 'Sweep k*');
const daApi = (path) => `${window.NVISION_ASSET_PREFIX || ''}api/${path}`;

const daCard = (title, help, id, height) => `
    <div style="flex:1 1 46%; min-width:420px; background:#fff; border:1px solid #f1f5f9; border-radius:8px; padding:10px;">
        <h3 style="margin-left:10px;">${title} <span tabindex="0" class="help-icon-visible" title="${daEsc(help)}">?</span></h3>
        <div id="${id}" style="height:${height}px; width:100%;"></div>
    </div>`;

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

// ── Simulation (Stages tab) ──────────────────────────────────────────────────────────────────────────────────────────

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

function daSimTable(locs) {
    const lagCols = ['0', '10', '30', '100', '1000'];
    const rows = locs.map(([name, loc]) => {
        const g = loc.gaussian;
        const rho = loc.rho_j_tau_at || {};
        const viol = loc.bound_violated;
        return `<tr>
            <td style="padding:3px 10px; font-weight:600; text-align:left;">${daShort(name)}</td>
            <td style="${daCell}">${loc.n_runs}${loc.n_undetected ? ` (+${loc.n_undetected} never)` : ''}</td>
            <td style="${daCell}">${daFmt(loc.detected_step_median, 4)}</td>
            ${lagCols.map((k) => `<td style="${daCell}">${daFmt(rho[k], 2)}</td>`).join('')}
            <td style="${daCell} ${viol ? 'color:#b91c1c; font-weight:700;' : ''}">${viol == null ? '–' : viol ? 'yes ⚠' : 'no'}</td>
            <td style="${daCell}">${Math.round(100 * g.share_reached)} %</td>
            <td style="${daCell}">${daFmt(g.onset_lag_median, 3)} (${daFmt(g.onset_lag_q25, 3)}–${daFmt(g.onset_lag_q75, 3)})</td>
            <td style="${daCell}">${daFmt(g.onset_step_median, 4)}</td>
        </tr>`;
    });
    return daTable(
        ['Locator', 'runs detected', 'median detection step', ...lagCols.map((k) => `ρ<sub>J_τ</sub> at k=${k}`), 'bound broken',
            'Gaussian by the budget', 'onset, k after detection (IQR)', 'onset step'],
        rows,
        `ρ<sub>J_τ</sub> ≤ 1 is the restarted van Trees bound; "bound broken" = the lower end of its bootstrap interval is above 1.01 at some k. Where the posterior at detection is still the box of the window (narrow windows), its smoothed J_τ is too small and ρ rises far above 1: the bound does not apply there, not a strategy beating it. Lags are limited to those every detected run reaches in the budget (up to ${locs.length ? locs[0][1].max_lag : '–'}).`,
    );
}

// Claimed vs true information rate after detection, and the share of measurements placed far from the posterior peak.
function daRateTable(locs) {
    const ci = (c) => (c ? ` [${daFmt(c[0], 2)}, ${daFmt(c[1], 2)}]` : '');
    const rate = (w) => (!w || w.unavailable
        ? [`<td style="${daCell}" colspan="3" title="${daEsc(w ? w.unavailable : '')}">–</td>`]
        : [`<td style="${daCell}">${daFmt(w.q_hat, 2)}${ci(w.q_hat_ci)}</td>`,
            `<td style="${daCell}">${daFmt(w.q_true, 2)}${ci(w.q_true_ci)}</td>`,
            `<td style="${daCell}">${daFmt(w.q_hat_over_q, 2)}${ci(w.q_hat_over_q_ci)}</td>`]).join('');
    const pct = (v) => (v == null ? '–' : `${(100 * v).toFixed(1)} %`);
    const rows = locs.map(([name, loc]) => {
        const c = loc.claimed_rate || {}, f = loc.far_placed || {};
        return `<tr>
            <td style="padding:3px 10px; font-weight:600; text-align:left;">${daShort(name)}</td>
            ${rate(c.from_detection)}${rate(c.from_gaussian)}
            <td style="${daCell}">${pct(f.converge_share_median)} (${f.converge_n_runs ?? 0})</td>
            <td style="${daCell}">${pct(f.after_detection_share_median)} (${f.after_detection_n_runs ?? 0})</td>
            <td style="${daCell}">${pct(f.far_mass_tau_median)}</td>
        </tr>`;
    });
    return daTable(
        ['Locator', 'q̂ from detection', 'q true', 'q̂/q', 'q̂ from Gaussian onset', 'q true', 'q̂/q',
            'far-placed, detection → Gaussian (runs)', 'far-placed, detection → end (runs)', 'posterior mass far at detection'],
        rows,
        'q̂ = slope of 1/⟨σ_claimed²⟩ per measurement in units of I_max (q true: the same from the true error ⟨e²⟩), straight least squares over the lag grid, 95 % bootstrap interval over runs. q̂/q > 1 is overconfident. Right after detection the posterior collapses from two peaks to one, which shrinks σ_claimed by more than I_max per measurement without overconfidence, so the second window starts at each run\'s Gaussian onset. Far-placed = share of the stored points (≤ ~80 per run) placed more than 3 FWHM from the mode of the belief before them, median over runs.',
    );
}

async function renderDetectionAligned(root, genList, noiseList, colorOf) {
    const token = String(Date.now() + Math.random());
    root.dataset.token = token;
    if (!genList.length || !noiseList.length) { root.innerHTML = ''; return; }
    root.innerHTML = '<p style="color:#64748b; padding:1em;">Aligning the runs at their own detection…</p>';
    await ensurePlotly();
    const params = new URLSearchParams({ generator: genList.join(','), noise: noiseList.join(',') });
    let rows;
    try {
        const resp = await fetch(daApi(`detection-aligned?${params}`), { cache: 'no-store' });
        if (!resp.ok) throw new Error(`/api/detection-aligned returned ${resp.status} (needs \`nv serve\`)`);
        rows = await resp.json();
    } catch (err) {
        if (root.dataset.token === token) root.innerHTML = `<p style="color:#721c24; background:#f8d7da; padding:1em; border-radius:6px;">${daEsc(err.message)}</p>`;
        return;
    }
    if (root.dataset.token !== token) return;   // a newer selection took over
    const cases = new Map();
    for (const r of rows) {
        if (!DA_LOCATORS.includes(r.strategy)) continue;
        const key = `${r.generator}\u0000${r.noise}`;
        if (!cases.has(key)) cases.set(key, { generator: r.generator, noise: r.noise, locs: [], skipped: [] });
        (r.unavailable ? cases.get(key).skipped : cases.get(key).locs).push([r.strategy, r]);
    }
    if (!cases.size) { root.innerHTML = '<p style="color:#94a3b8; padding:1em;">No runs of the study locators for this selection.</p>'; return; }
    const list = [...cases.values()];
    list.forEach((uc) => uc.locs.sort((a, b) => DA_LOCATORS.indexOf(a[0]) - DA_LOCATORS.indexOf(b[0])));
    root.innerHTML = list.map((uc, i) => `
        <section style="margin:0 0 2em;">
            <h3 class="uc-title" style="margin:0.6em 0; padding:0.4em 0.8em; font-size:1em; color:#334155; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px;">${daEsc(uc.generator)} · ${daEsc(uc.noise)}</h3>
            ${uc.skipped.map(([n, r]) => `<p class="notice-amber" style="margin:0 0 0.5em; font-size:0.85em; color:#92400e; background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:0.4em 0.8em;">${daShort(n)}: ${daEsc(r.unavailable)}</p>`).join('')}
            ${uc.locs.length ? `
            <div style="display:flex; flex-wrap:wrap; gap:1.5em;">
                ${daCard('Localize: error against van Trees restarted at detection',
                    'Each run is counted from its own detection (k = 0 is the posterior the dip test fired on). ρ(k) = mean over runs of the restarted bound 1/(J + k·I_max) over the mean squared true error at k. Solid: J = each run\'s own posterior Fisher information J_τ, a bound, so ρ ≤ 1 (dashed line). Dotted: J = J_d = 12/Δd², the elimination model of the article, a comparison only. Right after detection ρ is near 0 while the run still has to find which side of the line it is on (localize); it climbs once the run has locked onto the line. Logarithmic k, so k = 0 is in the table.',
                    `da-rho-${i}`, 420)}
                ${daCard('Refine starts: the posterior becomes Gaussian',
                    'Negentropy of each run\'s posterior of zeeman_shift (nats; 0 for a Gaussian, about 0.18 for a uniform, more for two peaks), counted from the run\'s own detection: median (line) and quartiles (band). The density estimate itself leaves about 0.01 on a Gaussian cloud. Dotted: the threshold under which the posterior counts as one Gaussian peak; the diamond is the median step from which a run stays under it to the end of its budget.',
                    `da-neg-${i}`, 420)}
            </div>
            <div style="overflow-x:auto; margin-top:1em;">${daSimTable(uc.locs)}</div>
            <div style="overflow-x:auto; margin-top:1em;">${daRateTable(uc.locs)}</div>` : ''}
        </section>`).join('');
    list.forEach((uc, i) => {
        if (!uc.locs.length) return;
        daPlotRho(document.getElementById(`da-rho-${i}`), uc.locs, colorOf);
        daPlotNegentropy(document.getElementById(`da-neg-${i}`), uc.locs, colorOf, uc.locs[0][1].gaussian.threshold);
    });
}

// ── Lab (Lab tab, "From detection") ──────────────────────────────────────────────────────────────────────────────────

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

function daPlotLabSigma(div, wins) {
    const palette = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#db2777', '#0891b2'];
    const traces = [];
    wins.filter((w) => w.lags).forEach((w, j) => {
        const c = palette[j % palette.length];
        const g = `${w.window_mhz}`;
        traces.push(
            { ...daPositive(w.lags, w.mse.map(Math.sqrt)), mode: 'lines', name: `${w.window_mhz} MHz rms error vs reference`, line: { color: c, width: 2.5 }, legendgroup: g },
            { ...daPositive(w.lags, w.mean_claimed_var.map(Math.sqrt)), mode: 'lines', name: `${w.window_mhz} MHz rms claimed σ`, line: { color: c, width: 2, dash: 'dash' }, legendgroup: g },
            { ...daPositive(w.lags, w.bound_pinned.map(Math.sqrt)), mode: 'lines', name: `${w.window_mhz} MHz 1/√(J_pin + k I_max)`, line: { color: c, width: 1.5, dash: 'dot' }, legendgroup: g },
        );
    });
    Plotly.newPlot(div, traces, daLayout('measurements since the run\'s own detection, k', 'MHz', 'log', 'log'), { responsive: true });
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

async function renderLabDetectionAligned(pane, name) {
    pane.innerHTML = '<p style="color:#64748b; padding:1em;">Aligning the lab runs at their own detection (reads every run file)…</p>';
    let wins;
    try {
        const resp = await fetch(daApi(`lab-detection-aligned?name=${encodeURIComponent(name)}`), { cache: 'no-store' });
        if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);
        wins = await resp.json();
    } catch (err) {
        pane.innerHTML = `<p style="color:#721c24; background:#f8d7da; padding:1em; border-radius:6px;">${daEsc(err.message)}</p>`;
        return;
    }
    if (!wins.length) { pane.innerHTML = '<p style="color:#94a3b8; padding:1em;">No runs in this lab.</p>'; return; }
    await ensurePlotly();
    const anyCurve = wins.some((w) => w.lags);
    pane.innerHTML = `
        <p style="margin:0 0 0.6em; font-size:0.85em; color:#475569;">The lab saves only each run's final particles, so the onset of a Gaussian posterior cannot be read here.</p>
        <div style="display:flex; flex-wrap:wrap; gap:1.5em;">
            ${daCard('After detection: information per measurement against the window',
                'Blue: the claimed information rate q̂ after each run\'s own detection, in units of I_max of the lab line. Orange: the placement efficiency, the mean I(d)/I_max of where the runs measured after detection. Dashed: the rate a sweep over the whole window gets (Eq. sweep-info). A strategy that measures on the steepest slopes reaches 1 (dotted); one that keeps measuring across the window falls onto the sweep line.',
                'da-lab-rates', 420)}
            ${anyCurve ? daCard('Error and claimed σ since detection',
                'Per window, counted from each run\'s own detection: rms error against the lab reference (solid), rms claimed σ (dashed), and the van Trees form 1/√(J_pin + k·I_max) pinned on the claimed σ at detection (dotted), the rate a run measuring on the steepest slopes would follow. The error flattens at the drift of the line between recordings.',
                'da-lab-sigma', 420) : ''}
        </div>
        <div style="overflow-x:auto; margin-top:1em;">${daLabTable(wins)}</div>`;
    daPlotLabRates(document.getElementById('da-lab-rates'), wins);
    if (anyCurve) daPlotLabSigma(document.getElementById('da-lab-sigma'), wins);
}

window.renderDetectionAligned = renderDetectionAligned;
window.renderLabDetectionAligned = renderLabDetectionAligned;
window.DA_MAX_USE_CASES = DA_MAX_USE_CASES;
