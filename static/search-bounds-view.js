// Information tab, Search scope: the search-phase bounds against the measured search, per use case and locator.
// Data: /api/search-bounds (nvision/cli/search_bounds_api.py; every formula lives in nvision/metrics/search_bounds.py,
// this file only draws). Called by app.js (hlApplySearchBounds) with the Information tab's selection.
// Needs ensurePlotly from plotly-utils.js.

// More use cases than this make the request take minutes (each rebuilds a line and loads every run); ask for fewer.
const SB_MAX_USE_CASES = 12;
const SB_LOCATORS = ['Bayesian-SBED', 'SimpleSweep-optimal-x0.5', 'SimpleSweep-optimal', 'SimpleSweep-optimal-x2'];
const SB_BOUND_STYLE = {
    finite_reach: { name: 'finite reach (W − nΔd)³/12W', color: '#b45309', dash: 'dash' },
    change_of_measure: { name: 'change of measure (eq. search-bound)', color: '#15803d', dash: 'dash' },
    mean_model: { name: 'elimination model (eq. mean-curve)', color: '#7c3aed', dash: 'dot' },
};

const sbFmt = (v, digits = 3) => (v == null || !Number.isFinite(v) ? '–' : Number(v).toPrecision(digits));
const sbEsc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const sbShort = (name) => name.replace('SimpleSweep-optimal-x', 'Sweep k*×').replace('SimpleSweep-optimal', 'Sweep k*').replace('SimpleSweep-', 'Sweep ').replace('perFWHM', '/FWHM');
const sbMHz = (hz) => `${(hz / 1e6).toPrecision(3)} MHz`;

const sbCard = (title, help, id, height) => `
    <div style="flex:1 1 46%; min-width:420px; background:#fff; border:1px solid #f1f5f9; border-radius:8px; padding:10px;">
        <h3 style="margin-left:10px;">${title} <span tabindex="0" class="help-icon-visible" title="${sbEsc(help)}">?</span></h3>
        <div id="${id}" style="height:${height}px; width:100%;"></div>
    </div>`;

function sbLayout(xTitle, yTitle, yType) {
    return {
        template: 'plotly_white',
        margin: { t: 30, b: 55, l: 70, r: 15 },
        hovermode: 'closest',
        legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1, font: { size: 10 } },
        xaxis: { title: xTitle, type: 'log', gridcolor: '#f1f5f9' },
        yaxis: { title: yTitle, type: yType, gridcolor: '#f1f5f9' },
    };
}

// Mean squared TRUE error against the three bound curves, in units of the prior variance W²/12 (1 = no information).
function sbPlotMse(div, uc, colorOf) {
    const prior = uc.width_phys ** 2 / 12;
    const traces = [];
    for (const [key, style] of Object.entries(SB_BOUND_STYLE)) {
        const y = uc.curves[key];
        if (!y) continue;
        traces.push({
            x: uc.n, y: y.map((v) => (v == null ? null : Math.max(v / prior, 1e-12))), type: 'scatter', mode: 'lines',
            name: style.name, line: { color: style.color, dash: style.dash, width: 2 },
        });
    }
    for (const name of SB_LOCATORS) {
        const loc = uc.locators[name];
        if (!loc) continue;
        traces.push({
            x: uc.n, y: loc.mse.map((v) => (v == null ? null : Math.max(v / prior, 1e-12))), type: 'scatter', mode: 'lines',
            name: `${sbShort(name)} (${loc.n_runs} runs)`, line: { color: colorOf(name), width: 2.5 }, connectgaps: false,
        });
    }
    const layout = sbLayout('measurements n', 'mean squared true error / (W²/12)', 'log');
    layout.yaxis.range = [-6, 0.3];   // the bounds fall many decades below any run; keep the runs readable
    Plotly.newPlot(div, traces, layout, { responsive: true });
}

// Survival S(n) = share of runs with |error| > t, with the floor the bound implies, and the fit range.
function sbPlotSurvival(div, uc, colorOf) {
    const traces = [{
        x: uc.n, y: uc.curves.survival_floor, type: 'scatter', mode: 'lines', name: 'bound: exp(−(nκ̄+ln 2)/(1−2t/W))',
        line: { color: SB_BOUND_STYLE.change_of_measure.color, dash: 'dash', width: 2 },
    }];
    for (const name of SB_LOCATORS) {
        const loc = uc.locators[name];
        if (!loc) continue;
        traces.push({
            x: uc.n, y: loc.survival.map((v) => (v == null ? null : Math.max(v, 1e-4))), type: 'scatter', mode: 'lines',
            name: sbShort(name), line: { color: colorOf(name), width: 2.5 },
        });
    }
    const layout = sbLayout('measurements n', `S(n) = share of runs with |error| > ${sbFmt(uc.tolerance_hwhm, 2)} HWHM`, 'log');
    layout.yaxis.range = [-3.5, 0.1];
    layout.shapes = [0.9, 0.05].map((s) => ({
        type: 'line', xref: 'paper', x0: 0, x1: 1, y0: Math.log10(s), y1: Math.log10(s), yref: 'y',
        line: { color: '#cbd5e1', width: 1, dash: 'dot' },
    }));
    Plotly.newPlot(div, traces, layout, { responsive: true });
}

function sbKappaTable(uc) {
    const floor = uc.kappa_ratio_floor;
    const rows = SB_LOCATORS.filter((n) => uc.locators[n]).map((name) => {
        const loc = uc.locators[name];
        const ratio = loc.kappa_ratio;
        const over = ratio != null && ratio > floor;
        const isSweep = name.startsWith('SimpleSweep');
        return `<tr>
            <td style="padding:3px 10px; font-weight:600;">${sbShort(name)}</td>
            <td style="padding:3px 10px; text-align:right;">${loc.n_runs}</td>
            <td style="padding:3px 10px; text-align:right;">${sbFmt(loc.kappa_eff)}</td>
            <td style="padding:3px 10px; text-align:right; ${over ? 'color:#b91c1c; font-weight:700;' : ''}">${sbFmt(ratio)}${over ? ' ⚠' : ''}</td>
            <td style="padding:3px 10px; text-align:right;">${isSweep ? sbFmt(uc.predicted_sweep_ratio) : '–'}</td>
        </tr>`;
    }).join('');
    return `<table style="border-collapse:collapse; font-size:0.85em;">
        <thead><tr style="color:#475569; text-align:right;">
            <th style="padding:3px 10px; text-align:left;">Locator</th><th style="padding:3px 10px;">runs</th>
            <th style="padding:3px 10px;">κ_eff</th><th style="padding:3px 10px;">κ_eff / κ̄</th>
            <th style="padding:3px 10px;">sweep prediction 1/(2.4 ln(W/Δd))</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <p style="margin:0.4em 0 0; font-size:0.8em; color:#64748b;">κ_eff = −d ln S/dn, least squares of ln S on n where S is between 0.9 and 0.05.
        The bound allows κ_eff/κ̄ up to 1/(1 − 2t/W) = ${sbFmt(floor)} for this t, not 1; ⚠ = above it.</p>`;
}

function sbDetectionTable(uc) {
    const rows = SB_LOCATORS.filter((n) => uc.locators[n]).map((name) => {
        const d = uc.locators[name].detection;
        return `<tr>
            <td style="padding:3px 10px; font-weight:600;">${sbShort(name)}</td>
            <td style="padding:3px 10px; text-align:right;">${sbFmt(d.median, 4)}</td>
            <td style="padding:3px 10px; text-align:right;">${sbFmt(d.p75, 4)}</td>
            <td style="padding:3px 10px; text-align:right;">${(100 * d.capped_share).toFixed(0)} %</td>
            <td style="padding:3px 10px; text-align:right;">${sbFmt(d.median_over_inv_kappa)} · ${sbFmt(d.p75_over_inv_kappa)}</td>
            <td style="padding:3px 10px; text-align:right;">${d.median_over_cells == null ? '–' : `${sbFmt(d.median_over_cells)} · ${sbFmt(d.p75_over_cells)}`}</td>
        </tr>`;
    }).join('');
    return `<table style="border-collapse:collapse; font-size:0.85em;">
        <thead><tr style="color:#475569; text-align:right;">
            <th style="padding:3px 10px; text-align:left;">Locator</th><th style="padding:3px 10px;">median n_d</th>
            <th style="padding:3px 10px;">p75</th><th style="padding:3px 10px;">never (capped)</th>
            <th style="padding:3px 10px;">× κ̄ (median · p75)</th><th style="padding:3px 10px;">÷ W/Δd (median · p75)</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <p style="margin:0.4em 0 0; font-size:0.8em; color:#64748b;">n_d = <code>dip_detected_step</code>; runs that never detected count at the
        budget (${uc.budget}), so median and p75 are lower bounds when the capped share is high. At high SNR (1/κ̄ below W/Δd) compare
        with W/(2Δd), the mean of a uniform dip over W/Δd probe cells.</p>`;
}

function sbHeader(uc) {
    // Rows recorded with another prior (stored stage_prior_info is not 4π²/W²) are stale: say which locators.
    const relabel = Math.abs(uc.snr_label - uc.snr) > 1e-6 * uc.snr
        ? `<p style="margin:0 0 0.6em; font-size:0.85em; color:#92400e;">Run label SNR(${sbFmt(uc.snr_label, 3)}) predates the article's per-line SNR convention: its article SNR is ${sbFmt(uc.snr, 3)} (σ = ${sbFmt(uc.noise_sigma, 3)}). All numbers here use ${sbFmt(uc.snr, 3)}.</p>`
        : '';
    const staleNames = Object.entries(uc.locators).filter(([, l]) => l.prior_info_match_share < 1).map(([n]) => sbShort(n));
    const stale = staleNames.length
        ? `<p style="margin:0 0 0.6em; font-size:0.85em; color:#92400e;">Stored prior information of ${staleNames.join(', ')} differs from 4π²/W²: those rows were recorded with another prior (a re-run refreshes them); the search metrics here do not use it.</p>`
        : '';
    const reach = uc.reach
        ? `Δd = ${sbMHz(uc.reach.dd_phys)} (${uc.reach.kind}), W/Δd = ${sbFmt(uc.cells)}`
        : 'no single-measurement reach (√k·SNR ≤ z)';
    const regime = uc.regime ? ` · regime: <b>${uc.regime}</b> (1/κ̄ ${uc.regime === 'finite-reach' ? '<' : '≥'} W/Δd)` : '';
    return `<p style="margin:0 0 0.6em; font-size:0.85em; color:#475569;">
        W = ${sbMHz(uc.width_phys)} (the zeeman_shift range the generator draws) · Γ = ${sbMHz(uc.fwhm_phys)} · SNR = ${sbFmt(uc.snr, 3)}
        (run label ${sbFmt(uc.snr_label, 3)}) · σ = ${sbFmt(uc.noise_sigma, 3)} · k = ${uc.shots} ·
        κ̄ = <b>${sbFmt(uc.kappa_bar)}</b> /meas (1/κ̄ = ${sbFmt(uc.inv_kappa_bar)}; Lorentzian closed form ${sbFmt(uc.kappa_lorentzian)}) ·
        ${reach}${regime} · I<sub>max</sub> = ${sbFmt(uc.fisher_max)} Hz⁻²</p>${relabel}${stale}`;
}

async function renderSearchBounds(root, genList, noiseList, colorOf) {
    const token = String(Date.now() + Math.random());
    root.dataset.token = token;
    if (!genList.length || !noiseList.length) { root.innerHTML = ''; return; }
    root.innerHTML = '<p style="color:#64748b; padding:1em;">Computing the search bounds from the cache…</p>';
    await ensurePlotly();
    const params = new URLSearchParams({ generator: genList.join(','), noise: noiseList.join(',') });
    let cases;
    try {
        const resp = await fetch(`${window.NVISION_ASSET_PREFIX || ''}api/search-bounds?${params}`, { cache: 'no-store' });
        if (!resp.ok) throw new Error(`/api/search-bounds returned ${resp.status} (needs \`nv serve\`)`);
        cases = await resp.json();
    } catch (err) {
        if (root.dataset.token === token) root.innerHTML = `<p style="color:#721c24; background:#f8d7da; padding:1em; border-radius:6px;">${sbEsc(err.message)}</p>`;
        return;
    }
    if (root.dataset.token !== token) return;   // a newer selection took over
    if (!cases.length) { root.innerHTML = '<p style="color:#94a3b8; padding:1em;">No runs of the search locators for this selection.</p>'; return; }
    root.innerHTML = cases.map((uc, i) => `
        <section style="margin:0 0 2em;">
            <h3 class="uc-title" style="margin:0.6em 0; padding:0.4em 0.8em; font-size:1em; color:#334155; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px;">${sbEsc(uc.generator)} · ${sbEsc(uc.noise)}</h3>
            ${uc.error ? `<p style="color:#92400e; background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:0.6em 1em;">Not judged against the search bounds: ${sbEsc(uc.error)}.</p>` : `
            ${sbHeader(uc)}
            <div style="display:flex; flex-wrap:wrap; gap:1.5em;">
                ${sbCard('Mean squared true error vs the search bounds',
                    'Mean over the runs of (true |estimate − truth|)², in units of the prior variance W²/12, against the finite-reach bound (W−nΔd)³/12W (eq. search-mean), the change-of-measure bound (eq. search-bound) and the elimination model (eq. mean-curve). A run that has stopped keeps its final error; a step is drawn only while at least half of the runs still run. W is the range the generator draws zeeman_shift from. Below a bound curve = a violation of that bound (the finite-reach curve needs a single-measurement reach, so it is absent at low SNR).',
                    `sb-mse-${i}`, 420)}
                ${sbCard('Survival: share of runs still wrong',
                    'S(n) = share of runs whose true error exceeds t (one HWHM). The dashed curve is the lower bound on S that the change-of-measure argument gives, exp(−(nκ̄+ln 2)/(1−2t/W)): a locator under it would violate the bound. The slope of ln S between S = 0.9 and 0.05 is κ_eff.',
                    `sb-surv-${i}`, 420)}
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:2em; margin-top:1em;">
                <div style="overflow-x:auto;"><h4 style="margin:0 0 0.3em;">Search efficiency κ_eff/κ̄</h4>${sbKappaTable(uc)}</div>
                <div style="overflow-x:auto;"><h4 style="margin:0 0 0.3em;">Steps until a dip is detected</h4>${sbDetectionTable(uc)}</div>
            </div>`}
        </section>`).join('');
    cases.forEach((uc, i) => {
        if (uc.error) return;
        sbPlotMse(document.getElementById(`sb-mse-${i}`), uc, colorOf);
        sbPlotSurvival(document.getElementById(`sb-surv-${i}`), uc, colorOf);
    });
}

window.renderSearchBounds = renderSearchBounds;
window.SB_MAX_USE_CASES = SB_MAX_USE_CASES;
