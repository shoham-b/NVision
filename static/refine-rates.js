// "Refine & speedup" section of the Information tab: what Bayesian-SBED gains per measurement after it found the dip,
// against a drive-frequency sweep (articles/sections/theory.tex, Sec. speedup; docs/evaluation_metrics.md).
//
// Everything is computed server-side by /api/refine-rates (nvision/metrics/refine_rates*.py) from the cache; this file
// only draws it. Entry point: window.NVRefineRates.render(container, generators, noises), called by app.js's
// renderStagesTab with the current selection. Needs the live server (nv serve): a static export has no API.
(function () {
    const SBED = 'Bayesian-SBED';
    const SWEEPS = ['SimpleSweep-2perFWHM', 'SimpleSweep-3perFWHM', 'SimpleSweep-4perFWHM'];
    const COLORS = { [SBED]: '#1e90ff', 'SimpleSweep-2perFWHM': '#94a3b8', 'SimpleSweep-3perFWHM': '#64748b', 'SimpleSweep-4perFWHM': '#334155' };
    const SNR_COLORS = ['#1e90ff', '#e05c00', '#00a878', '#9b59b6', '#e74c3c', '#f39c12'];
    const label = (s) => s.replace('Bayesian-', '').replace('SimpleSweep', 'Sweep');
    const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const fmt = (v, d = 3) => (v == null || !Number.isFinite(v) ? '–' : Math.abs(v) >= 1000 ? v.toFixed(0) : v.toPrecision(d));
    const pct = (v) => (v == null ? '–' : `${Math.round(100 * v)}%`);
    const median = (xs) => {
        const v = xs.filter((x) => x != null && Number.isFinite(x)).sort((a, b) => a - b);
        if (!v.length) return null;
        const m = v.length >> 1;
        return v.length % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
    };
    const CARD = 'flex:1 1 46%; min-width:420px; background:var(--surface,#fff); border:1px solid var(--slate-100,#f1f5f9); border-radius:8px; padding:10px;';
    const help = (t) => `<span tabindex="0" class="help-icon-visible" title="${esc(t)}">?</span>`;
    const card = (title, tip, id, h = 400) =>
        `<div style="${CARD}"><h3 style="margin-left:10px;">${title} ${help(tip)}</h3><div id="${id}" style="height:${h}px; width:100%;"></div></div>`;
    const ucName = (u) => `${u.generator} · ${u.noise}`;
    const layout = (x, y, extra = {}) => ({
        margin: { l: 60, r: 20, t: 10, b: 60 }, font: { size: 11 }, hovermode: 'closest',
        xaxis: { title: { text: x, font: { size: 11 } }, gridcolor: '#eef2f7' },
        yaxis: { title: { text: y, font: { size: 11 } }, gridcolor: '#eef2f7' },
        legend: { orientation: 'h', yanchor: 'top', y: -0.2, xanchor: 'left', x: 0, font: { size: 9 } },
        ...extra,
    });
    const plot = (id, traces, lay) => Plotly.react(document.getElementById(id), traces, lay, { displayModeBar: false, responsive: true });
    const note = (id, text) => {
        const el = document.getElementById(id);
        if (el) { Plotly.purge(el); el.innerHTML = `<p style="color:#94a3b8;padding:1em;font-size:0.9em;">${esc(text)}</p>`; }
    };
    const locs = (u) => [SBED, ...SWEEPS].filter((s) => u.locators[s]);
    const bestSweepQ = (u, key) => {
        const v = SWEEPS.map((s) => u.locators[s]).filter(Boolean).map((d) => key(d)).filter((x) => x != null && x > 0);
        return v.length ? Math.max(...v) : null;
    };

    // ---- summary table: median over the use cases of each use case's own number (rule 7) ----
    function renderTable(box, cases) {
        const th = (t, tip) => `<th style="padding:2px 8px;text-align:right" title="${esc(tip)}">${t}</th>`;
        const td = (v) => `<td style="padding:2px 8px;text-align:right">${v}</td>`;
        const rows = [SBED, ...SWEEPS].filter((s) => cases.some((u) => u.locators[s])).map((s) => {
            const ds = cases.map((u) => u.locators[s]).filter(Boolean);
            const fits = ds.map((d) => d.fit).filter(Boolean);
            return `<tr><td style="padding:2px 12px 2px 0">${label(s)}</td>` +
                td(`${median(ds.map((d) => d.n_place))} / ${median(ds.map((d) => d.n_runs))}`) +
                td(fmt(median(ds.map((d) => d.q_place_median)))) +
                td(fmt(median(fits.map((f) => f.q)))) +
                td(fmt(median(ds.map((d) => d.q_err_over_place)), 2)) +
                td(fmt(median(fits.map((f) => f.n0)), 3)) +
                td(fmt(median(fits.map((f) => f.r2)), 2)) +
                td(pct(median(ds.map((d) => d.capped_share)))) + '</tr>';
        }).join('');
        const th0 = median(cases.map((u) => u.theory.q_sweep_lorentzian)), thx = median(cases.map((u) => u.theory.q_sweep_exact));
        box.innerHTML = '<table class="metric-table" style="border-collapse:collapse;font-size:0.85em;"><thead><tr>' +
            '<th style="text-align:left;padding:2px 12px 2px 0">Locator</th>' +
            th('Runs with positions / runs', 'Repeats whose measurement positions are known (q_place) over all repeats (q_err). Fixed-budget runs store every position; ordinary SBED runs only the first NVISION_GRAPH_REPEATS repeats of a combo.') +
            th('q_place', 'Mean oracle efficiency η of the measurements after dip_detected_step: how close to the steepest slope the locator measured (1 = every measurement at it). Per run, then the median over runs, then the median over use cases.') +
            th('q_err', 'Slope of 1 / mean(err²) over the runs, in units of I_max (stage_fisher_star): the information the ESTIMATE gained per measurement. Median over use cases.') +
            th('q_err / q_place', 'What the particle filter keeps of the information the placement offered. 1 = nothing lost.') +
            th('n₀', 'Intercept of the same line: 1 / mean(err²) = q · I_max · (n − n₀). Measurements lost before the steady growth began.') +
            th('R²', 'How straight 1 / mean(err²) is over the fitted window. Low = the model of a constant rate does not describe this locator there; do not read q_err and n₀ off it.') +
            th('Capped', 'Share of runs whose claimed uncertainty never met the convergence rule within the budget.') +
            `</tr></thead><tbody>${rows}</tbody></table>` +
            `<p style="font-size:0.8em;color:#475569;margin:0.4em 0 0;">Theory for a sweep, q_sweep = I_sweep / I_max: Lorentzian formula 8π/27 · Γ/W = <b>${fmt(th0)}</b>; exact for these line shapes, mean(g²) / max(g²) over the window = <b>${fmt(thx)}</b> (the two differ for a hyperfine triplet or a Voigt line). One shot per measurement for every locator, so the cost axis (measurements) is also shots.</p>`;
    }

    // ---- cost with the line's prediction, one use case ----
    function renderCost(id, u) {
        const traces = [];
        for (const s of locs(u)) {
            const d = u.locators[s], color = COLORS[s];
            traces.push({
                x: u.gain, y: d.cost.t_median, mode: 'lines+markers', name: label(s), legendgroup: s, line: { color, width: 2 }, marker: { color, size: 5 },
                customdata: d.cost.share.map((sh, i) => [sh, d.cost.t_p75[i]]),
                hovertemplate: `gain %{x:.3g}<br>%{y:.4g} measurements (median)<br>p75 %{customdata[1]:.4g}<br>reached by %{customdata[0]:.0%} of the runs<extra>${label(s)}</extra>`,
            });
            const f = d.fit;
            if (f && f.q > 0 && Number.isFinite(f.n0)) {
                traces.push({
                    x: u.gain, y: u.tau_phys.map((t) => f.n0 + 1 / (t * t * f.q * u.i_max)), mode: 'lines', legendgroup: s, showlegend: false,
                    line: { color, width: 1.5, dash: 'dot' }, hovertemplate: `n₀ + τ⁻² / (q I_max) = ${fmt(f.n0)} + τ⁻² / (${fmt(f.q)} I_max)<extra>${label(s)} prediction</extra>`,
                });
            }
        }
        plot(id, traces, layout('Precision gain over the prior, σ₀ / τ', 'Measurements to reach τ', { xaxis: { ...layout('', '').xaxis, type: 'log', title: { text: 'Precision gain over the prior, σ₀ / τ', font: { size: 11 } } }, yaxis: { ...layout('', '').yaxis, type: 'log', title: { text: 'Measurements to reach τ (median)', font: { size: 11 } } } }));
    }

    // ---- ratio of the costs, T_sweep / T_SBED, per target ----
    function renderRatio(id, cases) {
        const withRatio = cases.filter((u) => u.speedup && u.speedup.ratio.some((v) => v != null));
        if (!withRatio.length) { note(id, 'No use case has Bayesian-SBED and a sweep reaching a target in at least half of their runs.'); return; }
        const gain = withRatio[0].gain;
        const traces = withRatio.length > 1 ? withRatio.map((u) => ({
            x: gain, y: u.speedup.ratio, mode: 'lines', line: { color: '#94a3b8', width: 1 }, opacity: 0.5, showlegend: false, name: ucName(u),
            hovertemplate: `${esc(ucName(u))}<br>gain %{x:.3g}: %{y:.3g}×<extra></extra>`,
        })) : [];
        const med = gain.map((_, i) => {
            const v = withRatio.map((u) => u.speedup.ratio[i]).filter((x) => x != null);
            return v.length >= Math.max(1, Math.ceil(withRatio.length / 2)) ? median(v) : null;
        });
        traces.push({ x: gain, y: med, mode: 'lines+markers', name: 'T_sweep / T_SBED (best sweep spacing; median over use cases)', line: { color: COLORS[SBED], width: 3 }, marker: { size: 5 }, hovertemplate: 'gain %{x:.3g}: %{y:.3g}×<extra></extra>' });
        const rExact = median(withRatio.map((u) => u.theory.r_inf_exact)), rLor = median(withRatio.map((u) => u.theory.r_inf_lorentzian));
        const rMeas = median(withRatio.map((u) => u.speedup.r_inf_err));
        const line = (y, color, name, dash) => ({ x: [gain[0], gain[gain.length - 1]], y: [y, y], mode: 'lines', name, line: { color, width: 1.5, dash }, hovertemplate: `${name}: %{y:.3g}<extra></extra>` });
        if (rExact) traces.push(line(rExact, '#475569', 'R∞ theory, exact line shape', 'dash'));
        if (rLor) traces.push(line(rLor, '#475569', 'R∞ theory, Lorentzian 27/(8π) W/Γ', 'dot'));
        if (rMeas) traces.push(line(rMeas, '#e05c00', 'R∞ measured, q_err ratio', 'dashdot'));
        plot(id, traces, layout('Precision gain over the prior, σ₀ / τ', 'Cost of the best sweep ÷ cost of SBED', { xaxis: { ...layout('', '').xaxis, type: 'log', title: { text: 'Precision gain over the prior, σ₀ / τ', font: { size: 11 } } }, yaxis: { ...layout('', '').yaxis, type: 'log', title: { text: 'T_sweep(τ) / T_SBED(τ)', font: { size: 11 } } } }));
    }

    // ---- R_inf vs W / Gamma, one colour per SNR ----
    function renderRinf(id, cases) {
        const pts = cases.filter((u) => u.speedup && u.speedup.r_inf_err).map((u) => ({ u, x: u.window_over_fwhm }));
        const place = cases.filter((u) => u.speedup && u.speedup.r_inf_place).map((u) => ({ u, x: u.window_over_fwhm }));
        if (!pts.length && !place.length) { note(id, 'No use case has a positive rate fit for both SBED and a sweep.'); return; }
        const snrs = [...new Set(cases.map((u) => u.snr))].sort((a, b) => (a ?? 0) - (b ?? 0));
        const traces = [];
        snrs.forEach((snr, i) => {
            const color = SNR_COLORS[i % SNR_COLORS.length], name = snr == null ? 'SNR n/a' : `SNR ${snr}`;
            const own = pts.filter((p) => p.u.snr === snr);
            // one point per (SNR, W/Γ): the median over the use cases sharing them (other axes: line shape, isotope)
            const byX = new Map();
            own.forEach((p) => { const k = p.x.toFixed(3); if (!byX.has(k)) byX.set(k, []); byX.get(k).push(p.u.speedup.r_inf_err); });
            const xs = [...byX.keys()].map(Number).sort((a, b) => a - b);
            traces.push({ x: own.map((p) => p.x), y: own.map((p) => p.u.speedup.r_inf_err), mode: 'markers', name, legendgroup: name, marker: { color, size: 7, opacity: 0.45 }, text: own.map((p) => ucName(p.u)), hovertemplate: '%{text}<br>W/Γ %{x:.3g}: R∞ %{y:.3g}<extra></extra>' });
            traces.push({ x: xs, y: xs.map((x) => median(byX.get(x.toFixed(3)))), mode: 'lines', name: `${name} (median)`, legendgroup: name, showlegend: false, line: { color, width: 2 }, hoverinfo: 'skip' });
            const pl = place.filter((p) => p.u.snr === snr);
            traces.push({ x: pl.map((p) => p.x), y: pl.map((p) => p.u.speedup.r_inf_place), mode: 'markers', name: `${name} placement`, legendgroup: name, showlegend: false, marker: { color, size: 8, symbol: 'circle-open' }, text: pl.map((p) => ucName(p.u)), hovertemplate: '%{text}<br>W/Γ %{x:.3g}: q_place ratio %{y:.3g}<extra></extra>' });
        });
        const xmax = Math.max(...cases.map((u) => u.window_over_fwhm)), xmin = Math.min(...cases.map((u) => u.window_over_fwhm));
        traces.push({ x: [xmin, xmax], y: [27 / (8 * Math.PI) * xmin, 27 / (8 * Math.PI) * xmax], mode: 'lines', name: 'Theory 27/(8π) · W/Γ', line: { color: '#475569', width: 1.5, dash: 'dash' }, hovertemplate: 'theory %{y:.3g}<extra></extra>' });
        plot(id, traces, layout('W / Γ (drive window over the single-line FWHM)', 'R∞ = q(SBED) / q(best sweep)', { xaxis: { ...layout('', '').xaxis, type: 'log', title: { text: 'W / Γ (drive window over the single-line FWHM)', font: { size: 11 } } }, yaxis: { ...layout('', '').yaxis, type: 'log', title: { text: 'R∞ = q(SBED) / q(best sweep); solid = q_err, open = q_place', font: { size: 11 } } } }));
    }

    // ---- where the measurements fell relative to the true line centre ----
    function renderHist(idBefore, idAfter, u) {
        const edges = u.hist_edges, centers = edges.slice(0, -1).map((e, i) => 0.5 * (e + edges[i + 1]));
        const s = u.theory.slope_offset_gamma, peak = u.theory.peak_offset_gamma;
        const shapes = [-s, s].map((x) => ({ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: '#475569', width: 1, dash: 'dash' } }));
        const note1 = (part, id) => {
            const traces = [];
            for (const l of locs(u)) {
                const h = u.locators[l].hist[part];
                if (!h) continue;
                traces.push({ x: centers, y: h.fraction.map((f) => f / (edges[1] - edges[0])), mode: 'lines', line: { color: COLORS[l], width: 2, shape: 'hv' }, name: `${label(l)} (${h.n_runs} runs, ${pct(h.outside)} beyond ±8γ)`, hovertemplate: `${label(l)}<br>offset %{x:.2f} γ: %{y:.3f} per γ<extra></extra>` });
            }
            if (!traces.length) { note(id, `No ${part} data for this use case.`); return; }
            plot(id, traces, layout('drive_freq − nearest true line centre (units of the true HWHM γ)', 'Share of measurements per γ', { shapes, annotations: [{ x: s, y: 1, xref: 'x', yref: 'paper', text: '±1/√3', showarrow: false, yanchor: 'bottom', font: { size: 9 } }, ...(peak ? [{ x: peak, y: 0.93, xref: 'x', yref: 'paper', text: `steepest: ${peak.toFixed(2)}`, showarrow: false, font: { size: 9, color: '#e05c00' } }] : [])], xaxis: { ...layout('', '').xaxis, range: [-8, 8], title: { text: 'drive_freq − nearest true line centre (units of the true HWHM γ)', font: { size: 11 } } } }));
        };
        note1('before', idBefore);
        note1('after', idAfter);
    }

    async function render(container, generators, noises) {
        if (!container) return;
        const gen = [...new Set([].concat(generators).filter(Boolean))].join(','), noise = [...new Set([].concat(noises).filter(Boolean))].join(',');
        container.innerHTML = '<h2 style="margin:1.2em 0 0.4em;">Refine &amp; speedup</h2><p style="color:#94a3b8;font-size:0.9em;">Rebuilding the runs from the cache (the first view of a selection takes a while)…</p>';
        let data;
        try {
            const resp = await fetch(`${window.NVISION_ASSET_PREFIX || ''}api/refine-rates?generator=${encodeURIComponent(gen)}&noise=${encodeURIComponent(noise)}`, { cache: 'no-store' });
            if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
            data = await resp.json();
            await ensurePlotly();
        } catch (err) {
            container.innerHTML = `<h2 style="margin:1.2em 0 0.4em;">Refine &amp; speedup</h2><p style="color:#721c24;font-size:0.9em;">Not available: ${esc(err.message || err)}. This section needs the live server (<code>nv serve</code>), not a static export.</p>`;
            return;
        }
        const cases = data.use_cases;
        const errors = [...data.failed.flatMap((f) => f.errors.map((e) => ({ ...e, where: ucName(f) }))), ...cases.flatMap((u) => u.errors.map((e) => ({ ...e, where: ucName(u) })))];
        const errHtml = errors.length ? `<details style="margin:0.5em 0;color:#92400e;font-size:0.85em;"><summary>${errors.reduce((a, e) => a + e.count, 0)} repeat(s) could not be rebuilt and are left out</summary><ul>${errors.map((e) => `<li>${esc(e.where)} · ${esc(e.strategy ?? 'use case')} ×${e.count}: ${esc(e.message)}</li>`).join('')}</ul></details>` : '';
        if (!cases.length) {
            container.innerHTML = `<h2 style="margin:1.2em 0 0.4em;">Refine &amp; speedup</h2><p style="color:#94a3b8;font-size:0.9em;">No use case of the selection has Bayesian-SBED and SimpleSweep runs that can be rebuilt (fixed-drive signals only).</p>${errHtml}`;
            return;
        }
        container.innerHTML = '<h2 style="margin:1.2em 0 0.4em;">Refine &amp; speedup ' +
            help('After the dip is found, an adaptive locator should gain about I_max per measurement (q → 1) by measuring at the steepest slope, a sweep only I_sweep = q_sweep · I_max (q_sweep = 8π/27 · Γ/W for a single Lorentzian), so the speedup of a precise target is R∞ = 27/(8π) · W/Γ, independent of SNR. q_place is where the locator measured (exact, from the true signal); q_err is what the estimate gained (the slope of 1 / mean(err²) in units of I_max). Their gap is the particle filter\'s loss. Every run measures the whole budget, so the error is known at every n.') + '</h2>' +
            `<p style="margin:0 0 0.6em;font-size:0.85em;color:#475569;">Every run measures the whole budget; the line is fitted over the late part of it, after the dip. Median over ${cases.length} use case${cases.length > 1 ? 's' : ''}; cost in measurements (one shot each).</p>` +
            errHtml +
            '<div id="rr-table" style="margin:0.6em 0; overflow-x:auto;"></div>' +
            (cases.length > 1 ? `<label style="font-size:0.85em;">Use case for the cost and histogram charts: <select id="rr-case" class="control-select">${cases.map((u, i) => `<option value="${i}">${esc(ucName(u))}</option>`).join('')}</select></label>` : '') +
            '<div style="display:flex; flex-wrap:wrap; gap:1.5em; margin-top:0.8em;">' +
            card('Cost to reach a precision, with the line\'s prediction', 'Median measurements until the true error first stays below τ (solid), against the precision gain over the prior σ₀/τ. Dotted: n₀ + τ⁻² / (q_err · I_max) from the fitted line. The measured cost is a median over single runs, the prediction the mean-square error reaching τ², so they agree only roughly. The share of runs that got there is in the hover; runs that never get there are censored, not counted.', 'rr-cost') +
            card('Speedup of the cost, T_sweep / T_SBED', 'Cost of the best sweep spacing at each target over the cost of SBED. It should approach R∞ (dashed) for a precise target. Thin lines: each use case; thick: the median (needs at least half the use cases). A target counts for a locator only where at least half its runs reached it.', 'rr-ratio') + '</div>' +
            '<div style="display:flex; flex-wrap:wrap; gap:1.5em; margin-top:1em;">' +
            card('R∞ against W / Γ', 'Measured speedup of the information rate, q_err(SBED) / q_err(best sweep) (solid dots) and the placement ratio q_place(SBED) / q_place(best sweep) (open), one colour per SNR, against the theory 27/(8π) · W/Γ for a single Lorentzian (dashed). Theory says the points of every SNR lie on one line. Triplets and Voigt lines move the theory line; see the table for the exact values.', 'rr-rinf') +
            card('Where SBED measured (before the dip is detected)', 'Histogram of drive_freq relative to the nearest TRUE line centre, in units of the true HWHM γ, measurements up to dip_detected_step. Share of measurements per γ; the share beyond ±8γ is in the legend.', 'rr-hist-before') + '</div>' +
            '<div style="display:flex; flex-wrap:wrap; gap:1.5em; margin-top:1em;">' +
            card('Where SBED measured (after the dip is detected)', 'Same histogram for the measurements after dip_detected_step. Dashed lines: ±1/√3 γ, where a Lorentzian\'s squared slope peaks; the orange label is where the exact sensitivity peaks for this use case. Greedy EIG should pile up there; a sweep should be flat.', 'rr-hist-after') + '</div>';
        renderTable(document.getElementById('rr-table'), cases);
        renderRatio('rr-ratio', cases);
        renderRinf('rr-rinf', cases);
        const drawCase = (i) => { renderCost('rr-cost', cases[i]); renderHist('rr-hist-before', 'rr-hist-after', cases[i]); };
        const sel = document.getElementById('rr-case');
        if (sel) sel.addEventListener('change', () => drawCase(Number(sel.value)));
        drawCase(0);
    }

    window.NVRefineRates = { render };
})();
