// Simulation page: one panel per claim of the article, in the order of a run's regimes (search + localize, converge,
// refine) and then calibration. Every number comes from the server; this file only fetches and draws:
//   /api/claims             nvision/metrics/claims_report.py (rms error and claimed σ, (n0, q) line, regimes, coverage)
//   /api/search-bounds      nvision/cli/search_bounds_api.py (change-of-measure bound, elimination model, search MSE)
//   /api/detection-aligned  nvision/metrics/detection_aligned.py (ρ against van Trees restarted at detection, negentropy)
//   /api/refine-rates       nvision/metrics/refine_rates_runs.py (placement q, offset histogram, speedup)
// Default scope: the `article` run group's search grid, the Lorentzian NVCenter-voigt-span<W>-gs0-hfunresolved line at
// every window W. The group's other families are selectable as a Line: the lab mirror (one generator per lab window)
// and tracking (one per Gaussian prior width), where Bayesian-SBED-KnownShape also runs.
// and SNR, Bayesian-SBED against SimpleSweep-optimal. Called by app.js (showScanViewPanels('article')).
// Needs ensurePlotly from plotly-utils.js and daPlotRho / daPlotNegentropy from detection-aligned-view.js.
(function () {
    const SBED = 'Bayesian-SBED';
    const SWEEP = 'SimpleSweep-optimal';
    const KNOWN = 'Bayesian-SBED-KnownShape';
    const LOCATORS = [SBED, KNOWN, SWEEP];
    const DEFAULT_FAMILY = 'NVCenter-voigt-span*-gs0-hfunresolved';
    // A family is a set of generators that differ in one number, the page's axis. First match wins.
    const AXES = [
        { re: /^(.*-span-lab-w)([\d.]+)(MHz-fwhm[\d.]+MHz)$/, name: 'Window', text: (v) => `${v} MHz` },
        { re: /^(.*-hfunresolved-track)([\d.]+)()$/, name: 'Prior σ₀', text: (v) => `${v} FWHM` },
        { re: /^(.*-span)([\d.]+)(-.*)$/, name: 'Window', text: (v) => `span ${v}` },
    ];
    const axisMatch = (gen) => { for (const a of AXES) { const m = a.re.exec(gen); if (m) return { a, m }; } return null; };
    const SNR_RE = /^SNR\(([\d.eE+-]+)\)$/;
    const GAUSS_WITHIN = { 1: 0.6827, 2: 0.9545 };
    const WORKERS = 3;

    const api = (path, params) => {
        const q = Object.entries(params || {}).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
        return `${window.NVISION_ASSET_PREFIX || ''}api/${path}${q ? `?${q}` : ''}`;
    };
    const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
    const colorOf = (s) => token(s === SBED ? 'highlight-1' : s === KNOWN ? 'highlight-3' : 'highlight-2');
    const label = (s) => (s === SBED ? 'SBED' : s === KNOWN ? 'SBED known shape' : s === SWEEP ? 'Sweep k*' : s);
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const fmt = (v, d = 3) => (v == null || !Number.isFinite(v) ? '–' : Number(v).toPrecision(d));
    const median = (xs) => {
        const v = xs.filter((x) => x != null && Number.isFinite(x)).sort((a, b) => a - b);
        if (!v.length) return null;
        const m = Math.floor(v.length / 2);
        return v.length % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
    };
    const spanOf = (gen) => { const x = axisMatch(gen); return x ? Number(x.m[2]) : null; };
    const familyOf = (gen) => { const x = axisMatch(gen); return x ? `${x.m[1]}*${x.m[3]}` : null; };
    const axisOf = (gen) => { const x = axisMatch(gen); return x ? x.a : AXES[AXES.length - 1]; };
    const snrOf = (noise) => { const m = SNR_RE.exec(noise); return m ? Number(m[1]) : null; };

    // ── Fetching: one queue for the whole page, the selected use case first; every response kept for the session ──
    const memo = new Map();     // url -> Promise of the parsed JSON
    const done = new Map();     // url -> the parsed JSON, once it has arrived
    const queue = [];
    let running = 0;
    function fetchJson(url, priority = false) {
        if (memo.has(url)) return memo.get(url);
        const p = new Promise((resolve, reject) => {
            const job = async () => {
                running += 1;
                try {
                    const resp = await fetch(url, { cache: 'no-store' });
                    const body = await resp.json().catch(() => ({}));
                    if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
                    done.set(url, body);
                    resolve(body);
                } catch (err) {
                    memo.delete(url);
                    reject(err);
                } finally {
                    running -= 1;
                    pump();
                }
            };
            if (priority) queue.unshift(job); else queue.push(job);
        });
        memo.set(url, p);
        pump();
        return p;
    }
    function pump() {
        while (running < WORKERS && queue.length) queue.shift()();
    }

    // ── Plot helpers (design system: white plot, chart-grid gridlines, slate reference lines) ──
    function layout(xTitle, yTitle, xType = 'log', yType = 'log') {
        return {
            template: 'plotly_white',
            margin: { t: 30, b: 55, l: 70, r: 15 },
            hovermode: 'closest',
            legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1, font: { size: 10 } },
            xaxis: { title: xTitle, type: xType, gridcolor: token('chart-grid') },
            yaxis: { title: yTitle, type: yType, gridcolor: token('chart-grid') },
        };
    }
    const hline = (y, dash = 'dash') => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: y, y1: y, line: { color: token('slate-500'), width: 1.5, dash } });
    const vline = (x, color, dash = 'dot') => ({ type: 'line', xref: 'x', x0: x, x1: x, yref: 'paper', y0: 0, y1: 1, line: { color, width: 1.5, dash } });
    const theory = (name, dash = 'dash') => ({ name, mode: 'lines', line: { color: token('slate-700'), width: 2, dash } });
    function empty(div, text) {
        div.innerHTML = `<p class="empty-state">${esc(text)}</p>`;
    }
    function plot(div, traces, lay) {
        if (!div) return;
        div.innerHTML = '';
        Plotly.newPlot(div, traces, lay, { responsive: true, displaylogo: false });
    }
    const card = (id, title, help, height = 400) => `
        <div class="chart-card">
            <h3>${title} <span tabindex="0" class="help-icon-visible" title="${esc(help)}">?</span></h3>
            <p class="chart-claim" id="${id}-claim"></p>
            <div id="${id}" style="height:${height}px; width:100%;"><p class="empty-state">Loading…</p></div>
        </div>`;
    const setClaim = (id, html) => { const el = document.getElementById(`${id}-claim`); if (el) el.innerHTML = html; };
    const val = (v) => `<span class="claim-value">${v}</span>`;

    // ── Page state ──
    const state = { family: null, gens: [], noises: [], uc: null, root: null, combos: null };

    function skeleton() {
        return `
        <h2>Simulation — the article's claims <span tabindex="0" class="help-icon-visible" title="Each panel tests one claim of the article on the cached runs, with the theory curve beside the data. A run passes four regimes: search + localize until its own dip test fires, converge until the posterior is a single Gaussian peak (negentropy < 0.05), then refine, where 1/MSE grows at q·I_max. Panels marked 'one use case' follow the Window and SNR pickers; the others span every use case of the line. Lab data: pick a lab as the Study.">?</span></h2>
        <div class="sticky-selectors" style="display:flex; flex-wrap:wrap; align-items:center; gap:0.5em 1.2em; margin:0 0 0.6em;">
            <span class="control-row" style="margin:0;"><span class="control-label">Line:</span>
                <select class="control-select" id="art-family" aria-label="Line shape family"></select></span>
            <span class="control-row" style="margin:0;"><span class="control-label" id="art-axis-label">Window:</span>
                <span class="segmented-control" id="art-window" role="radiogroup" aria-label="Drive window"></span></span>
            <span class="control-row" style="margin:0;"><span class="control-label">SNR:</span>
                <span class="segmented-control" id="art-snr" role="radiogroup" aria-label="SNR"></span></span>
            <span id="art-progress" style="font-size:0.85em; color:var(--slate-600);"></span>
        </div>
        <div id="art-notice"></div>
        <h3 class="group-label">Search + localize</h3>
        <div class="chart-pair">
            ${card('art-overview', 'Error over the run (one use case)', "Root mean square over the runs of the true error of zeeman_shift, in units of the line's FWHM Γ, against the measurement count n. Dotted: the elimination model, the article's mean curve (Eq. mean-curve: a search that clears the window one reach Δd per measurement, then van Trees from each detection). Dashed: the change-of-measure bound, which no strategy can beat during the search. Dash-dot: the data limit (n I_max)^-1/2, reached only by measuring every time at the steepest slope. Vertical lines: SBED's median detection step and median Gaussian onset.")}
            ${card('art-search', 'Search against the bounds (one use case)', 'ρ_search(n) = bound MSE ÷ measured mean squared true error over the runs, during the search. Solid: against the change-of-measure bound (Eq. search-bound), which holds for any strategy, adaptive or not: ρ must stay ≤ 1. Dotted: against the elimination model, a reference rather than a bound. Shown where enough runs are still running; vertical lines mark each locator\'s median detection step.')}
        </div>
        <div class="chart-pair" style="margin-top:1.5em;">
            ${card('art-searchcost', 'Search cost vs window (shared with the Lab page)', 'One point per use case: the median step at which the runs detected the dip (the dip test), against Eq. search-cost, (8/π)(W/Γ)/(k SNR²) with k = 1 shot, Γ the FWHM and SNR = C/σ the article SNR of the use case. A lower bound for any strategy at low SNR. The Lab page draws the same panel from the lab runs.')}
            ${card('art-detectors', 'Two detection definitions on the same runs', "One point per use case and locator: the median step of the lab's detector (the first steep, lasting drop of the claimed σ, located to the stored steps) against the median step of the simulation's dip test, over the runs both detect. On the dashed line the two definitions agree.")}
        </div>
        <h3 class="group-label">Converge</h3>
        <div class="chart-pair">
            ${card('art-rho', 'After detection: ρ against restarted van Trees (one use case)', "Every run is aligned at its own detection (lag k = n − n_d). ρ(k) = mean over runs of the van Trees bound restarted there, 1/(J_τ + k I_max) with J_τ the Fisher information of the run's posterior at detection, ÷ the mean squared true error. ρ ≤ 1 by van Trees; ρ near 1 means the run gains information at the data limit. Dotted: the same with the elimination model's J_d = 12/Δd².")}
            ${card('art-negentropy', 'Posterior negentropy after detection (one use case)', 'Median (band: quartiles) of the posterior negentropy, its distance from a Gaussian of the same variance, against the lag since the run\'s own detection. The posterior has converged to a single Gaussian peak once the negentropy stays below 0.05 (dotted); the diamond marks the median onset.')}
        </div>
        <h3 class="group-label">Refine</h3>
        <div class="chart-pair">
            ${card('art-q', 'Information rate q vs window', "q: 1/MSE grows late in the run as q·I_max·(n − n₀) (Eq. strategy-numbers). Filled: q fitted to the run-averaged true error over the late window (open: that window is not linear, so the late regime is not reached). Crosses: the placement efficiency, the mean I(d)/I_max of the measurements after detection. One point per use case (hover: SNR). Dashed: a sweep spreads its measurements over the window and gets q = 8πΓ/(27W) (Eq. sweep-info).")}
            ${card('art-hist', 'Where measurements fall after detection (one use case)', 'Share of the measurements after each run\'s detection per unit of offset from the nearest true line centre, in units of the true HWHM γ. The information of one measurement peaks on the steepest slopes, at ±γ/√3 (dashed); a sweep spreads its measurements evenly.')}
        </div>
        <div class="chart-pair" style="margin-top:1.5em;">
            ${card('art-speedup', 'Speedup over the sweep vs window', 'Filled: the measured speedup at a precise target, R∞ = q(SBED) ÷ q(best sweep), each q fitted to the late true error. Open: the ratio of the measurements to the most precise target both reach in at least half of their runs. Dotted: the prediction 27/(8π)·q·W/Γ ≈ 1.07 q W/Γ with q SBED\'s own rate in that use case (Eq. speedup times q); dashed: the same at q = 1.')}
            ${card('art-calib', 'True error vs claimed σ over the run (one use case)', 'Root mean square of the true error over the runs ÷ root mean square of the claimed σ, against n. 1 (dashed) = the claimed σ is honest on average; above 1 = overconfident. Before the posterior is Gaussian the claimed σ is not a 68 % interval, so read the curve from the Gaussian onset (vertical lines) on.')}
        </div>
        <div class="chart-pair" style="margin-top:1.5em;">
            ${card('art-rates', 'Information per measurement after detection (shared with the Lab page)', "Circles: the claimed rate q̂, the slope of 1/⟨σ̂²⟩ per measurement in units of I_max over the second half of the lags after each run's detection; it needs only the claimed σ, so the lab reads the same number. Crosses: the placement efficiency, the mean I(d)/I_max of the measurements after detection. Dashed: the sweep rate 8πΓ/(27W).")}
            ${card('art-spread', 'Claimed σ vs the spread of the runs (shared with the Lab page)', 'One point per use case and locator: the rms claimed σ at the end of the run against the rms true error over the runs (the prior ensemble). On the dashed line the claimed σ is honest on average. The Lab page draws the same panel with the scatter of repeated runs in place of the true error.')}
        </div>
        <h3 class="group-label">Calibration</h3>
        <div class="chart-pair">
            ${card('art-coverage', 'Coverage at the end of the run vs window', 'Share of runs whose final true error lies within 1 claimed σ (circles) and 2 claimed σ (diamonds), from the signed z = error / σ the runs store. A calibrated Gaussian posterior gives 0.683 and 0.954 (dashed). One point per use case; the accuracy at the end is not itself a quality metric (it reflects the stop target).')}
            ${card('art-pit', 'Coverage by the posterior\'s own quantiles (PIT)', 'Share of runs whose truth lies in the central 68 % and 95 % of their own particle posterior (probability integral transform), which holds also before the posterior is Gaussian. Needs runs recorded with the PIT series (pit_final).')}
        </div>`;
    }

    // ── Selection controls ──
    function segmented(el, items, current, onPick) {
        el.innerHTML = items.map(([value, text]) => `<button type="button" role="radio" aria-checked="${value === current}" class="${value === current ? 'is-active' : ''}" tabindex="${value === current ? 0 : -1}" data-value="${esc(value)}">${esc(text)}</button>`).join('');
        el.onclick = (e) => {
            const btn = e.target.closest('button');
            if (btn) onPick(btn.dataset.value);
        };
    }

    function renderControls() {
        const fam = document.getElementById('art-family');
        const families = [...new Set(state.combos.map((c) => familyOf(c.generator)).filter(Boolean))].sort();
        fam.innerHTML = families.map((f) => `<option value="${esc(f)}" ${f === state.family ? 'selected' : ''}>${esc(f)}</option>`).join('');
        fam.onchange = () => { selectFamily(fam.value); };
        const [g, n] = state.uc;
        document.getElementById('art-axis-label').textContent = `${axisOf(g).name}:`;
        segmented(document.getElementById('art-window'), state.gens.map((x) => [x, axisOf(x).text(spanOf(x))]), g, (v) => { state.uc = [v, state.uc[1]]; renderControls(); loadUseCase(); });
        segmented(document.getElementById('art-snr'), state.noises.map((x) => [x, String(snrOf(x) ?? x)]), n, (v) => { state.uc = [state.uc[0], v]; renderControls(); loadUseCase(); });
    }

    function selectFamily(family) {
        state.family = family;
        const ofFamily = state.combos.filter((c) => familyOf(c.generator) === family && LOCATORS.includes(c.strategy));
        state.gens = [...new Set(ofFamily.map((c) => c.generator))].sort((a, b) => spanOf(a) - spanOf(b));
        state.noises = [...new Set(ofFamily.map((c) => c.noise))].sort((a, b) => (snrOf(a) ?? 0) - (snrOf(b) ?? 0));
        if (!state.gens.length) {
            document.getElementById('art-notice').innerHTML = `<p class="notice-amber">No Bayesian-SBED or SimpleSweep-optimal runs for ${esc(family)}.</p>`;
            return;
        }
        // Default use case: the middle window at the middle SNR.
        state.uc = [state.gens[Math.floor((state.gens.length - 1) / 2)], state.noises[Math.floor((state.noises.length - 1) / 2)]];
        renderControls();
        loadUseCase();
        loadAcross();
    }

    // ── Data per use case ──
    const claimsUrl = (g) => api('claims', { generator: g, noise: state.noises.join(',') });
    const ratesUrl = (g) => api('refine-rates', { generator: g, noise: state.noises.join(',') });
    const boundsUrl = (g) => api('search-bounds', { generator: g, noise: state.noises.join(',') });
    const alignedUrl = (g) => api('detection-aligned', { generator: g, noise: state.noises.join(',') });

    async function claimsOf(g, n, priority = false) {
        const rows = await fetchJson(claimsUrl(g), priority);
        const out = {};
        rows.filter((r) => r.noise === n && LOCATORS.includes(r.strategy)).forEach((r) => { out[r.strategy] = r; });
        return out;
    }
    async function ratesOf(g, n, priority = false) {
        const body = await fetchJson(ratesUrl(g), priority);
        return (body.use_cases || []).find((u) => u.generator === g && u.noise === n) || null;
    }

    function loadUseCase() {
        const [g, n] = state.uc;
        const here = () => state.uc[0] === g && state.uc[1] === n;
        ['art-overview', 'art-search', 'art-rho', 'art-negentropy', 'art-hist', 'art-calib'].forEach((id) => {
            const div = document.getElementById(id);
            if (div) div.innerHTML = '<p class="empty-state">Loading (the first view of a use case reads every run, up to a minute)…</p>';
            setClaim(id, '');
        });
        const sb = fetchJson(boundsUrl(g), true).then((r) => r.find((u) => u.generator === g && u.noise === n));
        const da = fetchJson(alignedUrl(g), true);
        const cl = claimsOf(g, n, true);
        Promise.all([sb, cl]).then(([u, c]) => { if (here()) { drawOverview(u, c); drawSearch(u, c); drawCalibration(c); } })
            .catch((err) => { if (here()) ['art-overview', 'art-search'].forEach((id) => empty(document.getElementById(id), err.message)); });
        cl.catch((err) => { if (here()) empty(document.getElementById('art-calib'), err.message); });
        Promise.all([da, cl.catch(() => ({}))]).then(([rows, c]) => { if (here()) drawConverge(rows.filter((r) => r.noise === n && r.generator === g), Object.values(c).some((r) => r && r.prior_kind === 'gaussian')); })
            .catch((err) => { if (here()) ['art-rho', 'art-negentropy'].forEach((id) => empty(document.getElementById(id), err.message)); });
        ratesOf(g, n, true).then((u) => { if (here()) drawHist(u); })
            .catch((err) => { if (here()) empty(document.getElementById('art-hist'), err.message); });
    }

    // Every use case of the line, for the panels against W: claims and refine rates per window (all SNRs in one request).
    function loadAcross() {
        const family = state.family;
        const gens = [state.uc[0], ...state.gens.filter((x) => x !== state.uc[0])];
        let count = 0;
        const errors = [];
        const total = 4 * gens.length;
        const progress = () => {
            const el = document.getElementById('art-progress');
            if (!el) return;
            const failed = errors.length ? ` · ${errors.length} failed: ${errors.slice(0, 2).join('; ')}` : '';
            el.textContent = (count < total ? `Loading every use case: ${count}/${total} requests…` : `${state.gens.length} windows × ${state.noises.length} SNRs loaded.`) + failed;
        };
        progress();
        const redraw = () => { if (state.family === family) drawAcross(); };
        const track = (url) => fetchJson(url).catch((err) => { errors.push(err.message); }).then(() => { count += 1; progress(); redraw(); });
        gens.forEach((g) => { track(claimsUrl(g)); track(ratesUrl(g)); track(boundsUrl(g)); track(alignedUrl(g)); });
    }

    function drawAcross() {
        const claimRows = [], rateCases = [], failures = [], boundCases = [], alignedRows = [];
        for (const g of state.gens) {
            const b = done.get(boundsUrl(g));
            if (b) boundCases.push(...b);
            const a = done.get(alignedUrl(g));
            if (a) alignedRows.push(...a.filter((r) => LOCATORS.includes(r.strategy)));
            const c = done.get(claimsUrl(g));
            if (c) claimRows.push(...c.filter((r) => LOCATORS.includes(r.strategy)));
            const r = done.get(ratesUrl(g));
            if (r) {
                rateCases.push(...(r.use_cases || []));
                failures.push(...(r.failed || []));
            }
        }
        drawQ(claimRows, rateCases);
        drawSpeedup(rateCases, claimRows);
        drawCoverage(claimRows);
        drawPit(claimRows);
        drawShared(claimRows, rateCases, boundCases, alignedRows);
        const note = document.getElementById('art-notice');
        const msgs = failures.flatMap((f) => (f.errors || []).map((e) => `${f.generator} ${f.noise}: ${e.message}`));
        if (note) note.innerHTML = msgs.length ? `<p class="notice-amber">Refine rates could not rebuild ${msgs.length} use case(s): ${esc(msgs.slice(0, 3).join('; '))}${msgs.length > 3 ? '…' : ''}</p>` : '';
    }

    // ── Search + localize ──
    function drawOverview(u, c) {
        const div = document.getElementById('art-overview');
        const s = c[SBED] || c[SWEEP];
        if (!s || s.unavailable) { empty(div, (s && s.unavailable) || 'No claims for this use case.'); return; }
        const gamma = s.fwhm_phys, imax = s.i_max;
        const traces = [];
        for (const name of LOCATORS) {
            const r = c[name];
            if (!r || !r.rms_curve) continue;
            traces.push({ x: r.rms_curve.n, y: r.rms_curve.error.map((v) => v / gamma), mode: 'lines', name: `${label(name)} rms true error`, line: { color: colorOf(name), width: 2.5 } });
        }
        if (u) {
            const sq = (arr) => arr.map((v) => (v == null ? null : Math.sqrt(v) / gamma));
            if (u.curves.elimination_model) traces.push({ x: u.n, y: sq(u.curves.elimination_model), ...theory('elimination model (Eq. mean-curve)', 'dot') });
            traces.push({ x: u.n, y: sq(u.curves.change_of_measure), ...theory('change-of-measure bound (Eq. search-bound)', 'dash') });
        }
        const n = s.rms_curve ? s.rms_curve.n : [1, s.max_n];
        traces.push({ x: n, y: n.map((k) => 1 / Math.sqrt(k * imax) / gamma), ...theory('data limit (n I_max)^-1/2', 'dashdot') });
        const tracking = s.prior_kind === 'gaussian';
        // The bound comes from the server: it follows the line's random walk during the run.
        const vt = s.van_trees_from_start;
        if (tracking && vt) traces.push({ x: vt.n, y: vt.bound_sigma.map((v) => v / gamma), name: 'moving-line bound from n = 0, J₀ = 1/σ₀² (tracking)', mode: 'lines', line: { color: token('red-700'), width: 2, dash: 'dash' } });
        const lay = layout('measurements n', 'error / Γ');
        const sb = c[SBED];
        lay.shapes = [hline(s.sigma0_phys / gamma, 'dot')];
        if (sb && sb.regimes) {
            if (sb.regimes.detection_step) lay.shapes.push(vline(sb.regimes.detection_step, colorOf(SBED), 'dot'));
            if (sb.regimes.gaussian_onset_step) lay.shapes.push(vline(sb.regimes.gaussian_onset_step, colorOf(SBED), 'dash'));
        }
        lay.yaxis.range = [Math.log10(1 / Math.sqrt(n[n.length - 1] * imax) / gamma) - 0.3, Math.log10(2 * s.sigma0_phys / gamma)];
        plot(div, traces, lay);
        const q = (name) => (c[name] && c[name].fitted_rate ? c[name].fitted_rate.rate : null);
        if (tracking) {
            // rho_vT(n) = bound variance / mean squared true error, <= 1 at every n (server: van_trees_from_start).
            const rho = LOCATORS.filter((name) => c[name] && c[name].van_trees_from_start).map((name) => {
                const v = c[name].van_trees_from_start;
                return `${label(name)} ${val(fmt(v.rho_max, 2))} at n = ${val(fmt(v.rho_max_n, 3))}${v.violated ? ' (violates the bound)' : ''}`;
            });
            setClaim('art-overview', `Tracking (Gaussian prior σ₀ = ${val(fmt(s.sigma0_phys / gamma, 2))} Γ, the prior the truth is drawn around): the line keeps random-walking during the run (step ${val(fmt(s.line_walk_step_phys / gamma, 3))} Γ per measurement, which the locators are told and model), so the bound levels off. ρ = bound variance ÷ MSE ≤ 1 from the first measurement, with J₀ = 1/σ₀². Largest ρ: ${rho.join(', ')}.`);
            return;
        }
        setClaim('art-overview', `Plateau at the prior W/√12 (horizontal), then 1/√n at q × the data limit. q: SBED ${val(fmt(q(SBED), 2))}, sweep ${val(fmt(q(SWEEP), 2))}; SBED detects at n ≈ ${val(fmt(sb && sb.regimes && sb.regimes.detection_step, 3))} and is Gaussian from n ≈ ${val(fmt(sb && sb.regimes && sb.regimes.gaussian_onset_step, 3))}.`);
    }

    function drawSearch(u, c) {
        const div = document.getElementById('art-search');
        if (!u) { empty(div, 'No search-bounds data for this use case.'); return; }
        if (u.error) { empty(div, u.error); return; }
        const traces = [];
        const lay = layout('measurements n', 'ρ_search = bound MSE / measured MSE');
        lay.shapes = [hline(1)];
        const maxes = [];
        for (const name of LOCATORS) {
            const loc = u.locators[name];
            if (!loc) continue;
            const det = c[name] && c[name].regimes ? c[name].regimes.detection_step : null;
            const ratio = (curve) => u.n.map((k, i) => (loc.mse[i] == null || curve[i] == null || !(loc.mse[i] > 0) ? null : curve[i] / loc.mse[i]));
            const rho = ratio(u.curves.change_of_measure);
            traces.push({ x: u.n, y: rho, mode: 'lines', name: `${label(name)} vs change of measure`, line: { color: colorOf(name), width: 2.5 }, connectgaps: false });
            if (u.curves.elimination_model) traces.push({ x: u.n, y: ratio(u.curves.elimination_model), mode: 'lines', name: `${label(name)} vs elimination model`, line: { color: colorOf(name), width: 1.5, dash: 'dot' }, connectgaps: false });
            if (det) {
                lay.shapes.push(vline(det, colorOf(name)));
                const inSearch = rho.filter((v, i) => v != null && u.n[i] <= det);
                if (inSearch.length) maxes.push(`${label(name)} ${val(fmt(Math.max(...inSearch), 2))}`);
            }
        }
        plot(div, traces, lay);
        setClaim('art-search', `No strategy beats the change-of-measure bound during the search (ρ ≤ 1). Largest ρ before detection: ${maxes.join(', ') || '–'}. Regime: ${val(esc(u.regime || '–'))}.`);
    }

    // ── Converge ──
    function drawConverge(rows, tracking) {
        const locs = LOCATORS.map((name) => [name, rows.find((r) => r.strategy === name)]).filter(([, loc]) => loc && !loc.unavailable && loc.lags);
        const rhoDiv = document.getElementById('art-rho'), negDiv = document.getElementById('art-negentropy');
        if (!locs.length) {
            const why = rows.map((r) => `${label(r.strategy)}: ${r.unavailable || 'no data'}`).join('; ');
            empty(rhoDiv, why || 'No runs detected the dip.');
            empty(negDiv, why || 'No runs detected the dip.');
            return;
        }
        rhoDiv.innerHTML = ''; negDiv.innerHTML = '';
        daPlotRho(rhoDiv, locs, colorOf);
        const withG = locs.filter(([, loc]) => loc.gaussian && loc.gaussian.negentropy_by_lag);
        if (withG.length) daPlotNegentropy(negDiv, withG, colorOf, withG[0][1].gaussian.threshold);
        else empty(negDiv, 'These runs store no negentropy series: re-run them.');
        const rhoMax = locs.map(([name, loc]) => `${label(name)} ${val(fmt(loc.rho_van_trees_max, 2))}${loc.bound_violated ? ' (violates the bound)' : ''}`);
        const trackNote = tracking ? ' Tracking is judged by van Trees from n = 0 (error over the run, above): with the shape known the dip test fires late, after the posterior is already Gaussian, so this restart is for reference only.' : '';
        setClaim('art-rho', `Van Trees restarted at each run's detection with J_τ: ρ ≤ 1. Largest ρ: ${rhoMax.join(', ')}.${trackNote}`);
        const onset = withG.map(([name, loc]) => `${label(name)} ${val(fmt(loc.gaussian.onset_lag_median, 3))} (${val(Math.round(100 * loc.gaussian.share_reached) + ' %')} of runs)`);
        setClaim('art-negentropy', `The posterior becomes a single Gaussian peak (negentropy < 0.05). Median onset lag after detection: ${onset.join(', ') || '–'}.`);
    }

    // ── Refine ──
    function drawQ(claimRows, rateCases) {
        const div = document.getElementById('art-q');
        if (!claimRows.length && !rateCases.length) return;
        const traces = [];
        const xs = [];
        for (const name of LOCATORS) {
            const rows = claimRows.filter((r) => r.strategy === name && r.fitted_rate && r.fitted_rate.rate != null && r.window_phys && r.fwhm_phys);
            const x = rows.map((r) => r.window_phys / r.fwhm_phys);
            xs.push(...x);
            traces.push({
                x, y: rows.map((r) => r.fitted_rate.rate), mode: 'markers', name: `${label(name)} q (late true error)`,
                marker: { color: rows.map((r) => (r.fitted_rate.linear ? colorOf(name) : 'rgba(0,0,0,0)')), line: { color: colorOf(name), width: 1.5 }, size: 9, symbol: name === SBED ? 'circle' : 'square' },
                text: rows.map((r) => `${r.noise}${r.fitted_rate.linear ? '' : ' · not linear'}`), hovertemplate: 'W/Γ %{x:.3g}<br>q %{y:.3g}<br>%{text}<extra></extra>',
            });
            const placed = rateCases.filter((u) => u.locators && u.locators[name] && u.locators[name].placement_efficiency_median != null);
            traces.push({
                x: placed.map((u) => u.window_over_fwhm), y: placed.map((u) => u.locators[name].placement_efficiency_median), mode: 'markers', name: `${label(name)} placement ⟨I(d)/I_max⟩`,
                marker: { color: colorOf(name), size: 8, symbol: 'x-thin', line: { color: colorOf(name), width: 2 } },
                text: placed.map((u) => u.noise), hovertemplate: 'W/Γ %{x:.3g}<br>placement %{y:.3g}<br>%{text}<extra></extra>',
            });
            xs.push(...placed.map((u) => u.window_over_fwhm));
        }
        if (!xs.length) { empty(div, 'No use case has a rate fit yet.'); return; }
        const lo = Math.min(...xs) / 1.2, hi = Math.max(...xs) * 1.2;
        const grid = Array.from({ length: 40 }, (_, i) => lo * (hi / lo) ** (i / 39));
        traces.push({ x: grid, y: grid.map((x) => Math.min(1, (8 * Math.PI) / (27 * x))), ...theory('sweep 8πΓ/(27W)') });
        const lay = layout('window W / Γ', 'q = rate / I_max');
        lay.shapes = [hline(1, 'dot')];
        plot(div, traces, lay);
        const qs = claimRows.filter((r) => r.strategy === SBED && r.fitted_rate && r.fitted_rate.linear).map((r) => r.fitted_rate.rate);
        const sweepRatio = claimRows.filter((r) => r.strategy === SWEEP && r.fitted_rate && r.fitted_rate.linear && r.window_phys && r.fwhm_phys)
            .map((r) => r.fitted_rate.rate / ((8 * Math.PI) / (27 * (r.window_phys / r.fwhm_phys))));
        setClaim('art-q', `Article: SBED's q is about 0.5 at every window and SNR; the sweep follows 8πΓ/(27W). SBED q, median over ${qs.length} linear use cases: ${val(fmt(median(qs), 2))}; sweep q ÷ 8πΓ/(27W): ${val(fmt(median(sweepRatio), 2))}.`);
    }

    function drawHist(u) {
        const div = document.getElementById('art-hist');
        if (!u) { empty(div, 'No refine-rates data for this use case.'); return; }
        const edges = u.hist_edges, width = edges[1] - edges[0];
        const centers = edges.slice(0, -1).map((e, i) => 0.5 * (e + edges[i + 1]));
        const traces = [];
        const shares = [];
        for (const name of LOCATORS) {
            const h = u.locators[name] && u.locators[name].hist && u.locators[name].hist.after;
            if (!h) continue;
            traces.push({ x: centers, y: h.fraction.map((f) => f / width), mode: 'lines', line: { color: colorOf(name), width: 2, shape: 'hv' }, name: `${label(name)} (${h.n_runs} runs, ${Math.round(100 * h.outside)} % beyond ±8γ)` });
            const near = h.fraction.reduce((acc, f, i) => acc + (Math.abs(centers[i]) <= 2 ? f : 0), 0);
            shares.push(`${label(name)} ${val(Math.round(100 * near) + ' %')}`);
        }
        if (!traces.length) { empty(div, 'No measurement positions after detection for this use case (SBED needs its scan figures).'); return; }
        const s = u.theory.slope_offset_gamma;
        const lay = layout('drive_freq − nearest true line centre (units of γ)', 'share of measurements per γ', 'linear', 'linear');
        lay.shapes = [vline(-s, token('slate-700'), 'dash'), vline(s, token('slate-700'), 'dash')];
        plot(div, traces, lay);
        setClaim('art-hist', `Share of the measurements after detection within ±2γ of the line: ${shares.join(', ')}.`);
    }

    function drawSpeedup(rateCases, claimRows) {
        const div = document.getElementById('art-speedup');
        const cases = rateCases.filter((u) => u.speedup);
        if (!cases.length) { if (rateCases.length) empty(div, 'No use case has SBED and a sweep with a rate fit.'); return; }
        const qOf = (u) => {
            const r = claimRows.find((c) => c.generator === u.generator && c.noise === u.noise && c.strategy === SBED);
            return r && r.fitted_rate ? r.fitted_rate.rate : null;
        };
        const meas = cases.filter((u) => u.speedup.speedup_fitted);
        const precise = cases.map((u) => ({ u, r: [...(u.speedup.ratio || [])].reverse().find((v) => v != null) })).filter((p) => p.r != null);
        const pred = cases.map((u) => ({ u, q: qOf(u) })).filter((p) => p.q != null && p.q > 0);
        const traces = [
            { x: meas.map((u) => u.window_over_fwhm), y: meas.map((u) => u.speedup.speedup_fitted), mode: 'markers', name: 'measured R∞ = q(SBED)/q(best sweep)',
                marker: { color: colorOf(SBED), size: 9 }, text: meas.map((u) => `${u.noise} · best sweep ${u.speedup.best_sweep_fitted}`), hovertemplate: 'W/Γ %{x:.3g}<br>R∞ %{y:.3g}<br>%{text}<extra></extra>' },
            { x: precise.map((p) => p.u.window_over_fwhm), y: precise.map((p) => p.r), mode: 'markers', name: 'cost ratio at the most precise common target',
                marker: { color: 'rgba(0,0,0,0)', line: { color: colorOf(SBED), width: 1.5 }, size: 9, symbol: 'diamond' }, text: precise.map((p) => p.u.noise), hovertemplate: 'W/Γ %{x:.3g}<br>T_sweep/T_SBED %{y:.3g}<br>%{text}<extra></extra>' },
            { x: pred.map((p) => p.u.window_over_fwhm), y: pred.map((p) => (27 / (8 * Math.PI)) * p.q * p.u.window_over_fwhm), mode: 'markers', name: 'prediction 1.07 q W/Γ (SBED q)',
                marker: { color: token('slate-700'), size: 7, symbol: 'line-ew-open', line: { width: 2, color: token('slate-700') } }, text: pred.map((p) => `${p.u.noise} · q ${fmt(p.q, 2)}`), hovertemplate: 'W/Γ %{x:.3g}<br>predicted %{y:.3g}<br>%{text}<extra></extra>' },
        ];
        const xs = cases.map((u) => u.window_over_fwhm);
        const lo = Math.min(...xs) / 1.2, hi = Math.max(...xs) * 1.2;
        traces.push({ x: [lo, hi], y: [lo, hi].map((x) => (27 / (8 * Math.PI)) * x), ...theory('27/(8π) W/Γ (q = 1)') });
        const lay = layout('window W / Γ', 'speedup over the best sweep');
        lay.shapes = [hline(1, 'dot')];
        plot(div, traces, lay);
        const ratio = meas.map((u) => { const q = qOf(u); return q ? u.speedup.speedup_fitted / ((27 / (8 * Math.PI)) * q * u.window_over_fwhm) : null; });
        setClaim('art-speedup', `Article: at a precise target the speedup is about 1.07 q W/Γ. Measured ÷ predicted, median over ${meas.length} use cases: ${val(fmt(median(ratio), 2))}.`);
    }

    // ── Shared with the Lab page (static/claim-panels.js): the sim adapter ──
    const symbolOf = (name) => (name === SBED ? 'circle' : name === KNOWN ? 'diamond' : 'square');
    const findCase = (list, r) => list.find((u) => u.generator === r.generator && u.noise === r.noise);
    function drawShared(claimRows, rateCases, boundCases, alignedRows) {
        const P = window.NVClaimPanels;
        const xOf = (r) => (r.window_phys && r.fwhm_phys ? r.window_phys / r.fwhm_phys : null);
        const search = LOCATORS.map((name) => ({
            name: label(name), color: colorOf(name), symbol: symbolOf(name),
            points: claimRows.filter((r) => r.strategy === name).map((r) => {
                const u = findCase(boundCases, r);
                return { x: xOf(r), median: r.regimes ? r.regimes.detection_step : null, snr: u && !u.error ? u.snr : null, label: r.noise };
            }).filter((q) => q.x && q.snr),
        }));
        if (search.some((s) => s.points.length)) {
            const ratios = P.searchCostPanel(document.getElementById('art-searchcost'), search);
            setClaim('art-searchcost', `Eq. search-cost is a lower bound at low SNR. Median detection step ÷ Eq. search-cost over the use cases: ${val(fmt(median(ratios), 3))}.`);
        }
        const rates = LOCATORS.map((name) => ({
            name: label(name), color: colorOf(name),
            points: alignedRows.filter((r) => r.strategy === name).map((r) => {
                const c = findCase(claimRows.filter((x) => x.strategy === name), r);
                const u = findCase(rateCases, r);
                const late = r.claimed_rate_late || {};
                return {
                    x: c ? xOf(c) : null, claimedRate: late.claimed_rate ?? null,
                    placement: u && u.locators && u.locators[name] ? u.locators[name].placement_efficiency_median : null, label: r.noise,
                };
            }).filter((q) => q.x),
        }));
        if (rates.some((s) => s.points.length)) {
            P.ratesPanel(document.getElementById('art-rates'), rates);
            const q = rates[0].points.map((p) => p.claimedRate);
            setClaim('art-rates', `Claimed rate q̂ of SBED after detection, median over the use cases: ${val(fmt(median(q), 2))} (the lab reads the same number).`);
        }
        const spread = LOCATORS.map((name) => ({
            name: label(name), color: colorOf(name), symbol: symbolOf(name),
            points: claimRows.filter((r) => r.strategy === name && r.rms_curve).map((r) => ({
                claimed: r.rms_curve.claimed_sigma.at(-1), spread: r.rms_curve.error.at(-1), label: r.noise,
            })),
        }));
        if (spread.some((s) => s.points.length)) {
            const ratios = P.calibrationPanel(document.getElementById('art-spread'), spread, 'rms true error at the end of the run');
            setClaim('art-spread', `rms true error ÷ rms claimed σ at the end, median over use cases and locators: ${val(fmt(median(ratios), 2))} (1 = honest).`);
        }
        drawDetectors(alignedRows);
    }

    function drawDetectors(alignedRows) {
        const div = document.getElementById('art-detectors');
        const traces = [];
        const vals = [];
        const deltas = [];
        for (const name of LOCATORS) {
            const pts = alignedRows.filter((r) => r.strategy === name && r.detector_comparison && r.detector_comparison.median_a != null);
            if (!pts.length) continue;
            const c = pts.map((r) => r.detector_comparison);
            vals.push(...c.flatMap((d) => [d.median_a, d.median_b]));
            deltas.push(...c.map((d) => d.ratio_median));
            traces.push({
                x: c.map((d) => d.median_a), y: c.map((d) => d.median_b), mode: 'markers', name: label(name),
                marker: { color: colorOf(name), size: 9, symbol: symbolOf(name) },
                text: pts.map((r) => `${r.generator.replace(/^NVCenter-/, '')} ${r.noise} · both ${r.detector_comparison.n_both}/${r.detector_comparison.n_runs}, Δ median ${fmt(r.detector_comparison.delta_median, 3)}`),
                hovertemplate: 'dip test %{x}<br>sigma drop %{y}<br>%{text}<extra></extra>',
            });
        }
        if (!traces.length) {
            if (alignedRows.length) empty(div, 'No use case has runs that both detectors detect (re-run: needs the stored claimed σ series).');
            return;
        }
        const lo = Math.max(1, Math.min(...vals) / 1.5), hi = Math.max(...vals) * 1.5;
        traces.push({ x: [lo, hi], y: [lo, hi], mode: 'lines', name: 'the two definitions agree', line: { color: token('slate-500'), dash: 'dash', width: 1.5 } });
        const lay = layout("median step, simulation's dip test", "median step, lab's sigma drop");
        plot(div, traces, lay);
        setClaim('art-detectors', `Sigma drop ÷ dip test, median over use cases (per use case: median over runs): ${val(fmt(median(deltas), 3))}.`);
    }

    // ── Calibration ──
    function drawCalibration(c) {
        const div = document.getElementById('art-calib');
        const traces = [];
        const lay = layout('measurements n', 'rms true error ÷ rms claimed σ', 'log', 'log');
        lay.shapes = [hline(1)];
        const tail = [];
        for (const name of LOCATORS) {
            const r = c[name];
            if (!r || !r.rms_curve) continue;
            const y = r.rms_curve.error.map((e, i) => (r.rms_curve.claimed_sigma[i] > 0 ? e / r.rms_curve.claimed_sigma[i] : null));
            traces.push({ x: r.rms_curve.n, y, mode: 'lines', name: label(name), line: { color: colorOf(name), width: 2.5 } });
            if (r.regimes && r.regimes.gaussian_onset_step) lay.shapes.push(vline(r.regimes.gaussian_onset_step, colorOf(name), 'dot'));
            tail.push(`${label(name)} ${val(fmt(y[y.length - 1], 2))}`);
        }
        if (!traces.length) { empty(div, 'No claims for this use case.'); return; }
        plot(div, traces, lay);
        setClaim('art-calib', `Calibration is judged by coverage, not by the end-of-run accuracy. rms error ÷ rms claimed σ at the end: ${tail.join(', ')}.`);
    }

    function drawCoverage(claimRows) {
        const div = document.getElementById('art-coverage');
        const rows = claimRows.filter((r) => r.coverage_z_final && r.window_phys && r.fwhm_phys);
        if (!rows.length) { if (claimRows.length) empty(div, 'These runs store no signed z series: re-run them.'); return; }
        const traces = [];
        for (const name of LOCATORS) {
            const own = rows.filter((r) => r.strategy === name);
            for (const k of [1, 2]) {
                traces.push({
                    x: own.map((r) => r.window_phys / r.fwhm_phys), y: own.map((r) => r.coverage_z_final[`within_${k}`]), mode: 'markers',
                    name: `${label(name)} within ${k}σ`, marker: { color: colorOf(name), size: 8, symbol: k === 1 ? 'circle' : 'diamond' },
                    error_y: { type: 'data', symmetric: false, array: own.map((r) => r.coverage_z_final[`within_${k}_ci`][1] - r.coverage_z_final[`within_${k}`]), arrayminus: own.map((r) => r.coverage_z_final[`within_${k}`] - r.coverage_z_final[`within_${k}_ci`][0]), thickness: 1, width: 0, color: colorOf(name) },
                    text: own.map((r) => r.noise), hovertemplate: 'W/Γ %{x:.3g}<br>%{y:.2f}<br>%{text}<extra></extra>',
                });
            }
        }
        const lay = layout('window W / Γ', 'share of runs', 'log', 'linear');
        lay.yaxis.range = [0, 1.05];
        lay.shapes = [hline(GAUSS_WITHIN[1]), hline(GAUSS_WITHIN[2])];
        plot(div, traces, lay);
        const sb = rows.filter((r) => r.strategy === SBED);
        setClaim('art-coverage', `SBED within 1σ / 2σ, median over ${sb.length} use cases: ${val(fmt(median(sb.map((r) => r.coverage_z_final.within_1)), 2))} / ${val(fmt(median(sb.map((r) => r.coverage_z_final.within_2)), 2))} (calibrated: 0.68 / 0.95).`);
    }

    function drawPit(claimRows) {
        const div = document.getElementById('art-pit');
        if (!claimRows.length) return;
        const rows = claimRows.filter((r) => r.coverage_pit_final && r.window_phys && r.fwhm_phys);
        if (!rows.length) {
            div.innerHTML = '<p class="empty-state">Needs re-run: these runs store no PIT (pit_final), so coverage by the posterior\'s own quantiles cannot be read. Runs recorded after the PIT series was added show it here.</p>';
            setClaim('art-pit', 'Coverage holds also before the posterior is Gaussian when read from its own quantiles.');
            return;
        }
        const traces = [];
        for (const name of LOCATORS) {
            const own = rows.filter((r) => r.strategy === name);
            for (const k of [1, 2]) {
                traces.push({
                    x: own.map((r) => r.window_phys / r.fwhm_phys), y: own.map((r) => r.coverage_pit_final[`within_${k}`]), mode: 'markers',
                    name: `${label(name)} central ${k === 1 ? 68 : 95} %`, marker: { color: colorOf(name), size: 8, symbol: k === 1 ? 'circle' : 'diamond' },
                    text: own.map((r) => r.noise), hovertemplate: 'W/Γ %{x:.3g}<br>%{y:.2f}<br>%{text}<extra></extra>',
                });
            }
        }
        const lay = layout('window W / Γ', 'share of runs', 'log', 'linear');
        lay.yaxis.range = [0, 1.05];
        lay.shapes = [hline(GAUSS_WITHIN[1]), hline(GAUSS_WITHIN[2])];
        plot(div, traces, lay);
        const sb = rows.filter((r) => r.strategy === SBED);
        setClaim('art-pit', `SBED inside its central 68 % / 95 %, median over ${sb.length} use cases: ${val(fmt(median(sb.map((r) => r.coverage_pit_final.within_1)), 2))} / ${val(fmt(median(sb.map((r) => r.coverage_pit_final.within_2)), 2))}.`);
    }

    // ── Entry point ──
    async function render(root) {
        if (!root) return;
        await ensurePlotly();
        if (state.root === root && state.family) return;   // already drawn: keep the loaded data and the selection
        state.root = root;
        root.innerHTML = skeleton();
        try {
            const resp = await fetch(api('combos'), { cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            state.combos = await resp.json();
        } catch (err) {
            document.getElementById('art-notice').innerHTML = `<p class="notice-amber">Could not list the cached runs: ${esc(err.message)}</p>`;
            return;
        }
        const families = new Set(state.combos.map((c) => familyOf(c.generator)).filter(Boolean));
        if (!families.size) {
            document.getElementById('art-notice').innerHTML = '<p class="notice-amber">No window-grid runs (NVCenter-…-span&lt;W&gt;-…) in this cache.</p>';
            return;
        }
        selectFamily(families.has(DEFAULT_FAMILY) ? DEFAULT_FAMILY : [...families].sort()[0]);
    }

    window.NVArticleView = { render };
})();
