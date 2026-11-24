// Claim panels shared by the Simulation and Lab pages: the same component, fed by a per-source adapter that maps the
// server's numbers onto one point shape. Only the panels that need the prior ensemble (bounds on the mean squared
// error, ρ, the fitted rate, PIT) are simulation-only and live in article-view.js; these three hold for both sources
// because they use claimed quantities only (docs/evaluation_metrics.md, "The shared run record").
// Needs Plotly (ensurePlotly from plotly-utils.js).
(function () {
    const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

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
    const hline = (y, dash = 'dot') => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: y, y1: y, line: { color: token('slate-500'), width: 1.5, dash } });
    const theory = (name, dash = 'dash') => ({ name, mode: 'lines', line: { color: token('slate-700'), width: 2, dash } });
    const draw = (div, traces, lay) => { div.innerHTML = ''; Plotly.newPlot(div, traces, lay, { responsive: true, displaylogo: false }); };
    const range = (xs) => { const lo = Math.min(...xs) / 1.2, hi = Math.max(...xs) * 1.2; return Array.from({ length: 40 }, (_, i) => lo * (hi / lo) ** (i / 39)); };

    // Eq. search-cost: a lower bound for any strategy at low SNR, n_search ≳ (8/π)(W/Γ)/(k SNR²), k shots per measurement.
    const searchCost = (wOverGamma, snr, shots = 1) => (8 / Math.PI) * wOverGamma / (shots * snr * snr);
    // Eq. sweep-info: the information rate of a sweep over the window, in units of I_max.
    const sweepRate = (wOverGamma) => Math.min(1, (8 * Math.PI) / (27 * wOverGamma));

    // Search cost against Eq. search-cost. series: [{name, color, points: [{x: W/Γ, median, q25, q75, snr, label}]}].
    function searchCostPanel(div, series) {
        const traces = [];
        const xs = [];
        for (const s of series) {
            const p = s.points.filter((q) => q.median != null && q.x > 0);
            if (!p.length) continue;
            xs.push(...p.map((q) => q.x));
            traces.push({
                x: p.map((q) => q.x), y: p.map((q) => q.median), mode: 'markers', name: `${s.name} median detection step`,
                marker: { color: s.color, size: 9, symbol: s.symbol || 'circle' },
                error_y: { type: 'data', symmetric: false, array: p.map((q) => (q.q75 ?? q.median) - q.median), arrayminus: p.map((q) => q.median - (q.q25 ?? q.median)), thickness: 1, width: 0, color: s.color },
                text: p.map((q) => `${esc(q.label)} · ${(q.median / searchCost(q.x, q.snr)).toPrecision(3)} × Eq. search-cost`),
                hovertemplate: 'W/Γ %{x:.3g}<br>detection step %{y}<br>%{text}<extra></extra>',
            });
            traces.push({
                x: p.map((q) => q.x), y: p.map((q) => searchCost(q.x, q.snr)), mode: 'markers', name: `Eq. search-cost (${s.name}'s SNR)`,
                marker: { color: token('slate-700'), size: 7, symbol: 'line-ew-open', line: { width: 2, color: token('slate-700') } },
                text: p.map((q) => `SNR ${q.snr.toPrecision(3)}`), hovertemplate: 'W/Γ %{x:.3g}<br>(8/π)(W/Γ)/SNR² = %{y:.3g}<br>%{text}<extra></extra>',
            });
        }
        if (!traces.length) { div.innerHTML = '<p class="empty-state">No run detected the dip.</p>'; return []; }
        draw(div, traces, layout('window W / Γ', 'measurements to detection'));
        return series.flatMap((s) => s.points.filter((q) => q.median != null && q.snr).map((q) => q.median / searchCost(q.x, q.snr)));
    }

    // Information per measurement after detection: claimed rate q̂ and placement, against the sweep rate 8πΓ/(27W).
    // series: [{name, color, points: [{x: W/Γ, claimedRate, placement, label}]}].
    function ratesPanel(div, series) {
        const traces = [];
        const xs = [];
        for (const s of series) {
            const c = s.points.filter((q) => q.claimedRate != null && q.claimedRate > 0);
            const pl = s.points.filter((q) => q.placement != null);
            xs.push(...c.map((q) => q.x), ...pl.map((q) => q.x));
            if (c.length) {
                traces.push({ x: c.map((q) => q.x), y: c.map((q) => q.claimedRate), mode: 'markers', name: `${s.name} claimed rate q̂`,
                    marker: { color: s.color, size: 9, symbol: 'circle' }, text: c.map((q) => esc(q.label)), hovertemplate: 'W/Γ %{x:.3g}<br>q̂ %{y:.3g}<br>%{text}<extra></extra>' });
            }
            if (pl.length) {
                traces.push({ x: pl.map((q) => q.x), y: pl.map((q) => q.placement), mode: 'markers', name: `${s.name} placement ⟨I(d)/I_max⟩`,
                    marker: { color: s.color, size: 9, symbol: 'x-thin', line: { color: s.color, width: 2 } }, text: pl.map((q) => esc(q.label)), hovertemplate: 'W/Γ %{x:.3g}<br>placement %{y:.3g}<br>%{text}<extra></extra>' });
            }
        }
        if (!xs.length) { div.innerHTML = '<p class="empty-state">No claimed rate or placement for these runs.</p>'; return; }
        const grid = range(xs);
        traces.push({ x: grid, y: grid.map(sweepRate), ...theory('sweep 8πΓ/(27W)') });
        const lay = layout('window W / Γ', 'information per measurement / I_max');
        lay.shapes = [hline(1)];
        draw(div, traces, lay);
    }

    // Claimed σ against the observed spread. series: [{name, color, points: [{claimed, spread, label}]}]. The spread is
    // the scatter of repeated runs (lab) or the rms true error over the prior ensemble (simulation): the adapter says which.
    function calibrationPanel(div, series, spreadTitle) {
        const traces = [];
        const vals = [];
        for (const s of series) {
            const p = s.points.filter((q) => q.claimed > 0 && q.spread > 0);
            vals.push(...p.flatMap((q) => [q.claimed, q.spread]));
            traces.push({ x: p.map((q) => q.claimed), y: p.map((q) => q.spread), mode: 'markers', name: s.name,
                marker: { color: s.color, size: 10, symbol: s.symbol || 'circle' }, text: p.map((q) => esc(q.label)),
                hovertemplate: 'claimed σ %{x:.3g}<br>spread %{y:.3g}<br>%{text}<extra></extra>' });
        }
        if (!vals.length) { div.innerHTML = '<p class="empty-state">No claimed σ and spread for these runs.</p>'; return []; }
        const lo = Math.min(...vals) / 1.5, hi = Math.max(...vals) * 1.5;
        traces.push({ x: [lo, hi], y: [lo, hi], mode: 'lines', name: 'calibrated: spread = claimed σ', line: { color: token('slate-500'), dash: 'dash', width: 1.5 } });
        const lay = layout('claimed σ at the end of the run', spreadTitle);
        lay.xaxis.range = [Math.log10(lo), Math.log10(hi)];
        lay.yaxis.range = [Math.log10(lo), Math.log10(hi)];
        draw(div, traces, lay);
        return series.flatMap((s) => s.points.filter((q) => q.claimed > 0 && q.spread > 0).map((q) => q.spread / q.claimed));
    }

    window.NVClaimPanels = { searchCostPanel, ratesPanel, calibrationPanel, searchCost, sweepRate };
})();
