// Plotly loading and figure decode/fetch utilities.
// All functions are global — depended on by app.js main().
//
// resolveAssetPath  — prepends window.NVISION_ASSET_PREFIX to relative paths
// ensurePlotly      — lazy-loads Plotly from CDN on first call
// _fetchJson        — fetches a .json or .json.gz file, decodes arrays
// _decodePlotlyFigure — recursively decodes {__f32__} and {bdata,dtype} arrays
// _decodeBase64F32  — base64 → Float32Array (our custom format)
// _decodeBase64Typed — base64 → Float32Array|Float64Array (Plotly Python 5.x format)
// fetchGraphDef     — fetches and caches a graph definition from static/graphs/
// buildFigureFromData — builds a Plotly {data, layout} from a lean data object

function resolveAssetPath(relativePath) {
    if (!relativePath) {
        return '';
    }
    if (/^(?:[a-z]+:)?\/\//i.test(relativePath) || relativePath.startsWith('/')) {
        return relativePath;
    }
    const cleaned = String(relativePath).replace(/^\.?\//, '');
    const prefix = window.NVISION_ASSET_PREFIX || '';
    return prefix + cleaned;
}

let plotlyLoadPromise = null;
function ensurePlotly() {
    if (window.Plotly) {
        return Promise.resolve();
    }
    if (!plotlyLoadPromise) {
        plotlyLoadPromise = new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
            s.async = true;
            s.onload = () => resolve();
            s.onerror = () => reject(new Error('Plotly failed to load'));
            document.head.appendChild(s);
        });
    }
    return plotlyLoadPromise;
}

async function _fetchJson(url) {
    let resp = await fetch(resolveAssetPath(url), { cache: 'no-store' });
    // Manifest entries can have stale .json paths when the file was upgraded to .json.gz on disk.
    // Transparently retry with the .gz variant so old manifests keep working.
    if (!resp.ok && resp.status === 404 && url.endsWith('.json') && !url.endsWith('.json.gz')) {
        const gzUrl = url + '.gz';
        const gzResp = await fetch(resolveAssetPath(gzUrl), { cache: 'no-store' });
        if (gzResp.ok) {
            resp = gzResp;
            url = gzUrl;
        }
    }
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    let parsed;
    if (url.endsWith('.gz')) {
        const ds = new DecompressionStream('gzip');
        const text = await new Response(resp.body.pipeThrough(ds)).text();
        // JSON spec does not allow Infinity/NaN — replace with null
        parsed = JSON.parse(text.replace(/\bInfinity\b/g, 'null').replace(/-Infinity\b/g, 'null').replace(/\bNaN\b/g, 'null'));
    } else {
        parsed = await resp.json();
    }
    return _decodePlotlyFigure(parsed);
}

function _decodeBase64F32(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const u8 = new Uint8Array(len);
    for (let i = 0; i < len; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
}

function _decodeBase64Typed(b64, dtype) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const TypedArray = dtype === 'float32' ? Float32Array : Float64Array;
    return new TypedArray(u8.buffer);
}

function _decodePlotlyFigure(obj) {
    if (obj === null || obj === undefined || typeof obj !== 'object') return obj;
    // TypedArrays (Float32Array, Float64Array) are already decoded — pass through
    if (ArrayBuffer.isView(obj)) return obj;
    // Custom f32 encoding (our own format)
    if (obj.__f32__ !== undefined) return _decodeBase64F32(obj.__f32__);
    // Plotly Python 5.x numpy serialization: {dtype, bdata[, shape]}
    if (obj.bdata !== undefined && obj.dtype !== undefined) {
        return _decodeBase64Typed(obj.bdata, obj.dtype);
    }
    if (Array.isArray(obj)) return obj.map(_decodePlotlyFigure);
    const out = {};
    for (const k of Object.keys(obj)) out[k] = _decodePlotlyFigure(obj[k]);
    return out;
}

// ── Graph definition cache + figure builders ─────────────────────────────────

const _defCache = new Map();

async function fetchGraphDef(graphType) {
    if (_defCache.has(graphType)) return _defCache.get(graphType);
    const url = resolveAssetPath(`graphs/${graphType}.json`);
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(`Missing graph def: ${graphType} (${url})`);
    const def = await r.json();
    _defCache.set(graphType, def);
    return def;
}

async function buildFigureFromData(raw) {
    const def = await fetchGraphDef(raw._graph_type);
    switch (raw._graph_type) {
        case 'scan':       return _buildScanFigure(def, raw);
        case 'box':        return _buildBoxFigure(def, raw);
        case 'histogram':  return _buildHistogramFigure(def, raw);
        case 'violin':     return _buildViolinFigure(def, raw);
        case 'chart':      return _buildChartFigure(def, raw);
        default: throw new Error(`Unknown graph type: ${raw._graph_type}`);
    }
}

// ── Scan figure builder ───────────────────────────────────────────────────────

function _scanBaseline(yDense) {
    let max = -Infinity;
    for (const v of yDense) {
        if (v != null && Number.isFinite(v) && v > max) max = v;
    }
    const b = Number.isFinite(max) ? max : 1.0;
    return b > 1e-12 ? b : 1.0;
}

function _depthPct(y, baseline) {
    return y == null ? 0 : Math.max(0, (baseline - y) / baseline * 100);
}

function _buildScanFigure(def, data) {
    const traces = [];
    const hasMetrics = !!data.has_metrics;
    const sa = hasMetrics ? { xaxis: 'x', yaxis: 'y' } : {};
    const T = def.traces;

    // True signal
    traces.push(Object.assign({}, T.true_signal, { x: data.x_dense, y: data.y_dense }, sa));

    // Possible-measurement-range bands (Monte-Carlo envelope): outer ±2σ drawn
    // first so the inner ±1σ band layers on top of it. Falls back to the
    // legacy single-draw dotted line for older cached data that predates the bands.
    if (data.y_dense_noisy_lo && data.y_dense_noisy_lo.length) {
        if (data.y_dense_noisy_lo2 && data.y_dense_noisy_lo2.length) {
            traces.push(Object.assign({}, T.noisy_band_lo2, { x: data.x_dense, y: data.y_dense_noisy_lo2 }, sa));
            traces.push(Object.assign({}, T.noisy_band_hi2, { x: data.x_dense, y: data.y_dense_noisy_hi2 }, sa));
        }
        traces.push(Object.assign({}, T.noisy_band_lo, { x: data.x_dense, y: data.y_dense_noisy_lo }, sa));
        traces.push(Object.assign({}, T.noisy_band_hi, { x: data.x_dense, y: data.y_dense_noisy_hi }, sa));
    } else if (data.y_dense_noisy && data.y_dense_noisy.length) {
        traces.push(Object.assign({}, T.noisy_signal, { x: data.x_dense, y: data.y_dense_noisy }, sa));
    }

    // Measurement distribution (baseline anchor + density curve)
    if (data.meas_dist && data.meas_dist.x && data.meas_dist.x.length) {
        const md = data.meas_dist;
        const baseArr = new Array(md.x.length).fill(md.y_baseline);
        traces.push(Object.assign({}, T.meas_dist_baseline, { x: md.x, y: baseArr }, sa));
        traces.push(Object.assign({}, T.meas_dist, { x: md.x, y: md.y_curve, customdata: md.customdata }, sa));
    }

    // Measurement traces
    const m = data.measurements;
    const baseline = _scanBaseline(data.y_dense);
    if (m) {
        if (m.mode === 'phases') {
            if (m.coarse_x && m.coarse_x.length) {
                traces.push(Object.assign({}, T.coarse, {
                    x: m.coarse_x, y: m.coarse_y,
                    customdata: Array.from(m.coarse_y).map(y => _depthPct(y, baseline)),
                }, sa));
            }
            if (m.secondary_x && m.secondary_x.length) {
                traces.push(Object.assign({}, T.secondary, {
                    x: m.secondary_x, y: m.secondary_y,
                    customdata: Array.from(m.secondary_y).map(y => _depthPct(y, baseline)),
                }, sa));
            }
            if (m.tertiary_x && m.tertiary_x.length) {
                traces.push(Object.assign({}, T.tertiary, {
                    x: m.tertiary_x, y: m.tertiary_y,
                    customdata: Array.from(m.tertiary_y).map(y => _depthPct(y, baseline)),
                }, sa));
            }
            if (m.fine_x && m.fine_x.length) {
                const fineSteps = (m.fine_step && m.fine_step.length === m.fine_x.length)
                    ? Array.from(m.fine_step) : m.fine_x.map((_, i) => i);
                const colorbar = hasMetrics
                    ? { title: { text: 'inference step' }, len: 0.6, y: 0.8 }
                    : { title: { text: 'inference step' } };
                traces.push(Object.assign({}, T.inference, {
                    x: m.fine_x, y: m.fine_y,
                    marker: Object.assign({}, T.inference.marker, { color: fineSteps, colorbar }),
                    customdata: Array.from(m.fine_y).map(y => _depthPct(y, baseline)),
                }, sa));
            }
        } else if (m.mode === 'steps') {
            const stepIdx = (m.step && m.step.length === m.x.length)
                ? Array.from(m.step) : m.x.map((_, i) => i);
            const colorbar = hasMetrics
                ? { title: { text: 'step' }, len: 0.6, y: 0.8 }
                : { title: { text: 'step' } };
            traces.push(Object.assign({}, T.steps_noisy, {
                x: m.x, y: m.y,
                marker: Object.assign({}, T.steps_noisy.marker, { color: stepIdx, colorbar }),
                customdata: Array.from(m.y).map(y => _depthPct(y, baseline)),
            }, sa));
        }
    }

    // Mode signal (drawn on top of measurements)
    if (data.y_dense_mode && data.y_dense_mode.length) {
        traces.push(Object.assign({}, T.mode_signal, { x: data.x_dense, y: data.y_dense_mode }, sa));
    }

    // Sobol overlay
    if (data.sobol_measurements && data.sobol_measurements.x && data.sobol_measurements.x.length) {
        const sm = data.sobol_measurements;
        traces.push(Object.assign({}, T.sobol_meas, {
            x: sm.x, y: sm.y,
            customdata: Array.from(sm.y).map(y => _depthPct(y, baseline)),
        }, sa));
    }
    if (data.sobol_mode_y && data.sobol_mode_y.length) {
        traces.push(Object.assign({}, T.sobol_mode, { x: data.x_dense, y: data.sobol_mode_y }, sa));
    }

    // Sweep overlay
    if (data.sweep_measurements && data.sweep_measurements.x && data.sweep_measurements.x.length) {
        const sm = data.sweep_measurements;
        traces.push(Object.assign({}, T.sweep_meas, {
            x: sm.x, y: sm.y,
            customdata: Array.from(sm.y).map(y => _depthPct(y, baseline)),
        }, sa));
    }
    if (data.sweep_mode_y && data.sweep_mode_y.length) {
        traces.push(Object.assign({}, T.sweep_mode, { x: data.x_dense, y: data.sweep_mode_y }, sa));
    }

    // Metric traces (row 2 when has_metrics)
    if (hasMetrics) {
        const ma2 = { xaxis: 'x2', yaxis: 'y2' };
        if (data.entropy && data.entropy.length) {
            const steps = data.entropy.map((_, i) => i);
            traces.push(Object.assign({}, T.entropy, { x: steps, y: data.entropy }, ma2));
        }
        if (data.uncertainty && data.uncertainty.length) {
            const steps = data.uncertainty.map((_, i) => i);
            const yaxis = data.entropy ? 'y3' : 'y2';
            traces.push(Object.assign({}, T.uncertainty, { x: steps, y: data.uncertainty, xaxis: 'x2', yaxis }));
        }
    }

    // Build layout
    const baseLayout = JSON.parse(JSON.stringify(
        hasMetrics ? def.layout.with_metrics : def.layout.no_metrics
    ));

    // Dynamic shapes (focus window, per-dip windows)
    const shapes = [];
    const extraAnnotations = [];
    const yref = hasMetrics ? 'y domain' : 'paper';

    if (data.focus_window && data.focus_window.length === 2) {
        const [fw0, fw1] = data.focus_window;
        if (Number.isFinite(fw0) && Number.isFinite(fw1) && fw1 > fw0) {
            const fws = def.focus_window_style;
            shapes.push({
                type: 'rect', xref: 'x', yref,
                x0: fw0, x1: fw1, y0: 0, y1: 1,
                fillcolor: fws.fillcolor,
                line: { width: 1, color: fws.line_color },
                layer: 'below',
            });
            extraAnnotations.push({
                text: fws.annotation, x: fw0, xref: 'x',
                y: 1, yref, yanchor: 'bottom', xanchor: 'left',
                showarrow: false, font: { size: 11 },
            });
        }
    }

    if (data.per_dip_windows && data.per_dip_windows.length) {
        data.per_dip_windows.forEach(([lo, hi], i) => {
            if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return;
            const c = def.per_dip_colors[i % def.per_dip_colors.length];
            shapes.push({
                type: 'rect', xref: 'x', yref,
                x0: lo, x1: hi, y0: 0, y1: 1,
                fillcolor: c.fill, line: { width: 1, color: c.line }, layer: 'below',
            });
            extraAnnotations.push({
                text: `Dip ${i + 1}`, x: lo, xref: 'x',
                y: 1, yref, yanchor: 'bottom', xanchor: 'left',
                showarrow: false, font: { size: 11 },
            });
        });
    }

    // Mark the true (fixed, physical) center frequency with a vertical reference line.
    if (data.true_params && data.true_params.params && Number.isFinite(data.true_params.params.frequency)) {
        const cfs = def.center_freq_style;
        const cf = data.true_params.params.frequency;
        shapes.push({
            type: 'line', xref: 'x', yref,
            x0: cf, x1: cf, y0: 0, y1: 1,
            line: { width: 1.5, color: cfs.line_color, dash: 'dash' },
            layer: 'above',
        });
        extraAnnotations.push({
            text: cfs.annotation, x: cf, xref: 'x',
            y: 1, yref, yanchor: 'bottom', xanchor: 'center',
            showarrow: false, font: { size: 11, color: cfs.line_color },
        });
    }

    if (shapes.length) baseLayout.shapes = shapes;
    if (extraAnnotations.length) {
        const existing = baseLayout.annotations || [];
        baseLayout.annotations = existing.concat(extraAnnotations);
    }

    // Embed meta for narrowed_param_bounds / true_params UI panels
    const meta = {};
    if (data.narrowed_param_bounds) meta.narrowed_param_bounds = data.narrowed_param_bounds;
    if (data.per_dip_windows) meta.per_dip_windows = data.per_dip_windows;
    if (data.true_params) meta.true_params = data.true_params;
    if (Object.keys(meta).length) baseLayout.meta = meta;

    return { data: traces, layout: baseLayout };
}

// ── Box figure builder ────────────────────────────────────────────────────────

function _buildBoxFigure(def, data) {
    const traces = (data.series || []).map(s =>
        Object.assign({}, def.trace, { name: s.name, y: s.values })
    );
    const layout = Object.assign({}, def.layout, {
        title: { text: data.title || '' },
        yaxis: { title: { text: data.y_axis_title || '' } },
    });
    return { data: traces, layout };
}

// ── Histogram figure builder ──────────────────────────────────────────────────

function _buildHistogramFigure(def, data) {
    let traces;
    if (data.series && data.series.length) {
        // Multi-series histogram (milestone-style)
        traces = data.series.map(s => Object.assign({}, def.trace, {
            name: s.name, x: s.values,
            marker: { color: s.color || undefined },
        }));
    } else {
        traces = [Object.assign({}, def.trace, {
            name: data.name || '',
            x: data.values || [],
            marker: { color: data.color || 'steelblue' },
        })];
    }
    const layout = Object.assign({}, def.layout, {
        title: { text: data.title || '' },
        xaxis: { title: { text: data.xaxis_title || '' } },
        yaxis: { title: { text: data.yaxis_title || '' } },
    });
    return { data: traces, layout };
}

// ── Violin figure builder ─────────────────────────────────────────────────────

function _buildViolinFigure(def, data) {
    const traces = [Object.assign({}, def.trace, {
        name: data.name || '',
        y: data.values || [],
    })];
    const layout = Object.assign({}, def.layout, {
        title: { text: data.title || '' },
        yaxis: { title: { text: data.yaxis_title || '' } },
    });
    return { data: traces, layout };
}

// ── Generic chart figure builder (line / bar / scatter) ───────────────────────

// Convergence-rate/n_total/n_converged are optional per-series arrays (already
// present on grid_study.py's vs-noise payloads, and on the client-computed Grid
// Stats payload) giving hover text like "3/5 converged (60%)" per point.
function _convergenceHoverText(s) {
    if (!s.convergence_rate) return null;
    return s.convergence_rate.map((r, i) => {
        const nTot = s.n_total ? s.n_total[i] : null;
        const nConv = s.n_converged ? s.n_converged[i] : null;
        const pct = r != null ? `${Math.round(r * 100)}%` : 'n/a';
        return nTot != null && nConv != null ? `${nConv}/${nTot} converged (${pct})` : `converged: ${pct}`;
    });
}

function _buildChartFigure(def, data) {
    const mode = data.mode || 'lines+markers';
    const traces = (data.series || []).map(s => {
        const hoverText = _convergenceHoverText(s);
        const hoverExtra = hoverText
            ? { text: hoverText, hovertemplate: '%{y}<br>%{text}<extra>%{fullData.name}</extra>' }
            : {};
        if (mode === 'bar') {
            return Object.assign({ type: 'bar', name: s.name, x: s.x, y: s.y }, hoverExtra);
        }
        return Object.assign({ type: 'scatter', name: s.name, x: s.x, y: s.y, mode }, hoverExtra);
    });
    const xaxisExtra = data.xaxis_type ? { type: data.xaxis_type } : {};
    const layout = Object.assign({}, def.layout, {
        title: { text: data.title || '' },
        xaxis: Object.assign({ title: { text: data.xaxis_title || '' } }, xaxisExtra),
        yaxis: { title: { text: data.yaxis_title || '' } },
    });
    if (mode === 'bar') layout.barmode = 'group';
    return { data: traces, layout };
}
