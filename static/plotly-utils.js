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
        // JSON spec does not allow Infinity/NaN — replace with null. Must only match
        // these as bare JSON *values* (immediately between a structural char and another
        // structural char), not as a substring match anywhere in the text: a blind \b-bounded
        // replace also matches "NaN"/"Infinity" occurring inside a quoted base64 typed-array
        // payload whenever it happens to sit next to a non-word base64 char (+, /, =), silently
        // corrupting that payload's length/alignment. Our own writer (_f32_json.py) always
        // emits compact JSON with no whitespace, so requiring immediate adjacency is exact.
        parsed = JSON.parse(text.replace(/(?<=[:,[])(-Infinity|Infinity|NaN)(?=[,\]}])/g, 'null'));
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

// IEEE 754 half-precision -> JS double. No native Float16Array decode path
// (browser support for it is still spotty), so unpack sign/exponent/mantissa by hand.
function _halfToFloat(h) {
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h & 0x7C00) >> 10;
    const f = h & 0x03FF;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1F) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

function _decodeBase64F16(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const u8 = new Uint8Array(len);
    for (let i = 0; i < len; i++) u8[i] = bin.charCodeAt(i);
    const u16 = new Uint16Array(u8.buffer);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = _halfToFloat(u16[i]);
    return out;
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
    // Custom f32/f16 encoding (our own format)
    if (obj.__f32__ !== undefined) return _decodeBase64F32(obj.__f32__);
    if (obj.__f16__ !== undefined) return _decodeBase64F16(obj.__f16__);
    if (obj.__f16s__ !== undefined) {
        const [lo, span, b64] = obj.__f16s__;
        const half = _decodeBase64F16(b64);
        const out = new Float32Array(half.length);
        for (let i = 0; i < half.length; i++) out[i] = half[i] * span + lo;
        return out;
    }
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

// Real (e.g. MATLAB) runs have no ground truth, so y_dense is all-NaN — a
// "true signal" legend entry that never draws anything reads as a broken
// trace rather than "no ground truth for real data". Omit it in that case.
function _hasFiniteValues(arr) {
    if (!arr) return false;
    for (const v of arr) {
        if (v != null && Number.isFinite(v)) return true;
    }
    return false;
}

function _depthPct(y, baseline) {
    return y == null ? 0 : Math.max(0, (baseline - y) / baseline * 100);
}

// Where a dense curve crosses `halfY`, walking outward from `iMin` (the dip's own index) in
// direction `dir` (-1 left, +1 right). Linearly interpolated between the two straddling
// samples; falls back to the last finite sample if the curve never rises back to halfY.
function _halfDepthCrossing(xs, ys, iMin, halfY, dir, iLoBound, iHiBound) {
    const lo = iLoBound == null ? 0 : iLoBound;
    const hi = iHiBound == null ? xs.length - 1 : iHiBound;
    let i = iMin;
    let j = i + dir;
    while (j >= lo && j <= hi && ys[j] != null && Number.isFinite(ys[j]) && ys[j] <= halfY) {
        i = j;
        j += dir;
    }
    if (j < lo || j > hi || ys[j] == null || !Number.isFinite(ys[j])) return xs[i];
    const y0 = ys[i], y1 = ys[j];
    const t = y1 === y0 ? 0 : (halfY - y0) / (y1 - y0);
    return xs[i] + t * (xs[j] - xs[i]);
}

// Locates the deepest dip within index range [iLo, iHi] of a dense curve and measures it
// geometrically — full width at half depth, and fractional depth below baseline — so the
// plot can label what "width" and "contrast" mean using only the rendered curve itself.
// This deliberately reads no model parameters (c_total, linewidth, ...): those differ by
// lineshape (Lorentzian vs. Voigt vs. Saturation-Voigt, hyperfine or not), while "how
// wide/deep is the dip you're looking at" is the same measurement for all of them. Bounding
// to [iLo, iHi] keeps the half-depth walk from crossing into a neighboring dip.
function _findDip(xs, ys, baseline, iLo, iHi) {
    let iMin = -1;
    let yMin = Infinity;
    for (let i = iLo; i <= iHi; i++) {
        const y = ys[i];
        if (y != null && Number.isFinite(y) && y < yMin) { yMin = y; iMin = i; }
    }
    if (iMin < 0) return null;
    const depth = baseline - yMin;
    if (depth <= 0 || depth / baseline < 0.05) return null; // too shallow to bother annotating
    const halfY = baseline - depth / 2;
    const xLeft = _halfDepthCrossing(xs, ys, iMin, halfY, -1, iLo, iHi);
    const xRight = _halfDepthCrossing(xs, ys, iMin, halfY, 1, iLo, iHi);
    if (!(xRight > xLeft)) return null;
    return { xMin: xs[iMin], yMin, xLeft, xRight, fwhm: xRight - xLeft, depthPct: (depth / baseline) * 100, iLo, iHi };
}

// Indices of interior local minima of ys (y[i] <= both neighbors, strictly less than at
// least one, so flat runs only yield their first sample). Used to find every dip in a
// possibly-split spectrum (e.g. Voigt+Zeeman, two resolved groups) rather than only the
// single globally-deepest point.
function _localMinimaIndices(ys) {
    const idx = [];
    for (let i = 1; i < ys.length - 1; i++) {
        const y = ys[i], yPrev = ys[i - 1], yNext = ys[i + 1];
        if (y == null || !Number.isFinite(y) || yPrev == null || !Number.isFinite(yPrev) ||
            yNext == null || !Number.isFinite(yNext)) continue;
        if (y <= yPrev && y <= yNext && (y < yPrev || y < yNext)) idx.push(i);
    }
    return idx;
}

// Up to `maxDips` most prominent dips in a dense curve, each measured geometrically (see
// _findDip) — so a resolved split spectrum (e.g. two Zeeman groups) gets a width/contrast
// readout per dip instead of only the single deepest one. Candidate dip centers come from
// every local minimum, ranked by depth; a candidate within `minSeparationFrac` of the
// curve's x-span from an already-picked one is skipped as almost certainly the same dip
// (numerical plateau) rather than a second, resolved feature. Returned left-to-right.
function _findDips(xs, ys, baseline, maxDips) {
    maxDips = maxDips || 2;
    let candidates = _localMinimaIndices(ys);
    if (!candidates.length) {
        let iMin = -1, yMin = Infinity;
        for (let i = 0; i < ys.length; i++) {
            const y = ys[i];
            if (y != null && Number.isFinite(y) && y < yMin) { yMin = y; iMin = i; }
        }
        if (iMin >= 0) candidates = [iMin];
    }
    if (!candidates.length) return [];
    candidates.sort((a, b) => ys[a] - ys[b]); // deepest first
    const xSpan = xs[xs.length - 1] - xs[0];
    const minSep = Math.abs(xSpan) * 0.03;
    const picked = [];
    for (const i of candidates) {
        if (picked.length >= maxDips) break;
        if (picked.some((j) => Math.abs(xs[j] - xs[i]) < minSep)) continue;
        picked.push(i);
    }
    picked.sort((a, b) => xs[a] - xs[b]); // left to right
    // Bound each dip's half-depth walk at the midpoint with its picked neighbor(s) so
    // resolved-but-close dips don't measure into each other.
    const dips = [];
    for (let k = 0; k < picked.length; k++) {
        let iLo = 0, iHi = xs.length - 1;
        if (k > 0) {
            const mid = (xs[picked[k - 1]] + xs[picked[k]]) / 2;
            while (iLo < xs.length && xs[iLo] < mid) iLo++;
        }
        if (k < picked.length - 1) {
            const mid = (xs[picked[k]] + xs[picked[k + 1]]) / 2;
            iHi = iLo;
            while (iHi < xs.length && xs[iHi] <= mid) iHi++;
            iHi = Math.max(iHi - 1, iLo);
        }
        const dip = _findDip(xs, ys, baseline, iLo, iHi);
        if (dip) dips.push(dip);
    }
    return dips;
}

// Names whichever model parameter a geometric width/contrast reading is standing in for
// (e.g. 'linewidth' for a plain Lorentzian, 'fwhm_total' for a plain Voigt,
// 'homogeneous_linewidth' for a Voigt+Zeeman split, which samples that instead of
// fwhm_total directly), so the annotation's bracketed letter matches the same symbol used
// in the signal-equation panel (see PARAM_LETTERS / paramLetter in format-utils.js) —
// picked by which key is actually present on this combo's true params rather than by
// parsing the generator name.
function _paramLetterFor(trueParams, candidates) {
    if (!trueParams) return '';
    for (const key of candidates) {
        if (Number.isFinite(trueParams[key])) return paramLetterSuffix(key);
    }
    return '';
}

// A locator that revisits the same discrete frequency bin (e.g. matlab-run cycling
// through a real .mat file's per-shot data one shot at a time) plots several dots at
// the *exact* same x. Left alone they just stack invisibly on top of each other with
// no visual cue they're the same bin. Group them into one vertical stem per unique x
// (min-to-max span of repeated shots there) drawn under the dots, so repeats read as
// one column instead of a hidden pile.
function _buildStemTrace(xs, ys, template) {
    if (!xs || !xs.length) return null;
    const groups = new Map();
    for (let i = 0; i < xs.length; i++) {
        const x = xs[i];
        const y = ys[i];
        if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) continue;
        const g = groups.get(x);
        if (!g) groups.set(x, { min: y, max: y });
        else {
            if (y < g.min) g.min = y;
            if (y > g.max) g.max = y;
        }
    }
    const stemX = [];
    const stemY = [];
    let any = false;
    for (const [x, g] of groups) {
        if (g.max === g.min) continue;
        any = true;
        stemX.push(x, x, null);
        stemY.push(g.min, g.max, null);
    }
    if (!any) return null;
    return Object.assign({}, template, { x: stemX, y: stemY });
}

// Filters a scan figure's `measurements` object down to points recorded at or
// before `stepCap` (an inference-step number). Used for the "View at" convergence
// cap and the global play/scrub timeline on the scan measurements plot. Bins with
// no per-point step field (the initial coarse/secondary/tertiary sweep phases,
// which precede adaptive stepping) have no step concept and are always kept in full.
function _filterScanMeasurementsByStep(measurements, stepCap) {
    if (!measurements || stepCap == null) return measurements;
    const out = Object.assign({}, measurements);
    function keepIndices(steps, n) {
        const idx = [];
        for (let i = 0; i < n; i++) {
            const s = steps[i];
            if (s == null || s <= stepCap) idx.push(i);
        }
        return idx;
    }
    function pick(arr, idx) { return idx.map((i) => arr[i]); }
    if (measurements.mode === 'steps' && measurements.step && measurements.x &&
        measurements.step.length === measurements.x.length) {
        const idx = keepIndices(measurements.step, measurements.x.length);
        out.x = pick(measurements.x, idx);
        out.y = pick(measurements.y, idx);
        out.step = pick(measurements.step, idx);
        if (measurements.sweep_index && measurements.sweep_index.length === measurements.x.length) {
            out.sweep_index = pick(measurements.sweep_index, idx);
        }
    } else if (measurements.mode === 'phases' && measurements.fine_step && measurements.fine_x &&
               measurements.fine_step.length === measurements.fine_x.length) {
        const idx = keepIndices(measurements.fine_step, measurements.fine_x.length);
        out.fine_x = pick(measurements.fine_x, idx);
        out.fine_y = pick(measurements.fine_y, idx);
        out.fine_step = pick(measurements.fine_step, idx);
    }
    return out;
}

function _buildScanFigure(def, data) {
    const traces = [];
    const hasMetrics = !!data.has_metrics;
    const sa = hasMetrics ? { xaxis: 'x', yaxis: 'y' } : {};
    const T = def.traces;

    // Dense reference curve (omitted when there's nothing to draw, e.g. real data with
    // no measurements yet). Its label depends on what it actually is: a known parametric
    // ground truth for simulated runs ("true signal"), vs. just the per-bin mean over every
    // recorded shot for real MATLAB runs ("recorded mean signal" — not an independent
    // reference, so it must not be called "true" or "real").
    if (_hasFiniteValues(data.y_dense)) {
        const label = data.true_signal_label || T.true_signal.name;
        traces.push(Object.assign({}, T.true_signal, { x: data.x_dense, y: data.y_dense, name: label }, sa));
    }

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
        // Vertical stems grouping repeated same-frequency shots, drawn first so the
        // per-measurement dots layer on top of them.
        if (m.mode === 'phases') {
            const allX = [].concat(m.coarse_x || [], m.secondary_x || [], m.tertiary_x || [], m.fine_x || []);
            const allY = [].concat(m.coarse_y || [], m.secondary_y || [], m.tertiary_y || [], m.fine_y || []);
            const stem = _buildStemTrace(allX, allY, T.freq_stem);
            if (stem) traces.push(Object.assign({}, stem, sa));
        } else if (m.mode === 'steps') {
            const stem = _buildStemTrace(m.x, m.y, T.freq_stem);
            if (stem) traces.push(Object.assign({}, stem, sa));
        }

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
            // A real acquisition (e.g. MATLAB replay) scans every frequency once per
            // sweep, then scans them all again — sweep_index is that real time axis, and
            // is a truer color than the locator's own adaptive visit order (`step`), which
            // can revisit one bin many sweeps apart from the next. Use it when present.
            const hasSweep = m.sweep_index && m.sweep_index.length === m.x.length;
            const colorIdx = hasSweep
                ? Array.from(m.sweep_index)
                : (m.step && m.step.length === m.x.length) ? Array.from(m.step) : m.x.map((_, i) => i);
            const colorLabel = hasSweep ? 'sweep' : 'step';
            const colorbar = hasMetrics
                ? { title: { text: colorLabel }, len: 0.6, y: 0.8 }
                : { title: { text: colorLabel } };
            traces.push(Object.assign({}, T.steps_noisy, {
                x: m.x, y: m.y,
                marker: Object.assign({}, T.steps_noisy.marker, { color: colorIdx, colorbar }),
                customdata: Array.from(m.y).map(y => _depthPct(y, baseline)),
                hovertemplate: `x=%{x}<br>y=%{y:.4f}<br>down=%{customdata:.1f}%<br>${colorLabel}=%{marker.color}<extra></extra>`,
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

    // Mark what "width" and "contrast" mean on the actual curve: half-depth crossings and
    // baseline-to-trough drop of the most prominent dip(s), measured from the rendered
    // curve itself (see _findDip/_findDips) rather than any model's internal parameters —
    // except the bracketed letter suffix, which names whichever model parameter that
    // geometric reading stands in for (see _paramLetterFor), so it reads with the same
    // symbol as the signal-equation panel. Up to 2 resolved dips are found by local minima
    // (see _findDips), so a split spectrum (e.g. Voigt+Zeeman) gets one readout per dip
    // instead of just the deepest. Measured primarily on the true/real signal
    // (data.y_dense) — real data (data.y_dense all-NaN) has nothing to measure here — and,
    // when the locator's belief/mode curve (data.y_dense_mode) is present, the same
    // geometric reading is taken over that curve's own dip within the real dip's window
    // and shown alongside it, real vs. fit, color-matched to each curve (see T.true_signal
    // / T.mode_signal) so the two numbers read as a direct comparison rather than a second,
    // unrelated measurement.
    if (_hasFiniteValues(data.y_dense) && data.x_dense && data.x_dense.length === data.y_dense.length) {
        const tp = data.true_params && data.true_params.params;
        const widthLetter = _paramLetterFor(tp, ['fwhm_total', 'linewidth', 'homogeneous_linewidth']);
        const contrastLetter = _paramLetterFor(tp, ['c_total', 'dip_depth']);
        const dips = _findDips(data.x_dense, data.y_dense, baseline, 2);
        const realColor = (T.true_signal && T.true_signal.line && T.true_signal.line.color) || 'blue';
        const fitColor = (T.mode_signal && T.mode_signal.line && T.mode_signal.line.color) || '#d62728';
        const hasFit = data.y_dense_mode && data.y_dense_mode.length === data.y_dense.length;
        dips.forEach((dip) => {
            const fitDip = hasFit ? _findDip(data.x_dense, data.y_dense_mode, baseline, dip.iLo, dip.iHi) : null;

            shapes.push({
                type: 'line', xref: 'x', yref: 'y',
                x0: dip.xLeft, x1: dip.xRight, y0: dip.yMin, y1: dip.yMin,
                line: { width: 1.5, color: realColor, dash: 'dot' },
                layer: 'above',
            });
            const widthText = fitDip
                ? `↔ width${widthLetter}: <span style="color:${realColor}">real ≈ ${formatHzValue('linewidth', dip.fwhm)}</span> vs ` +
                  `<span style="color:${fitColor}">fit ≈ ${formatHzValue('linewidth', fitDip.fwhm)}</span>`
                : `↔ width${widthLetter} ≈ ${formatHzValue('linewidth', dip.fwhm)}`;
            extraAnnotations.push({
                text: widthText,
                x: (dip.xLeft + dip.xRight) / 2, xref: 'x',
                y: Math.min(dip.yMin, fitDip ? fitDip.yMin : dip.yMin), yref: 'y',
                yanchor: 'top', xanchor: 'center', yshift: -4,
                showarrow: false, font: { size: 11, color: realColor },
                bgcolor: 'rgba(255,255,255,0.75)',
            });

            shapes.push({
                type: 'line', xref: 'x', yref: 'y',
                x0: dip.xMin, x1: dip.xMin, y0: dip.yMin, y1: baseline,
                line: { width: 1.5, color: realColor, dash: 'dot' },
                layer: 'above',
            });
            const contrastText = fitDip
                ? `↕ contrast${contrastLetter}: <span style="color:${realColor}">real ≈ ${dip.depthPct.toFixed(1)}%</span> vs ` +
                  `<span style="color:${fitColor}">fit ≈ ${fitDip.depthPct.toFixed(1)}%</span>`
                : `↕ contrast${contrastLetter} ≈ ${dip.depthPct.toFixed(1)}%`;
            extraAnnotations.push({
                text: contrastText,
                x: dip.xMin, xref: 'x',
                y: (Math.min(dip.yMin, fitDip ? fitDip.yMin : dip.yMin) + baseline) / 2, yref: 'y',
                yanchor: 'middle', xanchor: 'left', xshift: 8,
                showarrow: false, font: { size: 11, color: realColor },
                bgcolor: 'rgba(255,255,255,0.75)',
            });

            if (fitDip) {
                // Fit's own dip geometry drawn dashed (matching the mode curve's own line
                // dash) and offset a touch from the real markers so the two sit side by
                // side on the chart rather than one hiding the other.
                const xOff = (dip.xRight - dip.xLeft) * 0.08;
                shapes.push({
                    type: 'line', xref: 'x', yref: 'y',
                    x0: fitDip.xLeft, x1: fitDip.xRight, y0: fitDip.yMin, y1: fitDip.yMin,
                    line: { width: 1.5, color: fitColor, dash: 'dash' },
                    layer: 'above',
                });
                shapes.push({
                    type: 'line', xref: 'x', yref: 'y',
                    x0: fitDip.xMin + xOff, x1: fitDip.xMin + xOff, y0: fitDip.yMin, y1: baseline,
                    line: { width: 1.5, color: fitColor, dash: 'dash' },
                    layer: 'above',
                });
            }
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
    if (data.found_params) meta.found_params = data.found_params;
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
