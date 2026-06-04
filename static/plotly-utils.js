// Plotly loading and figure decode/fetch utilities.
// All functions are global — depended on by app.js main().
//
// resolveAssetPath  — prepends window.NVISION_ASSET_PREFIX to relative paths
// ensurePlotly      — lazy-loads Plotly from CDN on first call
// _fetchJson        — fetches a .json or .json.gz file, decodes arrays
// _decodePlotlyFigure — recursively decodes {__f32__} and {bdata,dtype} arrays
// _decodeBase64F32  — base64 → Float32Array (our custom format)
// _decodeBase64Typed — base64 → Float32Array|Float64Array (Plotly Python 5.x format)

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
    const resp = await fetch(resolveAssetPath(url), { cache: 'no-store' });
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
