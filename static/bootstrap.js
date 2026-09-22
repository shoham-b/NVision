// Sets window.MANIFEST and window.SETTINGS before main() runs.
// report.py inlines both into the HTML; this file only handles the case
// where the manifest was too large to inline (window.MANIFEST === null).
// Exposed as window.NVISION_BOOTSTRAP (a Promise) so reload.js can await it.
// A fetch failure here is NOT swallowed — it rejects this promise, and
// reload.js's .catch() shows a visible error banner instead of silently
// rendering an empty results page.

if (!window.NVISION_ASSET_PREFIX) {
    window.NVISION_ASSET_PREFIX = '';
}

// Single source of truth for how a generator's grid_* metrics (recorded by
// nvision/runner/metrics.py, one repeat's worth fetched cheaply per distinct
// generator via /api/generator-grid-info) map to swept-parameter axes and a
// family key. Shared by this file (to decide which generators' data to fetch,
// before app.js's main() even exists) and by app.js (to build the Study/facet
// pickers), so there is exactly one implementation instead of two independent
// name-regex parsers that had to be manually kept in sync (see git history --
// that used to be a real drift hazard, not a hypothetical one).
window.NVISION_GRID_AXIS_ORDER = ['width', 'contrast', 'saturation', 'sigma_inhom', 'hyperfine'];
window.NVISION_GRID_AXIS_INFO = {
    width: { label: 'Width (MHz)', short: 'width', metricsKey: 'grid_linewidth', scaleToMHz: true },
    contrast: { label: 'Contrast', short: 'contrast', metricsKey: 'grid_c_total', scaleToMHz: false },
    saturation: { label: 'Saturation', short: 'saturation', metricsKey: 'grid_saturation', scaleToMHz: false },
    sigma_inhom: { label: 'sigma_inhom (MHz)', short: 'sigma_inhom', metricsKey: 'grid_sigma_inhom', scaleToMHz: true },
    hyperfine: { label: 'Isotope', short: 'isotope', metricsKey: 'grid_hyperfine', scaleToMHz: false },
};

// gridInfo: one generator's {grid_variant, grid_linewidth, ...} dict from
// /api/generator-grid-info (or undefined/null for a generator with no recorded
// grid_* metrics, e.g. a bare non-grid or MATLAB generator). Returns null for
// those; otherwise {family, familyLabel, axes}.
window.nvisionGeneratorAxes = function (gridInfo) {
    if (!gridInfo) return null;
    const variant = gridInfo.grid_variant;
    if (!variant) return null;
    const axes = [];
    for (const key of window.NVISION_GRID_AXIS_ORDER) {
        const info = window.NVISION_GRID_AXIS_INFO[key];
        const raw = gridInfo[info.metricsKey];
        if (raw == null) continue;
        const value = info.scaleToMHz && typeof raw === 'number' ? raw / 1e6 : raw;
        axes.push({ key, label: info.label, value });
    }
    if (!axes.length) return null;
    return {
        family: `${variant}_${axes.map((a) => a.key).join('_')}`,
        familyLabel: `${variant} (${axes.map((a) => window.NVISION_GRID_AXIS_INFO[a.key].short).join(' × ')})`,
        axes,
    };
};

// Every generator name that belongs to the same family as `name` (including itself),
// or just `[name]` when it has no grid_info (not part of any parsed grid family).
function _nvisionFamilyScope(name, allNames, gridInfoByName) {
    const parsed = window.nvisionGeneratorAxes(gridInfoByName[name]);
    if (!parsed) return [name];
    return allNames.filter((n) => {
        const p = window.nvisionGeneratorAxes(gridInfoByName[n]);
        return p && p.family === parsed.family;
    });
}

function _nvisionHashGenerator() {
    try {
        const params = new URLSearchParams((location.hash || '').replace(/^#/, ''));
        return params.get('generator') || null;
    } catch (e) {
        return null;
    }
}

// Escape hatch set by nvisionNavigateToGenerator's drift-safety valve — forces a full,
// unscoped manifest fetch instead of computing a (possibly-wrong) family scope.
function _nvisionForcedUnscoped() {
    try {
        const params = new URLSearchParams((location.hash || '').replace(/^#/, ''));
        return params.get('nvisionUnscoped') === '1';
    } catch (e) {
        return false;
    }
}

window.NVISION_BOOTSTRAP = (async () => {
    const prefix = window.NVISION_ASSET_PREFIX;

    async function fetchJson(path) {
        const response = await fetch(`${prefix}${path}`, { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`${path} returned ${response.status}`);
        }
        return await response.json(); // may be Content-Encoding: gzip; browser decompresses
    }

    async function fetchStaticManifest() {
        const response = await fetch(`${prefix}plots_manifest.json.gz`, { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`plots_manifest.json.gz returned ${response.status}`);
        }
        const ds = new DecompressionStream('gzip');
        const text = await new Response(response.body.pipeThrough(ds)).text();
        return JSON.parse(text);
    }

    // window.MANIFEST is inlined by report.py as an array (possibly empty) or null (too large
    // to inline, or always null in live mode — see render_index_html). Only fetch if not
    // already set.
    if (Array.isArray(window.MANIFEST)) {
        // Already inlined — nothing to do
    } else if (window.NVISION_LIVE_API) {
        // Scope the manifest fetch to the current generator's parameter-study family instead
        // of pulling every combo in the whole cache: on a large shared cache (tens of GB,
        // many unrelated generators/noises/strategies/repeats) that full scan is what made
        // every page load slow, even though any one view only ever looks at one family at a
        // time (see nvision project memory on /api/manifest performance). /api/combos is the
        // cheap picker index (no per-repeat scan) this scoping decision is based on.
        let allGenerators = [];
        let gridInfoByName = {};
        try {
            const [combos, gridInfo] = await Promise.all([
                fetchJson('api/combos'),
                fetchJson('api/generator-grid-info'),
            ]);
            allGenerators = [...new Set(combos.map((c) => c.generator))].sort();
            gridInfoByName = gridInfo;
        } catch (e) {
            console.warn('api/combos or api/generator-grid-info failed, falling back to an unscoped manifest fetch:', e);
        }
        window.NVISION_ALL_GENERATORS = allGenerators;
        // Exposed for app.js so it doesn't need a second fetch of the same data.
        window.NVISION_GENERATOR_GRID_INFO = gridInfoByName;

        let scopeGenerators = null; // null == unscoped (fallback when combos is empty/unknown)
        if (allGenerators.length && !_nvisionForcedUnscoped()) {
            const hashGenerator = _nvisionHashGenerator();
            const target = hashGenerator && allGenerators.includes(hashGenerator) ? hashGenerator : allGenerators[0];
            scopeGenerators = _nvisionFamilyScope(target, allGenerators, gridInfoByName);
        }
        window.NVISION_LOADED_GENERATORS = scopeGenerators;

        const manifestPath = scopeGenerators
            ? `api/manifest?generators=${encodeURIComponent(scopeGenerators.join(','))}`
            : 'api/manifest';
        window.MANIFEST = await fetchJson(manifestPath);
        if (!Array.isArray(window.MANIFEST)) {
            throw new Error('Manifest fetch did not return an array');
        }
    } else {
        console.log('Large manifest detected, fetching via JSON...');
        const fetched = await fetchStaticManifest();
        if (!Array.isArray(fetched)) {
            throw new Error('Manifest fetch did not return an array');
        }
        window.MANIFEST = fetched;
    }

    // window.SETTINGS is inlined by report.py; use defaults if missing.
    if (!window.SETTINGS) {
        window.SETTINGS = { out_dir: '', generated_at: null };
    }
})();

// True when `generatorName`'s data isn't already loaded — i.e. the initial fetch was scoped
// to a different family and this generator wasn't part of it. app.js's generator/study
// picker calls this to decide whether a selection can render instantly from the already-
// loaded `plots` array, or needs nvisionNavigateToGenerator's reload first.
window.nvisionNeedsWiderLoad = function (generatorName) {
    const loaded = window.NVISION_LOADED_GENERATORS;
    return Array.isArray(loaded) && !loaded.includes(generatorName);
};

// Reloads the page scoped to `generatorName`'s family instead of the currently-loaded one.
// A full reload (not an in-place re-fetch) because main() owns its plots/scanPlots/etc. as
// plain `const`s computed once at call time — see nvision-ui's app.js internal structure —
// and is only ever invoked once per page load (recalculate/'r' already reloads the whole
// page for the same reason). Preserves noise/strategy/repeat/view/bayesTab from the current
// hash on a best-effort basis; applyHashToDom() already falls back gracefully if any of them
// don't exist under the new generator.
let _nvisionNavigating = false; // guards against re-entrant calls before location.reload() lands
window.nvisionNavigateToGenerator = function (generatorName) {
    // location.reload() schedules a navigation; it does NOT stop this page's own script from
    // continuing to run in the meantime. app.js's segmented control re-render dispatches a
    // fresh 'controlchange' after setting its value, which re-enters updateScanSignalControls
    // before the reload actually lands — without this guard, every one of those re-entrant
    // calls independently decides a reload is needed and calls this again.
    if (_nvisionNavigating) return;
    _nvisionNavigating = true;

    // Safety valve against a reload loop: this scope's family and app.js's grouping
    // both go through the same window.nvisionGeneratorAxes (see above), so they
    // can't drift out of sync the way two independent regex parsers previously
    // could -- one reload should always be sufficient. Kept as defense-in-depth
    // against any other cause of an unexpected repeat request (e.g. a stale
    // /api/generator-grid-info cache mid-reload) instead of reloading forever.
    let attempted = [];
    try {
        attempted = JSON.parse(sessionStorage.getItem('nvisionGeneratorLoadAttempts') || '[]');
    } catch (e) { /* ignore */ }
    if (attempted.includes(generatorName)) {
        console.error(
            `nvisionNavigateToGenerator: already reloaded once for "${generatorName}" and it still ` +
            'needs a wider load -- bootstrap.js and app.js\'s family grouping have drifted out of ' +
            'sync. Falling back to an unscoped manifest fetch instead of reloading again.'
        );
        const params = new URLSearchParams((location.hash || '').replace(/^#/, ''));
        params.set('generator', generatorName);
        params.set('nvisionUnscoped', '1');
        location.hash = params.toString();
        location.reload();
        return;
    }
    try {
        sessionStorage.setItem('nvisionGeneratorLoadAttempts', JSON.stringify([...attempted, generatorName]));
    } catch (e) { /* ignore */ }

    const params = new URLSearchParams((location.hash || '').replace(/^#/, ''));
    params.set('generator', generatorName);
    location.hash = params.toString();
    location.reload();
};
