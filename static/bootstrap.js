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

// Parses the same generator-name patterns as app.js's parseGeneratorFacetsFromNameLegacy
// (main()'s copy is the source of truth for how the UI groups parameter-grid studies into
// one "family" — keep these in sync). Duplicated here, not shared, because this file runs
// before app.js even loads and must decide which generators' data to fetch before any of
// main()'s closures exist. Only the family *key* is needed here (which names belong
// together), not the parsed axis values app.js uses to build the facet dropdowns.
function _nvisionLegacyFamilyKey(name) {
    if (/^NVCenter-saturation_voigt-c[\d.]+-si[\d.]+MHz$/.test(name)) {
        return 'saturation_voigt_saturation_sigma_inhom';
    }
    let m = /^NVCenter-([a-zA-Z_]+)-w[\d.]+MHz-c[\d.]+-si[\d.]+MHz$/.exec(name);
    if (m) return `${m[1]}_width_contrast_sigma_inhom`;
    m = /^NVCenter-([a-zA-Z_]+)-w[\d.]+MHz-c[\d.]+$/.exec(name);
    if (m) return `${m[1]}_width_contrast`;
    return null; // ungrouped — this generator is its own scope
}

// Every generator name that belongs to the same family as `name` (including itself),
// or just `[name]` when it doesn't parse as part of a grid family.
function _nvisionFamilyScope(name, allNames) {
    const key = _nvisionLegacyFamilyKey(name);
    if (!key) return [name];
    return allNames.filter((n) => _nvisionLegacyFamilyKey(n) === key);
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
        try {
            const combos = await fetchJson('api/combos');
            allGenerators = [...new Set(combos.map((c) => c.generator))].sort();
        } catch (e) {
            console.warn('api/combos failed, falling back to an unscoped manifest fetch:', e);
        }
        window.NVISION_ALL_GENERATORS = allGenerators;

        let scopeGenerators = null; // null == unscoped (fallback when combos is empty/unknown)
        if (allGenerators.length && !_nvisionForcedUnscoped()) {
            const hashGenerator = _nvisionHashGenerator();
            const target = hashGenerator && allGenerators.includes(hashGenerator) ? hashGenerator : allGenerators[0];
            scopeGenerators = _nvisionFamilyScope(target, allGenerators);
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

    // Safety valve against a reload loop: this scope's family and app.js's own
    // groupGeneratorsByFamily/parseGeneratorFacetsFromNameLegacy are two independent
    // implementations of the same grouping (see _nvisionLegacyFamilyKey's comment) and
    // are expected to always agree, making one reload always sufficient -- but if they
    // ever drift out of sync, bail out to an unscoped fetch instead of reloading forever.
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
