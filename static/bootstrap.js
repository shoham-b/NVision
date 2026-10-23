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
window.NVISION_GRID_AXIS_ORDER = ['drive_mode', 'window', 'gaussian_share', 'width', 'contrast', 'saturation', 'sigma_inhom', 'hyperfine'];
window.NVISION_GRID_AXIS_INFO = {
    // Derived from grid_variant ("cw_drive"/"pulsed_drive"), not a recorded grid_* field.
    drive_mode: { label: 'Drive', short: 'drive', metricsKey: 'grid_drive_mode', scaleToMHz: false },
    // Drive-frequency window as a multiple of the signal's total FWHM (the SNR x window study grid).
    window: { label: 'Window (FWHM)', short: 'window', metricsKey: 'grid_window_fwhm', scaleToMHz: false },
    // Gaussian part of the total FWHM at a fixed total FWHM (the SNR x window study); 0 = pure Lorentzian.
    gaussian_share: { label: 'Gaussian share of FWHM', short: 'gauss share', metricsKey: 'grid_gaussian_share', scaleToMHz: false },
    width: { label: 'HWHM (MHz)', short: 'hwhm', metricsKey: 'grid_hwhm', scaleToMHz: true },
    contrast: { label: 'Contrast', short: 'contrast', metricsKey: 'grid_contrast_total', scaleToMHz: false },
    saturation: { label: 'Saturation', short: 'saturation', metricsKey: 'grid_saturation', scaleToMHz: false },
    sigma_inhom: { label: 'sigma_inhom (MHz)', short: 'sigma_inhom', metricsKey: 'grid_sigma_inhom', scaleToMHz: true },
    hyperfine: { label: 'Isotope', short: 'isotope', metricsKey: 'grid_hyperfine', scaleToMHz: false },
};

// gridInfo: one generator's {grid_variant, grid_hwhm, ...} dict from
// /api/generator-grid-info (or undefined/null for a generator with no recorded
// grid_* metrics, e.g. a bare non-grid or MATLAB generator). Returns null for
// those; otherwise {family, familyLabel, axes}.
//
// Physics-level grouping (not the generator class names):
//  - A Lorentzian width x contrast grid generator is the sigma_inhom = 0 slice of the
//    Voigt lineshape (sigma_inhom=0 draws lorentz_frac=1, see presets.py), so it joins
//    the Voigt family with sigma_inhom = 0 and its own isotope instead of being its own Study.
//  - CW and pulsed drive generators are one "drive" family with a CW/pulsed axis; drive
//    only changes the physics for locators that also choose the MW amplitude.
const NVISION_DRIVE_MODES = { cw_drive: 'cw', pulsed_drive: 'pulsed' };
const NVISION_FAMILY_LABELS = { voigt: 'Lorentzian / Voigt', drive: 'CW / pulsed drive' };

window.nvisionGeneratorAxes = function (rawGridInfo) {
    if (!rawGridInfo) return null;
    let variant = rawGridInfo.grid_variant;
    if (!variant) return null;
    const gridInfo = { ...rawGridInfo };
    if (variant === 'lorentzian' && gridInfo.grid_hwhm != null && gridInfo.grid_sigma_inhom == null) {
        variant = 'voigt';
        gridInfo.grid_sigma_inhom = 0;
    } else if (variant in NVISION_DRIVE_MODES) {
        gridInfo.grid_drive_mode = NVISION_DRIVE_MODES[variant];
        variant = 'drive';
    }
    const axes = [];
    // The SNR study fixes the contrast and the total FWHM and varies the line shape by its Gaussian share; a run is
    // picked by SNR, window and share, so the contrast, HWHM and sigma_inhom (which only restate the share) are no axes.
    const studyFixed = gridInfo.grid_window_fwhm != null ? new Set(['width', 'contrast', 'sigma_inhom']) : new Set();
    for (const key of window.NVISION_GRID_AXIS_ORDER) {
        if (studyFixed.has(key)) continue;
        const info = window.NVISION_GRID_AXIS_INFO[key];
        const raw = gridInfo[info.metricsKey];
        if (raw == null) continue;
        const value = info.scaleToMHz && typeof raw === 'number' ? raw / 1e6 : raw;
        axes.push({ key, label: info.label, value });
    }
    if (!axes.length) return null;
    return {
        family: `${variant}_${axes.map((a) => a.key).join('_')}`,
        familyLabel: `${NVISION_FAMILY_LABELS[variant] || variant} (${axes.map((a) => window.NVISION_GRID_AXIS_INFO[a.key].short).join(' × ')})`,
        axes,
    };
};

// "SNR 20" for an `SNR(20)` noise name (the name of a run of the SNR studies, which carries no sigma); for an older
// `Gauss(sigma)` name only when the signal's contrast_total is given (SNR = contrast / sigma, "no noise" for sigma 0);
// null otherwise.
window.nvisionNoiseSnrLabel = function (noiseName, contrastTotal) {
    const snr = /^SNR\(([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\)/.exec(noiseName);
    if (snr) return `SNR ${Number(snr[1])}`;
    const m = /^Gauss\(([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\)/.exec(noiseName);
    if (!m || !(contrastTotal > 0)) return null;
    const sigma = parseFloat(m[1]);
    return sigma > 0 ? `SNR ${Number((contrastTotal / sigma).toPrecision(3))}` : 'no noise';
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
        let fullGenerator = null; // the one generator that also gets its non-scan (bayesian_*) entries
        if (allGenerators.length && !_nvisionForcedUnscoped()) {
            const hashGenerator = _nvisionHashGenerator();
            const target = hashGenerator && allGenerators.includes(hashGenerator) ? hashGenerator : allGenerators[0];
            scopeGenerators = _nvisionFamilyScope(target, allGenerators, gridInfoByName);
            fullGenerator = target;
        }
        window.NVISION_LOADED_GENERATORS = scopeGenerators;
        window.NVISION_FULL_GENERATOR = fullGenerator;
        try {
            // This generator is now fully loaded, so a later switch back to it may reload once again.
            const attempts = JSON.parse(sessionStorage.getItem('nvisionGeneratorLoadAttempts') || '[]');
            sessionStorage.setItem('nvisionGeneratorLoadAttempts', JSON.stringify(attempts.filter((g) => g !== fullGenerator)));
        } catch (e) { /* ignore */ }

        const manifestPath = scopeGenerators
            ? `api/manifest?generators=${encodeURIComponent(scopeGenerators.join(','))}&full=${encodeURIComponent(fullGenerator)}`
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
    // Only the one `full` generator carries its non-scan (bayesian_*) entries; the rest of the
    // family is scan-only, so selecting any other generator needs a reload scoped to it.
    const loaded = window.NVISION_LOADED_GENERATORS;
    if (!Array.isArray(loaded)) return false;
    return !loaded.includes(generatorName) || generatorName !== window.NVISION_FULL_GENERATOR;
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
