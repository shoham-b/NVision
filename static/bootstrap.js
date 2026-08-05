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
window.NVISION_BOOTSTRAP = (async () => {
    async function fetchManifest() {
        const prefix = window.NVISION_ASSET_PREFIX;
        // window.NVISION_LIVE_API is set explicitly by report.py's render_index_html —
        // no guessing across candidate URLs: nv serve always has /api/manifest live;
        // a static export (nv matlab, GCS upload) always has plots_manifest.json.gz
        // written next to index.html (see write_plots_manifest).
        if (window.NVISION_LIVE_API) {
            const response = await fetch(`${prefix}api/manifest`, { cache: 'no-store' });
            if (!response.ok) {
                throw new Error(`api/manifest returned ${response.status}`);
            }
            // May respond with Content-Encoding: gzip — browser decompresses automatically.
            return await response.json();
        }
        const response = await fetch(`${prefix}plots_manifest.json.gz`, { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`plots_manifest.json.gz returned ${response.status}`);
        }
        const ds = new DecompressionStream('gzip');
        const text = await new Response(response.body.pipeThrough(ds)).text();
        return JSON.parse(text);
    }

    // window.MANIFEST is inlined by report.py as an array (possibly empty) or null (too large to inline).
    // Only fetch if not already set.
    if (Array.isArray(window.MANIFEST)) {
        // Already inlined — nothing to do
    } else {
        console.log('Large manifest detected, fetching via JSON...');
        const fetched = await fetchManifest();
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
