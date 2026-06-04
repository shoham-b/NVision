// Sets window.MANIFEST and window.SETTINGS before main() runs.
// report.py inlines both into the HTML; this file only handles fallback cases.
// Exposed as window.NVISION_BOOTSTRAP (a Promise) so reload.js can await it.

if (!window.NVISION_ASSET_PREFIX) {
    window.NVISION_ASSET_PREFIX = '';
}
window.NVISION_BOOTSTRAP = (async () => {
    async function fetchManifest(prefix) {
        const candidates = [
            // Live endpoint when served by nv serve (always fresh)
            { url: `${prefix}api/manifest`, gz: false },
            // Compressed static file (fastest for file:// or CDN)
            { url: `${prefix}plots_manifest.json.gz`, gz: true },
            { url: `${prefix}./plots_manifest.json.gz`, gz: true },
            // Uncompressed fallback
            { url: `${prefix}plots_manifest.json`, gz: false },
            { url: `${prefix}./plots_manifest.json`, gz: false },
            { url: '../artifacts/plots_manifest.json.gz', gz: true },
            { url: '../artifacts/plots_manifest.json', gz: false },
        ];
        for (const { url, gz } of candidates) {
            try {
                const response = await fetch(url, { cache: 'no-store' });
                if (!response.ok) continue;
                let data;
                if (gz) {
                    const ds = new DecompressionStream('gzip');
                    const text = await new Response(response.body.pipeThrough(ds)).text();
                    data = JSON.parse(text);
                } else {
                    // /api/manifest may respond with Content-Encoding: gzip — browser decompresses automatically
                    data = await response.json();
                }
                if (Array.isArray(data)) return data;
            } catch (e) {
                // silently try next candidate
            }
        }
        return null;
    }

    // window.MANIFEST is inlined by report.py as an array (possibly empty) or null (too large to inline).
    // Only fetch if not already set.
    if (Array.isArray(window.MANIFEST)) {
        // Already inlined — nothing to do
    } else {
        // null = too large to inline; undefined = not generated yet
        if (window.MANIFEST === null) {
            console.log('Large manifest detected, fetching via JSON...');
        }
        const fetched = await fetchManifest(window.NVISION_ASSET_PREFIX);
        if (fetched) {
            window.MANIFEST = fetched;
        } else {
            window.MANIFEST = [];
            console.warn('Could not fetch manifest from API or filesystem. Using empty manifest.');
        }
    }

    // window.SETTINGS is inlined by report.py; use defaults if missing.
    if (!window.SETTINGS) {
        window.SETTINGS = { out_dir: '', generated_at: null };
    }
})();
