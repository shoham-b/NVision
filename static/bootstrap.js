// Loads manifest.js and settings.js from disk (or uses inlined versions).
// Sets window.NVISION_ASSET_PREFIX, window.MANIFEST, window.SETTINGS before main() runs.
// Exposed as window.NVISION_BOOTSTRAP (a Promise) so reload.js can await it.

if (!window.NVISION_ASSET_PREFIX) {
    window.NVISION_ASSET_PREFIX = '';
}
window.NVISION_BOOTSTRAP = (async () => {
    async function loadScript(candidates, onLoaded) {
        const cacheBust = `v=${Date.now()}`;
        for (const candidate of candidates) {
            const loaded = await new Promise((resolve) => {
                const script = document.createElement('script');
                const sep = candidate.src.includes('?') ? '&' : '?';
                script.src = `${candidate.src}${sep}${cacheBust}`;
                script.async = false;
                script.onload = () => resolve(true);
                script.onerror = () => resolve(false);
                document.head.appendChild(script);
            });
            if (loaded) {
                if (onLoaded) onLoaded(candidate);
                return candidate;
            }
        }
        return null;
    }

    async function fetchManifest(prefix) {
        const candidates = [
            `${prefix}plots_manifest.json`,
            `${prefix}./plots_manifest.json`,
            '../artifacts/plots_manifest.json',
        ];
        for (const url of candidates) {
            try {
                const response = await fetch(url, { cache: 'no-store' });
                if (response.ok) {
                    const data = await response.json();
                    if (Array.isArray(data)) {
                        return data;
                    }
                }
            } catch (e) {
                console.warn(`Failed to fetch manifest from ${url}:`, e);
            }
        }
        return null;
    }

    // If MANIFEST was already inlined by report.py, skip loading manifest.js
    if (window.MANIFEST && Array.isArray(window.MANIFEST) && window.MANIFEST.length > 0) {
        // Already inlined — nothing to do
    } else if (window.MANIFEST === null) {
        // Manifest is too large to inline — fetch it as JSON
        console.log('Large manifest detected, fetching via JSON...');
        const fetched = await fetchManifest(window.NVISION_ASSET_PREFIX);
        if (fetched) {
            window.MANIFEST = fetched;
        } else {
            window.MANIFEST = [];
            console.warn('Could not fetch plots_manifest.json. Using empty manifest.');
        }
    } else {
        // No inline manifest — try loading manifest.js
        const manifestCandidate = await loadScript(
            [
                { src: 'manifest.js', prefix: '' },
                { src: './manifest.js', prefix: '' },
                { src: '../artifacts/manifest.js', prefix: '../artifacts/' },
            ],
            (candidate) => {
                window.NVISION_ASSET_PREFIX = candidate.prefix;
            }
        );
        if (!manifestCandidate) {
            window.NVISION_ASSET_PREFIX = '';
            window.MANIFEST = [];
            console.warn('Could not load manifest.js from current directory or ../artifacts/. Using empty manifest.');
        } else if (window.MANIFEST === null) {
            const fetched = await fetchManifest(window.NVISION_ASSET_PREFIX);
            if (fetched) {
                window.MANIFEST = fetched;
            } else {
                window.MANIFEST = [];
                console.warn('Could not fetch plots_manifest.json. Using empty manifest.');
            }
        }
    }

    // Same for settings — skip if already inlined
    if (!window.SETTINGS) {
        const settingsCandidate = await loadScript([
            { src: 'settings.js' },
            { src: './settings.js' },
            { src: '../artifacts/settings.js' },
        ]);
        if (!settingsCandidate) {
            window.SETTINGS = { out_dir: '', generated_at: null };
            console.warn('Could not load settings.js from current directory or ../artifacts/. Using default settings.');
        }
    }
})();
