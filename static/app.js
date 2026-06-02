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

function renderRunStatusBanner(statusData) {
    const banner = document.getElementById('run-status-banner');
    if (!banner) return;
    banner.className = '';
    banner.innerHTML = '';
    banner.hidden = true;

    if (!statusData || typeof statusData !== 'object') return;
    const status = statusData.status;
    const total = statusData.total_tasks;
    const completed = statusData.completed_tasks;

    if (status === 'scheduled') {
        banner.hidden = false;
        banner.classList.add('run-status-banner', 'run-status-scheduled');
        banner.innerHTML = '<span class="run-status-spinner"></span> Run scheduled — waiting to start.';
    } else if (status === 'running') {
        banner.hidden = false;
        banner.classList.add('run-status-banner', 'run-status-running');
        const progressText = (typeof completed === 'number' && typeof total === 'number')
            ? ` (${completed}/${total} tasks)`
            : '';
        banner.innerHTML = '<span class="run-status-spinner"></span> Run in progress' + progressText + ' — results will update automatically. You can refresh the page to see new data.';
    } else if (status === 'partial') {
        banner.hidden = false;
        banner.classList.add('run-status-banner', 'run-status-partial');
        banner.textContent = 'Partial results — the run was interrupted before all tasks completed. Some combinations may be missing.';
    } else if (status === 'error') {
        banner.hidden = false;
        banner.classList.add('run-status-banner', 'run-status-error');
        banner.textContent = 'Run encountered errors. Results may be incomplete.';
    }
}

async function fetchRunStatus() {
    const candidates = [
        'run_status.json',
        './run_status.json',
        '../artifacts/run_status.json',
    ];
    for (const url of candidates) {
        try {
            const response = await fetch(url, { cache: 'no-store' });
            if (response.ok) {
                const data = await response.json();
                if (data && typeof data === 'object') {
                    return data;
                }
            }
        } catch (e) {
            // ignore fetch errors
        }
    }
    return null;
}

function startRunStatusPolling() {
    let lastStatus = null;
    const interval = setInterval(async () => {
        const data = await fetchRunStatus();
        if (!data) {
            // If we previously saw a running status and now the file is gone,
            // assume the run finished and suggest a refresh.
            if (lastStatus === 'running' || lastStatus === 'scheduled') {
                clearInterval(interval);
                const banner = document.getElementById('run-status-banner');
                if (banner) {
                    banner.className = 'run-status-banner run-status-complete';
                    banner.innerHTML = 'Run complete! <button class="run-status-refresh-btn" onclick="window.location.reload()">Refresh page</button> to see the latest results.';
                    banner.hidden = false;
                }
            }
            return;
        }
        lastStatus = data.status;
        renderRunStatusBanner(data);
        if (data.status !== 'running' && data.status !== 'scheduled') {
            clearInterval(interval);
        }
    }, 3000);
}

function initRunStatusBanner() {
    const inlineStatus = window.RUN_STATUS || null;
    renderRunStatusBanner(inlineStatus);
    const shouldPoll = inlineStatus && (inlineStatus.status === 'running' || inlineStatus.status === 'scheduled');
    if (shouldPoll || !inlineStatus) {
        // Also fetch fresh status in case the inlined one is stale
        fetchRunStatus().then((fresh) => {
            if (fresh) {
                renderRunStatusBanner(fresh);
                if (fresh.status === 'running' || fresh.status === 'scheduled') {
                    startRunStatusPolling();
                }
            }
        });
    }
}

function initHelpToggles() {
    document.addEventListener('click', (e) => {
        const btn = e.target.closest('.help-toggle');
        if (!btn) return;
        const targetId = btn.getAttribute('aria-controls');
        if (!targetId) return;
        const panel = document.getElementById(targetId);
        if (!panel) return;
        const isHidden = panel.hidden;
        panel.hidden = !isHidden;
        btn.setAttribute('aria-expanded', String(isHidden));
    });
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        const btn = e.target.closest('.help-toggle');
        if (!btn) return;
        e.preventDefault();
        btn.click();
    });
}

function main() {
    initRunStatusBanner();
    initHelpToggles();

    let plots = [];
    let currentPlot = null;
    try {
        plots = window.MANIFEST;
        if (!Array.isArray(plots)) {
            throw new Error('Invalid manifest format');
        }
    } catch (error) {
        console.error('Error reading plots manifest from window.MANIFEST:', error);
        // Show an error message to the user
        const errorDiv = document.createElement('div');
        errorDiv.setAttribute('role', 'alert');
        errorDiv.style.padding = '20px';
        errorDiv.style.margin = '20px';
        errorDiv.style.border = '1px solid #f5c6cb';
        errorDiv.style.backgroundColor = '#f8d7da';
        errorDiv.style.color = '#721c24';
        errorDiv.style.borderRadius = '4px';
        errorDiv.innerHTML = '<h3>Error: Invalid data format</h3><p>Could not parse the plot data. The data might be corrupted or in an unexpected format.</p>';
        document.body.appendChild(errorDiv);
        return;
    }

    const settings = window.SETTINGS || {};
    const outDirDisplay = document.getElementById('out-dir-display');
    if (outDirDisplay && settings.out_dir) {
        outDirDisplay.textContent = `(${settings.out_dir})`;
    }

    const locatorResultsLink = document.querySelector('a[download][href="locator_results.csv"]');
    if (locatorResultsLink) {
        locatorResultsLink.href = resolveAssetPath('locator_results.csv');
    }

    plots.forEach((p) => {
        if (p && typeof p.path === 'string' && p.path.length > 0) {
            p.path = resolveAssetPath(p.path);
        }
    });

    const scanPlots = plots.filter((p) => p.type === 'scan');
    const bayesSection = document.getElementById('bayes-section-container');
    const bayesImage = document.getElementById('bayes-image');
    const bayesPlots = plots.filter((p) => p.type === 'bayesian');
    // New JSON-data types (generated by plots_data.py writers)
    const bayesPosteriorDataPlots = plots.filter((p) => p.type === 'bayesian_posterior_data');
    const bayesParamConvDataPlots = plots.filter((p) => p.type === 'bayesian_parameter_convergence_data');
    const bayesConvMetricsDataPlots = plots.filter((p) => p.type === 'bayesian_convergence_metrics_data');
    const bayesFisherDataPlots = plots.filter((p) => p.type === 'bayesian_fisher_data');
    // Legacy iframe types (kept for backward compatibility with cached runs)
    const bayesInteractivePlots = plots.filter((p) => p.type === 'bayesian_interactive');
    const bayesFisherPlots = plots.filter((p) => p.type === 'bayesian_fisher_information');
    const bayesFisherPairsPlots = plots.filter((p) => p.type === 'bayesian_fisher_crlb_pairs');
    const bayesEllipsePlots = plots.filter((p) => p.type === 'bayesian_covariance_ellipses_data');
    const bayesConvergencePlots = plots.filter((p) => p.type === 'bayesian_parameter_convergence');
    const bayesConvMetricsPlots = plots.filter((p) => p.type === 'bayesian_convergence_metrics');
    const bayesJitterPlots = plots.filter((p) => p.type === 'bayesian_jitter');

    const bayesInteractiveSection = document.getElementById('bayes-interactive-section');
    const bayesInteractiveIframe = document.getElementById('bayes-interactive-iframe');
    const bayesConvergenceSection = document.getElementById('bayes-convergence-section');
    const bayesConvergenceIframe = document.getElementById('bayes-convergence-iframe');
    const bayesConvMetricsIframe = document.getElementById('bayes-conv-metrics-iframe');
    const bayesFisherSection = document.getElementById('bayes-fisher-section');
    const bayesFisherIframe = document.getElementById('bayes-fisher-iframe');
    const bayesFisherPairsSection = document.getElementById('bayes-fisher-pairs-section');
    const bayesFisherPairsIframe = document.getElementById('bayes-fisher-pairs-iframe');
    const bayesCovarianceSection = document.getElementById('bayes-covariance-section');
    const bayesEllipseContainer = document.getElementById('bayes-ellipse-container');
    const bayesJitterContent = document.getElementById('bayes-jitter-content');
    const bayesStatsPlots = plots.filter((p) => p.type === 'bayesian_stats');
    const bayesStatsSection = document.getElementById('bayes-stats-section');
    const posteriorHistoryImage = document.getElementById('posterior-history-image');
    const convergenceStatsImage = document.getElementById('convergence-stats-image');
    const hasBayesArtifacts =
        bayesPlots.length > 0 ||
        bayesPosteriorDataPlots.length > 0 ||
        bayesInteractivePlots.length > 0 ||
        bayesParamConvDataPlots.length > 0 ||
        bayesConvMetricsDataPlots.length > 0 ||
        bayesFisherDataPlots.length > 0 ||
        bayesStatsPlots.length > 0 ||
        plots.some((p) => p.type === 'bayesian_parameter_convergence') ||
        plots.some((p) => p.type === 'bayesian_convergence_metrics') ||
        plots.some((p) => p.type === 'bayesian_fisher_bounds') ||
        plots.some((p) => p.type === 'bayesian_fisher_crlb_pairs') ||
        plots.some((p) => p.type === 'bayesian_jitter') ||
        plots.some((p) => p.type === 'bayesian_covariance_ellipses_data');
    // Global Timeline Controls Sync Controller
    const globalControls = document.getElementById('global-timeline-controls');
    const globalPlayBtn = document.getElementById('global-play-btn');
    const globalSlider = document.getElementById('global-range-slider');
    const globalLabel = document.getElementById('global-step-label');
    const globalSpeedSelect = document.getElementById('global-speed-select');

    let globalIsPlaying = false;
    let globalPlayInterval = null;
    let globalTotalFrames = 0;
    let globalStepValues = [];

    // Track registered iframes
    const activeIframes = new Set();

    function registerIframeForTimeline(iframe) {
        if (!iframe) return;

        function onIframeLoaded() {
            try {
                const win = iframe.contentWindow;
                if (win && win.showFrame && win.totalFrames !== undefined) {
                    activeIframes.add(iframe);
                    updateGlobalTimelineMetadata();
                } else {
                    let attempts = 0;
                    const poll = setInterval(() => {
                        attempts++;
                        if (win && win.showFrame && win.totalFrames !== undefined) {
                            activeIframes.add(iframe);
                            updateGlobalTimelineMetadata();
                            clearInterval(poll);
                        }
                        if (attempts > 30) clearInterval(poll);
                    }, 100);
                }
            } catch (e) {
                console.error("Timeline registration cross-origin error:", e);
            }
        }

        iframe.addEventListener('load', onIframeLoaded);
        onIframeLoaded();
    }

    function updateGlobalTimelineMetadata() {
        let maxFrames = 0;
        let steps = [];
        for (const iframe of activeIframes) {
            try {
                const win = iframe.contentWindow;
                if (win && win.totalFrames > maxFrames) {
                    maxFrames = win.totalFrames;
                    if (win.stepValues && win.stepValues.length > 0) {
                        steps = win.stepValues;
                    }
                }
            } catch (e) { }
        }

        if (maxFrames > 0) {
            globalTotalFrames = maxFrames;
            globalStepValues = steps;
            globalSlider.max = maxFrames - 1;
            globalControls.style.display = 'flex';
            updateGlobalLabel(parseInt(globalSlider.value, 10));
            syncFrames(parseInt(globalSlider.value, 10));
        } else {
            globalTotalFrames = 0;
            globalStepValues = [];
            globalControls.style.display = 'none';
            if (globalIsPlaying) {
                globalPause();
            }
        }
    }

    function updateGlobalLabel(idx) {
        const val = (globalStepValues && globalStepValues[idx] !== undefined) ? globalStepValues[idx] : idx;
        globalLabel.innerHTML = `Step: ${val} (${idx + 1} / ${globalTotalFrames})`;
    }

    function syncFrames(idx) {
        updateGlobalLabel(idx);
        for (const iframe of activeIframes) {
            try {
                const win = iframe.contentWindow;
                if (win && typeof win.showFrame === 'function') {
                    const localIdx = Math.min(idx, win.totalFrames - 1);
                    win.showFrame(localIdx);
                }
            } catch (e) { }
        }
    }

    globalSlider.addEventListener('input', (e) => {
        const idx = parseInt(e.target.value, 10);
        syncFrames(idx);
    });

    globalPlayBtn.addEventListener('click', () => {
        if (globalIsPlaying) {
            globalPause();
        } else {
            globalPlay();
        }
    });

    globalSpeedSelect.addEventListener('change', () => {
        if (globalIsPlaying) {
            globalPause();
            globalPlay();
        }
    });

    function globalPlay() {
        globalIsPlaying = true;
        globalPlayBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block; vertical-align:middle; margin-right:4px;"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg><span>Pause</span>';
        globalPlayBtn.style.background = '#ffebe6';
        globalPlayBtn.style.borderColor = '#ff4d4d';
        globalPlayBtn.style.color = '#960000';

        const intervalMs = parseInt(globalSpeedSelect.value, 10);
        globalPlayInterval = setInterval(() => {
            let nextIdx = parseInt(globalSlider.value, 10) + 1;
            if (nextIdx >= globalTotalFrames) {
                nextIdx = 0;
            }
            globalSlider.value = nextIdx;
            syncFrames(nextIdx);
        }, intervalMs);
    }

    function globalPause() {
        globalIsPlaying = false;
        globalPlayBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block; vertical-align:middle; margin-right:4px;"><path d="M8 5v14l11-7z"/></svg><span>Play</span>';
        globalPlayBtn.style.background = '#e6f2ff';
        globalPlayBtn.style.borderColor = '#1e90ff';
        globalPlayBtn.style.color = '#005b96';

        clearInterval(globalPlayInterval);
    }

    function updateBayesView(selectedPlot) {
        if (!bayesSection || !bayesImage) {
            return;
        }

        if (!selectedPlot) {
            bayesImage.hidden = true;
            bayesImage.removeAttribute('src');
            return;
        }

        const bayesPlot = bayesPlots.find(
            (p) =>
                p.generator === selectedPlot.generator &&
                p.noise === selectedPlot.noise &&
                p.strategy === selectedPlot.strategy &&
                p.repeat === selectedPlot.repeat
        );

        if (!bayesPlot) {
            bayesImage.hidden = true;
            bayesImage.removeAttribute('src');
            return;
        }

        bayesImage.src = bayesPlot.path;
        bayesImage.hidden = false;
    }

    // Helper: get-or-create a div container for JSON-rendered plots inside a section.
    // The iframe is hidden when the div is used, and cleared when neither is available.
    function _jsonContainer(section, iframeEl, divId) {
        let div = document.getElementById(divId);
        if (!div) {
            div = document.createElement('div');
            div.id = divId;
            section.appendChild(div);
        }
        div.style.display = 'block';
        if (iframeEl) { iframeEl.style.display = 'none'; iframeEl.src = ''; }
        return div;
    }

    function _iframeContainer(section, iframeEl, divId, src) {
        const div = document.getElementById(divId);
        if (div) { div.style.display = 'none'; div.innerHTML = ''; }
        if (iframeEl) { iframeEl.style.display = ''; iframeEl.src = src; }
    }

    function _matchesSelected(p, sel) {
        return p.generator === sel.generator &&
               p.noise === sel.noise &&
               p.strategy === sel.strategy &&
               p.repeat === sel.repeat;
    }

    function updateBayesInteractiveView(selectedPlot) {
        if (!bayesInteractiveSection || !bayesInteractiveIframe) {
            return;
        }

        // Clear registered iframes/adapters and reset the slider
        activeIframes.clear();
        globalTotalFrames = 0;
        globalStepValues = [];
        globalSlider.value = 0;
        globalSlider.max = 0;
        updateGlobalLabel(0);

        if (!selectedPlot) {
            bayesInteractiveSection.dataset.available = 'false';
            bayesInteractiveIframe.src = '';
            bayesConvergenceSection.dataset.available = 'false';
            bayesConvergenceIframe.src = '';
            bayesConvMetricsIframe.src = '';
            bayesFisherSection.dataset.available = 'false';
            bayesFisherIframe.src = '';
            bayesFisherPairsSection.dataset.available = 'false';
            bayesFisherPairsIframe.src = '';
            bayesCovarianceSection.dataset.available = 'false';
            bayesEllipseContainer.innerHTML = '';
            bayesJitterContent.innerHTML = '';
            ['bayes-posterior-container', 'bayes-conv-json-container',
             'bayes-conv-metrics-json-container', 'bayes-fisher-json-container'].forEach((id) => {
                const el = document.getElementById(id);
                if (el) { el.style.display = 'none'; el.innerHTML = ''; }
            });
            return;
        }

        // --- Posterior animation ---
        const posteriorDataPlot = bayesPosteriorDataPlots.find((p) => _matchesSelected(p, selectedPlot));
        const interactivePlot = bayesInteractivePlots.find((p) => _matchesSelected(p, selectedPlot));

        if (posteriorDataPlot) {
            const div = _jsonContainer(bayesInteractiveSection, bayesInteractiveIframe, 'bayes-posterior-container');
            bayesInteractiveSection.dataset.available = 'true';
            renderPosterior(div, posteriorDataPlot.path, (adapter) => {
                registerTimelineAdapter(adapter);
            });
        } else if (interactivePlot) {
            _iframeContainer(bayesInteractiveSection, bayesInteractiveIframe, 'bayes-posterior-container', interactivePlot.path);
            bayesInteractiveIframe.style.height = '85vh';
            bayesInteractiveSection.dataset.available = 'true';
            registerIframeForTimeline(bayesInteractiveIframe);
        } else {
            bayesInteractiveSection.dataset.available = 'false';
            bayesInteractiveIframe.src = '';
            bayesInteractiveIframe.style.height = '';
        }

        // --- Parameter convergence ---
        const paramConvDataPlot = bayesParamConvDataPlots.find((p) => _matchesSelected(p, selectedPlot));
        const convergencePlot = plots.find((p) => p.type === 'bayesian_parameter_convergence' && _matchesSelected(p, selectedPlot));
        const convMetricsDataPlot = bayesConvMetricsDataPlots.find((p) => _matchesSelected(p, selectedPlot));
        const convMetricsPlot = plots.find((p) => p.type === 'bayesian_convergence_metrics' && _matchesSelected(p, selectedPlot));

        if (paramConvDataPlot || convMetricsDataPlot) {
            bayesConvergenceSection.dataset.available = 'true';
            if (paramConvDataPlot) {
                const div = _jsonContainer(bayesConvergenceSection, bayesConvergenceIframe, 'bayes-conv-json-container');
                renderParameterConvergence(div, paramConvDataPlot.path);
            } else {
                _iframeContainer(bayesConvergenceSection, bayesConvergenceIframe, 'bayes-conv-json-container', convergencePlot ? convergencePlot.path : '');
            }
            if (convMetricsDataPlot) {
                const div = _jsonContainer(bayesConvergenceSection, bayesConvMetricsIframe, 'bayes-conv-metrics-json-container');
                renderConvergenceMetrics(div, convMetricsDataPlot.path);
            } else {
                _iframeContainer(bayesConvergenceSection, bayesConvMetricsIframe, 'bayes-conv-metrics-json-container', convMetricsPlot ? convMetricsPlot.path : '');
            }
        } else if (convergencePlot || convMetricsPlot) {
            bayesConvergenceSection.dataset.available = 'true';
            _iframeContainer(bayesConvergenceSection, bayesConvergenceIframe, 'bayes-conv-json-container', convergencePlot ? convergencePlot.path : '');
            _iframeContainer(bayesConvergenceSection, bayesConvMetricsIframe, 'bayes-conv-metrics-json-container', convMetricsPlot ? convMetricsPlot.path : '');
        } else {
            bayesConvergenceSection.dataset.available = 'false';
            bayesConvergenceIframe.src = '';
            bayesConvMetricsIframe.src = '';
        }

        // --- Fisher information ---
        const fisherDataPlot = bayesFisherDataPlots.find((p) => _matchesSelected(p, selectedPlot));
        const fisherPlot = plots.find((p) => p.type === 'bayesian_fisher_bounds' && _matchesSelected(p, selectedPlot));
        const fisherPairsPlot = plots.find((p) => p.type === 'bayesian_fisher_crlb_pairs' && _matchesSelected(p, selectedPlot));

        if (fisherDataPlot) {
            const div = _jsonContainer(bayesFisherSection, bayesFisherIframe, 'bayes-fisher-json-container');
            bayesFisherSection.dataset.available = 'true';
            bayesFisherPairsSection.dataset.available = 'false';
            bayesFisherPairsIframe.src = '';
            renderFisher(div, fisherDataPlot.path);
        } else if (fisherPlot) {
            _iframeContainer(bayesFisherSection, bayesFisherIframe, 'bayes-fisher-json-container', fisherPlot.path);
            bayesFisherSection.dataset.available = 'true';
            if (fisherPairsPlot) {
                bayesFisherPairsIframe.src = fisherPairsPlot.path;
                bayesFisherPairsSection.dataset.available = 'true';
            } else {
                bayesFisherPairsSection.dataset.available = 'false';
                bayesFisherPairsIframe.src = '';
            }
        } else {
            bayesFisherSection.dataset.available = 'false';
            bayesFisherIframe.src = '';
            bayesFisherPairsSection.dataset.available = 'false';
            bayesFisherPairsIframe.src = '';
        }

        // --- Covariance ellipses (already JSON-based) ---
        const ellipsePlot = plots.find((p) => p.type === 'bayesian_covariance_ellipses_data' && _matchesSelected(p, selectedPlot));
        if (ellipsePlot) {
            renderCovarianceEllipses(bayesEllipseContainer, ellipsePlot.path, (adapter) => {
                registerTimelineAdapter(adapter);
            });
        } else {
            bayesEllipseContainer.innerHTML = '';
        }

        // --- Jitter ---
        const jitterPlot = plots.find((p) => p.type === 'bayesian_jitter' && _matchesSelected(p, selectedPlot));
        if (jitterPlot && bayesJitterContent) {
            renderJitterView(bayesJitterContent, jitterPlot);
        } else if (bayesJitterContent) {
            bayesJitterContent.innerHTML = '';
        }

        if (ellipsePlot || jitterPlot) {
            bayesCovarianceSection.dataset.available = 'true';
        } else {
            bayesCovarianceSection.dataset.available = 'false';
        }
    }

    function updateBayesStatsView(selectedPlot) {
        if (!bayesStatsSection || !posteriorHistoryImage || !convergenceStatsImage) {
            return;
        }

        if (!selectedPlot) {
            bayesStatsSection.dataset.available = 'false';
            return;
        }

        const posteriorPlot = bayesStatsPlots.find(
            (p) =>
                p.kind === 'posterior_history' &&
                p.generator === selectedPlot.generator &&
                p.noise === selectedPlot.noise &&
                p.strategy === selectedPlot.strategy &&
                p.repeat === selectedPlot.repeat
        );

        const convergencePlot = bayesStatsPlots.find(
            (p) =>
                p.kind === 'convergence_stats' &&
                p.generator === selectedPlot.generator &&
                p.noise === selectedPlot.noise &&
                p.strategy === selectedPlot.strategy &&
                p.repeat === selectedPlot.repeat
        );

        if (posteriorPlot) {
            posteriorHistoryImage.src = posteriorPlot.path;
            posteriorHistoryImage.hidden = false;
        } else {
            posteriorHistoryImage.hidden = true;
        }

        if (convergencePlot) {
            convergenceStatsImage.src = convergencePlot.path;
            convergenceStatsImage.hidden = false;
        } else {
            convergenceStatsImage.hidden = true;
        }

        if (posteriorPlot || convergencePlot) {
            bayesStatsSection.dataset.available = 'true';
        } else {
            bayesStatsSection.dataset.available = 'false';
        }
    }



    function updateBayesTabs() {
        const tabBar = document.getElementById('bayes-tab-bar');
        if (!tabBar) return;

        const sections = [
            { id: 'bayes-interactive-section', label: 'Posterior Evolution' },
            { id: 'bayes-convergence-section', label: 'Convergence' },
            { id: 'bayes-fisher-section', label: 'Fisher Bounds' },
            { id: 'bayes-fisher-pairs-section', label: 'CRLB Pairs' },
            { id: 'bayes-covariance-section', label: 'Covariance' },
            { id: 'bayes-stats-section', label: 'Statistics' },
        ];

        const available = sections.filter((s) => {
            const el = document.getElementById(s.id);
            return el && el.dataset.available === 'true';
        });

        const bayesSectionContainer = document.getElementById('bayes-section-container');
        const noDataMsg = document.getElementById('bayes-no-data-message');

        if (bayesPlots.length === 0 && bayesInteractivePlots.length === 0 &&
                bayesPosteriorDataPlots.length === 0 && bayesParamConvDataPlots.length === 0 &&
                bayesConvMetricsDataPlots.length === 0 && bayesFisherDataPlots.length === 0 &&
                bayesEllipsePlots.length === 0 && bayesJitterPlots.length === 0) {
            if (bayesSectionContainer) bayesSectionContainer.style.display = 'none';
            return;
        } else if (bayesSectionContainer) {
            bayesSectionContainer.style.display = 'block';
        }

        if (available.length === 0) {
            tabBar.style.display = 'none';
            if (noDataMsg) noDataMsg.style.display = 'block';
        } else {
            tabBar.style.display = 'flex';
            if (noDataMsg) noDataMsg.style.display = 'none';
        }

        const currentActive = tabBar.querySelector('.bayes-tab-button.is-active');
        let activeId = currentActive ? currentActive.dataset.tab : null;
        if (!available.some((s) => s.id === activeId)) {
            activeId = available[0] ? available[0].id : null;
        }

        tabBar.innerHTML = '';
        for (const s of available) {
            const btn = document.createElement('button');
            btn.className = 'bayes-tab-button' + (s.id === activeId ? ' is-active' : '');
            btn.type = 'button';
            btn.textContent = s.label;
            btn.setAttribute("role", "tab");
            btn.setAttribute("aria-selected", s.id === activeId ? "true" : "false");
            btn.tabIndex = s.id === activeId ? 0 : -1;
            btn.setAttribute("aria-controls", s.id);
            btn.setAttribute("id", "tab-" + s.id);
            const panel = document.getElementById(s.id);
            if (panel) {
                panel.setAttribute("aria-labelledby", "tab-" + s.id);
            }
            btn.dataset.tab = s.id;
            btn.addEventListener('click', () => {
                tabBar.querySelectorAll('.bayes-tab-button').forEach((b) => {
                    b.classList.remove('is-active');
                    b.setAttribute("aria-selected", "false");
                    b.tabIndex = -1;
                });
                btn.setAttribute("aria-selected", "true");
                btn.tabIndex = 0;
                btn.classList.add('is-active');
                sections.forEach((sec) => {
                    const el = document.getElementById(sec.id);
                    if (el) el.classList.toggle('is-active', sec.id === s.id);
                });
            });
            btn.addEventListener('keydown', (e) => {
                const buttons = Array.from(tabBar.querySelectorAll('.bayes-tab-button'));
                const index = buttons.indexOf(btn);
                let nextIndex = null;
                if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                    nextIndex = (index + 1) % buttons.length;
                } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                    nextIndex = (index - 1 + buttons.length) % buttons.length;
                }
                if (nextIndex !== null) {
                    e.preventDefault();
                    buttons[nextIndex].focus();
                    buttons[nextIndex].click();
                }
            });
            tabBar.appendChild(btn);
        }

        sections.forEach((s) => {
            const el = document.getElementById(s.id);
            if (el) el.classList.toggle('is-active', s.id === activeId);
        });
    }

    const scanDefault = scanPlots.length > 0 ? scanPlots[0] : null;

    const scanGenerator = document.getElementById('scan-generator');
    const scanNoise = document.getElementById('scan-noise');
    const scanStrategy = document.getElementById('scan-strategy');
    const scanRepeat = document.getElementById('scan-repeat');
    const scanRepeatPrev = document.getElementById('scan-repeat-prev');
    const scanRepeatNext = document.getElementById('scan-repeat-next');
    const scanIframe = document.getElementById('scan-iframe');
    const scanMetrics = document.getElementById('scan-metrics');
    const scanSweepMetrics = document.getElementById('scan-sweep-metrics');
    let currentRepeatItems = [];
    let measurementDistributionVisible = null;

    function isMeasurementDistributionTrace(trace) {
        const name = (trace && trace.name) ? String(trace.name).trim().toLowerCase() : '';
        return name === 'measurement distribution';
    }

    function resolveTraceVisibleState(value) {
        if (Array.isArray(value)) {
            if (value.length === 0) {
                return null;
            }
            return resolveTraceVisibleState(value[0]);
        }
        if (value === true) {
            return true;
        }
        if (value === false || value === 'legendonly') {
            return false;
        }
        return null;
    }

    function applyMeasurementDistributionPreferenceInScanIframe() {
        if (!scanIframe || measurementDistributionVisible === null) {
            return;
        }
        const frameWindow = scanIframe.contentWindow;
        const frameDocument = scanIframe.contentDocument;
        if (!frameWindow || !frameDocument || !frameWindow.Plotly) {
            return;
        }
        const graphDiv = frameDocument.querySelector('.plotly-graph-div');
        if (!graphDiv || !Array.isArray(graphDiv.data)) {
            return;
        }
        const targetIndices = [];
        graphDiv.data.forEach((trace, idx) => {
            if (isMeasurementDistributionTrace(trace)) {
                targetIndices.push(idx);
            }
        });
        if (targetIndices.length === 0) {
            return;
        }
        const visibleValue = measurementDistributionVisible ? true : 'legendonly';
        frameWindow.Plotly.restyle(graphDiv, { visible: visibleValue }, targetIndices);
    }

    function bindScanIframeLegendPreferenceSync() {
        if (!scanIframe) {
            return;
        }
        const frameDocument = scanIframe.contentDocument;
        if (!frameDocument) {
            return;
        }
        const graphDiv = frameDocument.querySelector('.plotly-graph-div');
        if (!graphDiv || graphDiv.dataset.measureDistListenerAttached === '1') {
            return;
        }
        graphDiv.dataset.measureDistListenerAttached = '1';
        graphDiv.on('plotly_restyle', (restyleData) => {
            if (!Array.isArray(restyleData) || restyleData.length < 2) {
                return;
            }
            const updates = restyleData[0] || {};
            const traceIndices = Array.isArray(restyleData[1]) ? restyleData[1] : [];
            if (!('visible' in updates) || traceIndices.length === 0 || !Array.isArray(graphDiv.data)) {
                return;
            }
            const visibleUpdate = updates.visible;
            for (const traceIdx of traceIndices) {
                const trace = graphDiv.data[traceIdx];
                if (!isMeasurementDistributionTrace(trace)) {
                    continue;
                }
                const nextState = resolveTraceVisibleState(visibleUpdate);
                if (nextState !== null) {
                    measurementDistributionVisible = nextState;
                }
                break;
            }
        });
    }

    function formatMetricValue(value) {
        if (typeof value === 'number' && Number.isFinite(value)) {
            return value.toPrecision(3);
        }
        return 'N/A';
    }

    function escapeHtml(text) {
        if (typeof text !== 'string') return '';
        return text.replace(/[&<>"']/g, function (m) {
            return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]);
        });
    }

    function formatFrequency(value) {
        if (typeof value === 'number' && Number.isFinite(value)) {
            const absVal = Math.abs(value);
            if (absVal >= 1e9) {
                return (value / 1e9).toFixed(2) + ' GHz';
            } else if (absVal >= 1e6) {
                return (value / 1e6).toFixed(2) + ' MHz';
            } else if (absVal >= 1e3) {
                return (value / 1e3).toFixed(2) + ' kHz';
            } else {
                if (absVal >= 0.01 || absVal === 0) {
                    return value.toFixed(2) + ' Hz';
                } else {
                    return value.toPrecision(3) + ' Hz';
                }
            }
        }
        return 'N/A';
    }

    function formatCount(value) {
        if (typeof value === 'number' && Number.isFinite(value)) {
            return Math.round(value).toString();
        }
        return 'N/A';
    }

    function formatDuration(value) {
        if (typeof value === 'number' && Number.isFinite(value)) {
            const totalMs = value;
            const ms = Math.floor(totalMs % 1000);
            const totalSecs = Math.floor(totalMs / 1000);
            const secs = totalSecs % 60;
            const totalMins = Math.floor(totalSecs / 60);
            const mins = totalMins % 60;
            const hours = Math.floor(totalMins / 60);

            const msStr = ms.toString().padStart(3, '0');
            const secsStr = secs.toString().padStart(2, '0');
            const minsStr = mins.toString().padStart(2, '0');
            const hoursStr = hours.toString().padStart(2, '0');

            if (hours > 0) {
                return hoursStr + ':' + minsStr + ':' + secsStr + '.' + msStr;
            } else {
                return minsStr + ':' + secsStr + '.' + msStr;
            }
        }
        return 'N/A';
    }

    function formatTimestamp(isoString) {
        if (!isoString || typeof isoString !== 'string') {
            return 'N/A';
        }
        try {
            const d = new Date(isoString);
            if (Number.isNaN(d.getTime())) {
                return isoString;
            }
            return d.toLocaleString();
        } catch (e) {
            return isoString;
        }
    }

    function renderSweepMetricsPanel(container, metrics) {
        if (!container) return;
        container.innerHTML = '';
        if (metrics.sobol_baseline_steps != null && metrics.sobol_freq_steps != null && metrics.sobol_conv_diff == null) {
            metrics.sobol_conv_diff = metrics.sobol_baseline_steps - metrics.sobol_freq_steps;
        }
        // Build a displayable focus_window string from acquisition bounds,
        // but only for sweep locators that support focus (indicated by
        // the presence of expected_focused_points).
        if (metrics.expected_focused_points != null &&
            metrics.acquisition_lo != null && metrics.acquisition_hi != null) {
            const lo = Number(metrics.acquisition_lo);
            const hi = Number(metrics.acquisition_hi);
            if (Number.isFinite(lo) && Number.isFinite(hi)) {
                metrics.focus_window = '[' + lo.toExponential(3) + ', ' + hi.toExponential(3) + ']';
            }
        }
        const items = [
            { key: 'measurements_done', label: 'Measurements done', tip: 'Actual measurements taken before stopping or hitting the step limit.', fmt: formatCount },
            { key: 'dips_detected', label: 'Dips detected', tip: 'Dips found in the initial sweep after noise filtering. When the sweep is too sparse to detect dips, falls back to the true ground-truth dip count.', fmt: formatCount },
            { key: 'dips_merged', label: 'Dips merged', tip: 'Whether detected dips are close enough to be treated as one combined range.', fmt: function (v) { return v ? 'Yes' : 'No'; } },
            { key: 'min_dip_width', label: 'Dip width', tip: 'Width of the actual signal dip in physical frequency units.', fmt: formatFrequency },
            { key: 'total_signal_span', label: 'Signal span', tip: 'Total span from first dip start to last dip end in physical frequency units.', fmt: formatFrequency },
            { key: 'sweep_efficiency', label: 'Efficiency', tip: 'Expected uniform points / actual measurements. >1 means the locator was efficient.', fmt: formatMetricValue },
            { key: 'focus_window', label: 'Focus window', tip: 'Inferred frequency window the locator narrowed onto after detecting dips.', fmt: function (v) { return v; } },
        ];
        let any = false;
        for (const it of items) {
            const val = metrics[it.key];
            if (val == null || (typeof val === 'number' && !Number.isFinite(val))) continue;
            any = true;
            const el = document.createElement('div');
            el.className = 'metric-item';
            if (it.key === 'sweep_efficiency' && val != null) {
                if (val >= 1) {
                    el.classList.add('efficiency-good');
                } else if (val >= 0.5) {
                    el.classList.add('efficiency-medium');
                } else {
                    el.classList.add('efficiency-bad');
                }
            }
            let formula = '';
            if (it.key === 'sweep_efficiency' && metrics.expected_uniform_points != null && metrics.measurements_done != null) {
                formula = '<div class="metric-formula">' + formatCount(metrics.expected_uniform_points) + ' expected / ' + formatCount(metrics.measurements_done) + ' actual = ' + it.fmt(val) + '×</div>';
            }
            el.innerHTML =
                '<div class="metric-header">' +
                '<span class="metric-label">' + it.label + '</span>' +
                '<span class="help-icon" tabindex="0" title="' + it.tip.replace(/"/g, '&quot;') + '">?</span>' +
                '</div>' +
                '<div class="metric-value">' + it.fmt(val) + '</div>' + formula;
            container.appendChild(el);
        }
        container.hidden = !any;
    }

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

    // Parse plot data from scan HTML file on-demand (avoids bloating manifest)
    async function loadPlotDataFromScanHtml(plot) {
        if (!plot || !plot.path) return null;
        try {
            const url = resolveAssetPath(plot.path);
            const response = await fetch(url, { cache: 'no-store' });
            if (!response.ok) return null;
            const html = await response.text();
            return parsePlotDataFromHtml(html);
        } catch (e) {
            console.warn('Failed to load plot data from', plot.path, e);
            return null;
        }
    }

    function parsePlotDataFromHtml(html) {
        // Extract Plotly.newPlot data arrays + layout from HTML
        const m = html.match(/Plotly\.newPlot\(\s*"[^"]+",\s*/);
        if (!m) return null;
        const pos = m.index + m[0].length;
        try {
            const dataStr = html.slice(pos);
            // Find the end of the data array (first top-level array close)
            let depth = 0;
            let end = 0;
            for (let i = 0; i < dataStr.length; i++) {
                if (dataStr[i] === '[') depth++;
                else if (dataStr[i] === ']') {
                    depth--;
                    if (depth === 0) { end = i + 1; break; }
                } else if (dataStr[i] === '{' && depth === 0) {
                    // Started object before array - malformed
                    break;
                }
            }
            const data = JSON.parse(dataStr.slice(0, end));
            if (!Array.isArray(data)) return null;
            const out = extractPlotDataFromTraces(data);
            if (!out) return null;

            // Try to parse layout for narrowed_param_bounds in meta
            try {
                let layoutStart = pos + end;
                while (layoutStart < html.length && /[\s,]/.test(html[layoutStart])) layoutStart++;
                if (html[layoutStart] === '{') {
                    // find matching closing brace
                    let ldepth = 0, lend = 0;
                    for (let i = layoutStart; i < html.length; i++) {
                        if (html[i] === '{') ldepth++;
                        else if (html[i] === '}') {
                            ldepth--;
                            if (ldepth === 0) { lend = i + 1; break; }
                        }
                    }
                    if (lend > layoutStart) {
                        const layout = JSON.parse(html.slice(layoutStart, lend));
                        const meta = layout && layout.meta;
                        if (meta && meta.narrowed_param_bounds && typeof meta.narrowed_param_bounds === 'object') {
                            out.narrowed_param_bounds = meta.narrowed_param_bounds;
                        }
                    }
                }
            } catch (layoutErr) {
                // Non-critical: layout parse failures are silently ignored
            }
            return out;
        } catch (e) {
            console.warn('Failed to parse plot data from HTML:', e);
            return null;
        }
    }


    function extractPlotDataFromTraces(traces) {
        let x_dense = null, y_dense = null, y_dense_noisy = null, y_dense_mode = null;
        let coarse_x = [], coarse_y = [], secondary_x = [], secondary_y = [], tertiary_x = [], tertiary_y = [], fine_x = [], fine_y = [], fine_step = [];
        let step_x = [], step_y = [], step_idx = [];
        let has_metrics = false, focus_window = null;

        for (const tr of traces) {
            const name = tr.name || '';
            const mode = tr.mode || '';
            if ((name === 'locator most likely signal' || name === 'locator mode belief signal') && mode.includes('lines')) {
                y_dense_mode = tr.y;
            } else if (name === 'true signal' && mode.includes('lines')) {
                x_dense = tr.x;
                y_dense = tr.y;
            } else if (name === 'simulated noisy signal (over-frequency)' && mode.includes('lines')) {
                y_dense_noisy = tr.y?.map((v, i) => v != null ? v : (y_dense?.[i] || 0));
            } else if (name === 'measurements (coarse)') {
                coarse_x = tr.x || [];
                coarse_y = tr.y || [];
            } else if (name === 'measurements (secondary)') {
                secondary_x = tr.x || [];
                secondary_y = tr.y || [];
            } else if (name === 'measurements (tertiary)') {
                tertiary_x = tr.x || [];
                tertiary_y = tr.y || [];
            } else if (name === 'measurements (inference)') {
                fine_x = tr.x || [];
                fine_y = tr.y || [];
                fine_step = tr.marker?.color || fine_x.map((_, i) => i);
            } else if (name === 'measurements (noisy)') {
                step_x = tr.x || [];
                step_y = tr.y || [];
                step_idx = tr.marker?.color || step_x.map((_, i) => i);
            } else if (name === 'Entropy' || name === 'Uncertainty') {
                has_metrics = true;
            }
        }

        if (!x_dense || !y_dense) return null;

        const out = { x_dense, y_dense, has_metrics };
        if (y_dense_mode && y_dense_mode.length === x_dense.length) out.y_dense_mode = y_dense_mode;
        if (y_dense_noisy && y_dense_noisy.length === x_dense.length) out.y_dense_noisy = y_dense_noisy;

        if (coarse_x.length || secondary_x.length || tertiary_x.length || fine_x.length) {
            out.measurements = {
                mode: 'phases',
                coarse_x, coarse_y: coarse_y.map(y => y == null ? null : Number(y)),
                secondary_x, secondary_y: secondary_y.map(y => y == null ? null : Number(y)),
                tertiary_x, tertiary_y: tertiary_y.map(y => y == null ? null : Number(y)),
                fine_x, fine_y: fine_y.map(y => y == null ? null : Number(y)),
                fine_step: fine_step.map(s => Number(s))
            };
        } else if (step_x.length) {
            out.measurements = {
                mode: 'steps',
                x: step_x,
                y: step_y.map(y => y == null ? null : Number(y)),
                step: step_idx.map(s => Number(s))
            };
        } else {
            out.measurements = { mode: 'empty' };
        }

        return out;
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

    function addHeadToHeadMeasurementTraces(traces, m, label, side) {
        const cs = side === 'left' ? 'Oranges' : 'Purples';
        const symbol = side === 'left' ? 'circle' : 'diamond';
        if (!m || m.mode === 'empty') {
            return;
        }
        if (m.mode === 'phases') {
            const coarseColor = 'rgba(176,176,176,0.9)';
            const secondaryColor = 'rgba(255,127,14,0.9)';
            const tertiaryColor = 'rgba(148,0,211,0.9)';
            if (m.coarse_x && m.coarse_x.length) {
                traces.push({
                    type: 'scatter',
                    x: m.coarse_x,
                    y: m.coarse_y,
                    mode: 'markers',
                    name: `${label} (coarse)`,
                    marker: {
                        size: 7,
                        color: coarseColor,
                        symbol: symbol,
                        line: { width: 0.6, color: '#4a4a4a' },
                    },
                    hovertemplate: 'x=%{x}<br>y=%{y:.4f}<br>phase=initial sweep<extra></extra>',
                });
            }
            if (m.secondary_x && m.secondary_x.length) {
                traces.push({
                    type: 'scatter',
                    x: m.secondary_x,
                    y: m.secondary_y,
                    mode: 'markers',
                    name: `${label} (secondary)`,
                    marker: {
                        size: 7,
                        color: secondaryColor,
                        symbol: symbol,
                        line: { width: 0.6, color: '#8B4513' },
                    },
                    hovertemplate: 'x=%{x}<br>y=%{y:.4f}<br>phase=secondary sweep<extra></extra>',
                });
            }
            if (m.tertiary_x && m.tertiary_x.length) {
                traces.push({
                    type: 'scatter',
                    x: m.tertiary_x,
                    y: m.tertiary_y,
                    mode: 'markers',
                    name: `${label} (tertiary)`,
                    marker: {
                        size: 7,
                        color: tertiaryColor,
                        symbol: symbol,
                        line: { width: 0.6, color: '#4B0082' },
                    },
                    hovertemplate: 'x=%{x}<br>y=%{y:.4f}<br>phase=tertiary sweep<extra></extra>',
                });
            }
            if (m.fine_x && m.fine_x.length) {
                const fineSteps =
                    Array.isArray(m.fine_step) && m.fine_step.length === m.fine_x.length
                        ? m.fine_step
                        : m.fine_x.map((_, i) => i);
                const maxStep = Math.max(1, fineSteps.length - 1);
                const finePct = fineSteps.map((s) => (s / maxStep) * 100.0);
                traces.push({
                    type: 'scatter',
                    x: m.fine_x,
                    y: m.fine_y,
                    mode: 'markers',
                    name: `${label} (fine)`,
                    marker: {
                        size: 8,
                        color: fineSteps,
                        colorscale: cs,
                        showscale: false,
                        symbol: symbol,
                        line: { width: 0.5, color: '#222' },
                    },
                    customdata: finePct,
                    hovertemplate:
                        'x=%{x}<br>y=%{y:.4f}<br>inference step=%{marker.color}<br>inference progress=%{customdata:.1f}%<extra></extra>',
                });
            }
            return;
        }
        if (m.mode === 'steps') {
            traces.push({
                type: 'scatter',
                x: m.x,
                y: m.y,
                mode: 'markers',
                name: `${label} (measurements)`,
                marker: {
                    size: 8,
                    color: m.step,
                    colorscale: cs,
                    symbol: symbol,
                    line: { width: 0.5, color: '#222' },
                    showscale: false,
                },
            });
        }
    }

    function buildHeadToHeadFocusShapes(pdL, pdR) {
        const shapes = [];
        if (pdL.focus_window && pdL.focus_window.length === 2) {
            const x0 = pdL.focus_window[0];
            const x1 = pdL.focus_window[1];
            if (Number.isFinite(x0) && Number.isFinite(x1) && x1 > x0) {
                shapes.push({
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0,
                    x1,
                    y0: 0,
                    y1: 1,
                    fillcolor: 'rgba(46, 204, 113, 0.12)',
                    line: { color: 'rgba(46, 204, 113, 0.45)', width: 1 },
                    layer: 'below',
                });
            }
        }
        if (pdR.focus_window && pdR.focus_window.length === 2) {
            const x0 = pdR.focus_window[0];
            const x1 = pdR.focus_window[1];
            if (Number.isFinite(x0) && Number.isFinite(x1) && x1 > x0) {
                shapes.push({
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0,
                    x1,
                    y0: 0,
                    y1: 1,
                    fillcolor: 'rgba(59, 130, 246, 0.1)',
                    line: { color: 'rgba(59, 130, 246, 0.45)', width: 1, dash: 'dot' },
                    layer: 'below',
                });
            }
        }
        return shapes;
    }

    function buildHeadToHeadTraces(pdL, pdR, nameL, nameR) {
        const traces = [];
        traces.push({
            type: 'scatter',
            x: pdL.x_dense,
            y: pdL.y_dense,
            mode: 'lines',
            name: 'true signal',
            line: { color: '#2563eb', width: 2 },
        });
        if (
            pdL.y_dense_noisy &&
            pdL.y_dense_noisy.length &&
            pdL.y_dense_noisy.length === pdL.x_dense.length
        ) {
            traces.push({
                type: 'scatter',
                x: pdL.x_dense,
                y: pdL.y_dense_noisy,
                mode: 'lines',
                name: 'simulated noisy signal (over-frequency)',
                line: { color: '#fb923c', dash: 'dot', width: 1.5 },
            });
        }
        if (
            pdL.y_dense_mode &&
            pdL.y_dense_mode.length &&
            pdL.y_dense_mode.length === pdL.x_dense.length
        ) {
            traces.push({
                type: 'scatter',
                x: pdL.x_dense,
                y: pdL.y_dense_mode,
                mode: 'lines',
                name: nameL + ' (most likely)',
                line: { color: '#dc2626', dash: 'dash', width: 2 },
            });
        }
        if (
            pdR.y_dense_mode &&
            pdR.y_dense_mode.length &&
            pdR.y_dense_mode.length === pdR.x_dense.length
        ) {
            traces.push({
                type: 'scatter',
                x: pdR.x_dense,
                y: pdR.y_dense_mode,
                mode: 'lines',
                name: nameR + ' (most likely)',
                line: { color: '#9333ea', dash: 'dash', width: 2 },
            });
        }
        addHeadToHeadMeasurementTraces(traces, pdL.measurements, nameL, 'left');
        addHeadToHeadMeasurementTraces(traces, pdR.measurements, nameR, 'right');
        return traces;
    }

    function controlValue(control) {
        if (control instanceof HTMLSelectElement) {
            return control.value || control.dataset.value || '';
        }
        return control.dataset.value ?? '';
    }

    function setControlValue(control, value, { silent = false } = {}) {
        const normalized = value ?? '';
        control.dataset.value = normalized;
        const buttons = control.querySelectorAll('button');
        for (const button of buttons) {
            const isActive = button.dataset.value === normalized;
            button.classList.toggle('is-active', isActive);
            button.setAttribute('aria-checked', String(isActive));
            button.tabIndex = isActive ? 0 : -1;
        }
        if (!silent) {
            control.dispatchEvent(
                new CustomEvent('controlchange', {
                    bubbles: false,
                    detail: { value: normalized },
                }),
            );
        }
    }

    function renderSegmentedControl(control, items, previousValue, options = {}) {
        const disabledItems = options.disabledItems instanceof Set
            ? options.disabledItems
            : new Set(options.disabledItems || []);
        const uniqueItems = [
            ...new Set(
                items.filter((item) => item !== undefined && item !== null)
            ),
        ]
            .map((item) => String(item))
            .sort((a, b) => a.localeCompare(b));

        control.innerHTML = '';

        for (const item of uniqueItems) {
            const button = document.createElement('button');
            button.type = 'button';
            button.dataset.value = item;
            button.setAttribute('role', 'radio');
            button.setAttribute('aria-checked', 'false');
            button.tabIndex = -1;
            button.textContent = item;
            if (disabledItems.has(item)) {
                button.disabled = true;
                button.title = 'No scan data yet. Run this strategy with nvision run, then render.';
            }
            button.addEventListener('click', () => {
                if (button.disabled) {
                    return;
                }
                setControlValue(control, item);
            });
            button.addEventListener('keydown', (e) => {
                const buttons = Array.from(control.querySelectorAll('button'));
                const index = buttons.indexOf(button);
                let nextIndex = null;
                if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                    nextIndex = (index + 1) % buttons.length;
                } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                    nextIndex = (index - 1 + buttons.length) % buttons.length;
                }
                if (nextIndex !== null) {
                    e.preventDefault();
                    buttons[nextIndex].focus();
                    buttons[nextIndex].click();
                }
            });
            control.appendChild(button);
        }

        let nextValue = '';
        const enabledItems = uniqueItems.filter((item) => !disabledItems.has(item));
        if (enabledItems.length > 0) {
            if (previousValue && enabledItems.includes(previousValue)) {
                nextValue = previousValue;
            } else if (previousValue && uniqueItems.includes(previousValue)) {
                nextValue = enabledItems[0];
            } else {
                nextValue = enabledItems[0];
            }
        }

        setControlValue(control, nextValue, { silent: true });
        return nextValue;
    }

    function renderSelectControl(select, items, previousValue) {
        const uniqueItems = [
            ...new Set(items.filter((item) => item !== undefined && item !== null)),
        ]
            .map((item) => String(item))
            .sort((a, b) => Number(a) - Number(b));

        let nextValue = '';
        if (uniqueItems.length > 0) {
            if (previousValue && uniqueItems.includes(previousValue)) {
                nextValue = previousValue;
            } else {
                nextValue = uniqueItems[0];
            }
        }

        select.innerHTML = '';
        for (const item of uniqueItems) {
            const option = document.createElement('option');
            option.value = item;
            option.textContent = item;
            select.appendChild(option);
        }

        if (nextValue) {
            select.value = nextValue;
        } else {
            select.value = '';
        }
        select.dataset.value = nextValue;
        select.dataset.options = JSON.stringify(uniqueItems);
        return { value: nextValue, items: uniqueItems };
    }

    function updateRepeatNavButtons() {
        if (!scanRepeatPrev || !scanRepeatNext) {
            return;
        }
        if (currentRepeatItems.length === 0) {
            scanRepeatPrev.disabled = true;
            scanRepeatNext.disabled = true;
            return;
        }
        const currentValue = scanRepeat.value || currentRepeatItems[0] || '';
        const currentIndex = currentRepeatItems.indexOf(currentValue);
        const hasValidSelection = currentIndex !== -1;
        scanRepeatPrev.disabled = !hasValidSelection || currentIndex <= 0;
        scanRepeatNext.disabled = !hasValidSelection || currentIndex >= currentRepeatItems.length - 1;
    }

    function selectRepeatByIndex(index) {
        if (currentRepeatItems.length === 0) {
            return;
        }
        const clampedIndex = Math.max(0, Math.min(index, currentRepeatItems.length - 1));
        const nextValue = currentRepeatItems[clampedIndex];
        if (nextValue === undefined) {
            return;
        }
        scanRepeat.value = nextValue;
        scanRepeat.dataset.value = nextValue;
        updateRepeatNavButtons();
        findAndDisplayPlot();
    }

    function selectRepeatByOffset(offset) {
        if (currentRepeatItems.length === 0) {
            return;
        }
        const currentValue = scanRepeat.value || currentRepeatItems[0];
        let currentIndex = currentRepeatItems.indexOf(currentValue);
        if (currentIndex === -1) {
            currentIndex = 0;
        }
        const targetIndex = currentIndex + offset;
        if (targetIndex < 0 || targetIndex >= currentRepeatItems.length) {
            return;
        }
        selectRepeatByIndex(targetIndex);
    }

    // Gaussian noise standard deviation slider state and selectors
    const gaussStdSliderRow = document.getElementById('gaussian-noise-slider-row');
    const gaussStdSlider = document.getElementById('gauss-std-slider');
    const gaussStdValue = document.getElementById('gauss-std-value');
    const gaussStdPrev = document.getElementById('gauss-std-prev');
    const gaussStdNext = document.getElementById('gauss-std-next');
    let currentGaussSigmas = [];

    function getEffectiveScanNoise() {
        const selectedScanNoise = controlValue(scanNoise);
        if (selectedScanNoise === 'Gauss' && currentGaussSigmas.length > 0) {
            const idx = parseInt(gaussStdSlider.value, 10);
            const sigma = currentGaussSigmas[idx];
            let sigmaStr = sigma.toString();
            if (!sigmaStr.includes('.')) {
                sigmaStr += '.0';
            }
            return `Gauss(${sigmaStr})`;
        }
        return selectedScanNoise;
    }

    function updateGaussStdSlider() {
        const selectedScanGenerator = controlValue(scanGenerator);
        const selectedScanNoise = controlValue(scanNoise);
        
        if (selectedScanNoise === 'Gauss') {
            const rawNoises = [...new Set(
                scanPlots
                    .filter((p) => p.generator === selectedScanGenerator && p.noise.includes('Gauss'))
                    .map((p) => p.noise)
            )];
            
            const sigmas = rawNoises
                .map(n => {
                    const match = n.match(/Gauss\(([\d.]+)\)/);
                    return match ? parseFloat(match[1]) : null;
                })
                .filter(v => v !== null)
                .sort((a, b) => a - b);
            
            currentGaussSigmas = sigmas;
            
            if (sigmas.length > 0) {
                gaussStdSliderRow.style.display = 'flex';
                gaussStdSlider.min = 0;
                gaussStdSlider.max = sigmas.length - 1;
                gaussStdSlider.step = 1;
                
                let savedIdx = parseInt(gaussStdSlider.dataset.index, 10);
                if (isNaN(savedIdx) || savedIdx < 0 || savedIdx >= sigmas.length) {
                    if (scanDefault && scanDefault.noise && scanDefault.noise.includes('Gauss') && scanDefault.generator === selectedScanGenerator) {
                        const match = scanDefault.noise.match(/Gauss\(([\d.]+)\)/);
                        if (match) {
                            const targetSigma = parseFloat(match[1]);
                            const closestIdx = sigmas.findIndex(s => Math.abs(s - targetSigma) < 1e-6);
                            if (closestIdx !== -1) {
                                savedIdx = closestIdx;
                            }
                        }
                    }
                }
                if (isNaN(savedIdx) || savedIdx < 0 || savedIdx >= sigmas.length) {
                    savedIdx = 0;
                }
                
                gaussStdSlider.value = savedIdx;
                gaussStdSlider.dataset.index = savedIdx;
                gaussStdValue.textContent = sigmas[savedIdx].toFixed(4).replace(/\.?0+$/, '');
                
                gaussStdPrev.disabled = savedIdx === 0;
                gaussStdNext.disabled = savedIdx === sigmas.length - 1;
            } else {
                gaussStdSliderRow.style.display = 'none';
            }
        } else {
            gaussStdSliderRow.style.display = 'none';
        }
    }

    /** Left column: generator, noise (signal / experiment path). */
    function updateScanSignalControls() {
        const scanGeneratorItems = [...new Set(scanPlots.map((p) => p.generator))].sort();
        const selectedScanGenerator = renderSegmentedControl(
            scanGenerator,
            scanGeneratorItems,
            controlValue(scanGenerator),
        );

        const rawNoiseItems = [...new Set(
            scanPlots
                .filter((p) => p.generator === selectedScanGenerator)
                .map((p) => p.noise)
        )];
        
        const hasGauss = rawNoiseItems.some(n => n.includes("Gauss"));
        const nonGaussItems = rawNoiseItems.filter(n => !n.includes("Gauss"));
        const scanNoiseItems = hasGauss ? ["Gauss", ...nonGaussItems] : nonGaussItems;

        renderSegmentedControl(
            scanNoise,
            scanNoiseItems,
            controlValue(scanNoise),
        );

        updateGaussStdSlider();
    }

    function updateScanStrategyControl() {
        const selectedScanGenerator = controlValue(scanGenerator);
        const selectedScanNoise = getEffectiveScanNoise();
        const availableFromPlots = new Set(
            scanPlots
                .filter((p) => p.generator === selectedScanGenerator && p.noise === selectedScanNoise)
                .map((p) => p.strategy),
        );
        const gridStrategies =
            (window.STRATEGY_GRID && window.STRATEGY_GRID[selectedScanGenerator]) || [];
        const scanStrategyItems = [
            ...new Set([...gridStrategies, ...availableFromPlots]),
        ];
        const disabledItems = new Set(
            scanStrategyItems.filter((strategy) => !availableFromPlots.has(strategy)),
        );
        renderSegmentedControl(scanStrategy, scanStrategyItems, controlValue(scanStrategy), {
            disabledItems,
        });
    }

    function updateScanRepeatControl() {
        const selectedScanGenerator = controlValue(scanGenerator);
        const selectedScanNoise = getEffectiveScanNoise();
        const selectedScanStrategy = controlValue(scanStrategy);

        const scanRepeatItems = scanPlots
            .filter(
                (p) =>
                    p.generator === selectedScanGenerator &&
                    p.noise === selectedScanNoise &&
                    p.strategy === selectedScanStrategy
            )
            .map((p) => String(p.repeat ?? p.attempt ?? 1));
        const { value: selectedRepeat, items: repeatItems } = renderSelectControl(
            scanRepeat,
            scanRepeatItems,
            controlValue(scanRepeat) || scanRepeat.dataset.value || '',
        );
        currentRepeatItems = repeatItems;
        if (selectedRepeat) {
            scanRepeat.dataset.value = selectedRepeat;
        }
        updateRepeatNavButtons();

        const viewToggleContainer = document.getElementById('scan-view-toggle-container');
        if (viewToggleContainer) {
            if (repeatItems.length > 0) {
                viewToggleContainer.style.display = 'flex';
            } else {
                viewToggleContainer.style.display = 'none';
                const singleBtn = document.querySelector('#scan-view-mode button[data-value="single"]');
                if (singleBtn) singleBtn.click();
            }
        }
    }

    function updateAllScanControls() {
        updateScanSignalControls();
        updateScanStrategyControl();
        updateScanRepeatControl();
    }

    function findAndDisplayPlot() {
        const scanGeneratorValue = controlValue(scanGenerator);
        const scanNoiseValue = getEffectiveScanNoise();
        const scanStrategyValue = controlValue(scanStrategy);
        const scanRepeatValue = controlValue(scanRepeat);

        if (
            scanGeneratorValue &&
            scanNoiseValue &&
            scanStrategyValue &&
            scanRepeatValue
        ) {
            const repeatNumber = parseInt(scanRepeatValue, 10);
            const plot = scanPlots.find(
                (p) =>
                    p.generator === scanGeneratorValue &&
                    p.noise === scanNoiseValue &&
                    p.strategy === scanStrategyValue &&
                    p.repeat === repeatNumber
            );
            currentPlot = plot;
            scanIframe.src = plot ? plot.path : '';
            if (plot) {
                function buildScanItems(phaseData, isOverall, totalMeasurements) {
                    const repeatTotal = plot.repeat_total ?? null;
                    const attemptLabel = repeatTotal
                        ? 'Attempt ' + plot.repeat + ' of ' + repeatTotal
                        : 'Attempt ' + plot.repeat;
                    // For sweep-only runs, phaseData.measurements is the authoritative total.
                    const phaseMeasurements = phaseData.measurements != null ? phaseData.measurements : totalMeasurements;
                    const items = [
                        { label: 'Attempt', val: attemptLabel, tip: 'Which repeat attempt this scan corresponds to.' },
                        { label: 'Measurements', val: formatCount(phaseMeasurements), tip: 'Total number of measurements (sweep + acquisition) taken in this repeat.' },
                    ];
                    if (phaseData.steps_to_fb != null) {
                        items.push({ label: 'Freq. converged', val: formatCount(phaseData.steps_to_fb), tip: 'Measurements taken until center frequency (fb) converged below threshold.' });
                    }
                    // Show total sweep steps (if any)
                    const sweepSteps = phaseData.sweep_steps;
                    if (sweepSteps != null && sweepSteps > 0) {
                        const sweepStr = phaseMeasurements != null && phaseMeasurements > 0
                            ? sweepSteps + '/' + phaseMeasurements
                            : String(sweepSteps);
                        items.push({ label: 'Sweep steps', val: sweepStr, tip: 'Measurements spent in the initial coarse/focused sweep phase.' });
                    }
                    if (phaseData.duration_ms != null) {
                        items.push({ label: 'Duration', val: formatDuration(phaseData.duration_ms), tip: 'Wall-clock time for this repeat.' });
                    }
                    if (phaseData.last_run != null) {
                        items.push({ label: 'Last run', val: formatTimestamp(phaseData.last_run), tip: 'Timestamp when this repeat was executed.' });
                    }
                    if (phaseData.abs_err_x != null) {
                        items.push({ label: 'Abs error', val: formatFrequency(phaseData.abs_err_x), tip: 'Absolute frequency error vs ground truth. Lower is better.' });
                    }
                    if (phaseData.uncert != null) {
                        items.push({ label: 'Uncertainty', val: formatFrequency(phaseData.uncert), tip: 'Final estimated standard deviation of the frequency estimate. Lower is better.' });
                    }

                    // Milestone metrics
                    if (phaseData.err_fb_at_milestone != null) {
                        items.push({ label: 'fb Err @ Milestone', val: formatFrequency(phaseData.err_fb_at_milestone), tip: 'Absolute error of center frequency at the moment of convergence.' });
                    }
                    if (phaseData.err_fc_at_milestone != null) {
                        items.push({ label: 'fc Err @ Milestone', val: formatFrequency(phaseData.err_fc_at_milestone), tip: 'Absolute error of splitting at the moment fb converged.' });
                    }
                    if (phaseData.err_fc_diff != null) {
                        items.push({ label: 'fc Err Gain', val: formatFrequency(phaseData.err_fc_diff), tip: 'Reduction in splitting error achieved by continuing after fb convergence.' });
                    }
                    return items;
                }



                function buildFreqConvergenceRows(phaseData) {
                    if (!phaseData) return [];
                    const metrics = phaseData.metrics || {};
                    const sobolFreqSteps = phaseData.sobol_freq_steps != null ? phaseData.sobol_freq_steps : metrics.sobol_freq_steps;
                    const stepsToFb = phaseData.steps_to_fb != null ? phaseData.steps_to_fb : metrics.steps_to_fb;
                    const uncertFb = phaseData.uncert_fb_at_milestone != null ? phaseData.uncert_fb_at_milestone : metrics.uncert_fb_at_milestone;
                    const errFb = phaseData.err_fb_at_milestone != null ? phaseData.err_fb_at_milestone : metrics.err_fb_at_milestone;
                    const sobolFreqUncert = phaseData.sobol_freq_uncert_at_conv != null ? phaseData.sobol_freq_uncert_at_conv : metrics.sobol_freq_uncert_at_conv;
                    const sobolFreqErr = phaseData.sobol_freq_err_at_conv != null ? phaseData.sobol_freq_err_at_conv : metrics.sobol_freq_err_at_conv;
                    const uncert = phaseData.uncert != null ? phaseData.uncert : metrics.uncert;
                    const absErr = phaseData.abs_err_x != null ? phaseData.abs_err_x : metrics.abs_err_x;

                    const freqStepsExpected = (sobolFreqSteps != null && stepsToFb != null && stepsToFb < sobolFreqSteps);
                    const sbedFreqErrExpected = (errFb != null && uncertFb != null && errFb < uncertFb);
                    const sobolFreqErrExpected = (sobolFreqErr != null && sobolFreqUncert != null && sobolFreqErr < sobolFreqUncert);
                    const uncertFbDiffExpected = (uncertFb != null && uncert != null && uncertFb - uncert > 0);
                    const errFbDiffExpected = (errFb != null && absErr != null && errFb - absErr > 0);

                    return [
                        // Row 1: Steps — sobol - sbed
                        [
                            { label: 'Sobol freq convergence', val: sobolFreqSteps != null ? formatCount(sobolFreqSteps) : 'N/A', tip: 'Steps needed for simple Sobol frequency uncertainty to drop below threshold.', cardClass: freqStepsExpected ? 'expected-card' : '' },
                            { label: 'Sbed freq convergence', val: stepsToFb != null ? formatCount(stepsToFb) : 'N/A', tip: 'Steps needed for Sbed frequency uncertainty to drop below threshold.', cardClass: freqStepsExpected ? 'expected-card' : '' },
                            { label: 'Freq convergence savings', val: (sobolFreqSteps != null && stepsToFb != null) ? formatCount(sobolFreqSteps - stepsToFb) : 'N/A', tip: 'Difference in steps needed for frequency convergence (positive = Sbed was faster).', cardClass: freqStepsExpected ? 'expected-card' : '' }
                        ],
                        // Row 2: Uncertainty — frequency - overall
                        [
                            { label: 'Sobol freq uncertainty', val: sobolFreqUncert != null ? formatFrequency(sobolFreqUncert) : 'N/A', tip: 'Uncertainty (standard deviation) of Sobol frequency estimate at the moment of convergence.', cardClass: sobolFreqErrExpected ? 'expected-card' : '' },
                            { label: 'Sbed freq uncertainty', val: uncertFb != null ? formatFrequency(uncertFb) : 'N/A', tip: 'Uncertainty (standard deviation) of Sbed frequency estimate at the moment of convergence.', cardClass: sbedFreqErrExpected ? 'expected-card' : '' },
                            { label: 'Freq uncert difference', val: (uncertFb != null && uncert != null) ? formatFrequency(uncertFb - uncert) : 'N/A', tip: 'Reduction in Sbed frequency uncertainty from freq-convergence milestone to final (positive = uncertainty decreased).', cardClass: uncertFbDiffExpected ? 'expected-card' : '' }
                        ],
                        // Row 3: Absolute Error — frequency - overall
                        [
                            { label: 'Sobol freq error', val: sobolFreqErr != null ? formatFrequency(sobolFreqErr) : 'N/A', tip: 'Absolute error of Sobol frequency estimate vs ground truth at the moment of convergence.', cardClass: sobolFreqErrExpected ? 'expected-card' : '' },
                            { label: 'Sbed freq error', val: errFb != null ? formatFrequency(errFb) : 'N/A', tip: 'Absolute error of Sbed frequency estimate vs ground truth at the moment of convergence.', cardClass: sbedFreqErrExpected ? 'expected-card' : '' },
                            { label: 'Freq error difference', val: (errFb != null && absErr != null) ? formatFrequency(errFb - absErr) : 'N/A', tip: 'Change in Sbed absolute frequency error from freq-convergence milestone to final (positive = error decreased, negative = error increased).', cardClass: errFbDiffExpected ? 'expected-card' : '' }
                        ]
                    ];
                }

                function buildComparisonRows(phaseData, plotContext) {
                    if (!phaseData) return [];
                    const metrics = phaseData.metrics || {};
                    const sobolBaseline = phaseData.sobol_baseline_steps != null ? phaseData.sobol_baseline_steps : metrics.sobol_baseline_steps;
                    const measurements = phaseData.measurements != null ? phaseData.measurements : metrics.measurements;

                    const uncert = phaseData.uncert != null ? phaseData.uncert : metrics.uncert;
                    const absErr = phaseData.abs_err_x != null ? phaseData.abs_err_x : metrics.abs_err_x;

                    // Sobol baseline final uncertainty and error
                    const sobolPlot = findSobolBaselineForPlot(plotContext);
                    const sobolOverallUncert = sobolPlot ? sobolPlot.uncert : (phaseData.sobol_freq_uncert_at_conv || metrics.sobol_freq_uncert_at_conv);
                    const sobolOverallErr = sobolPlot ? sobolPlot.abs_err_x : (phaseData.sobol_freq_err_at_conv || metrics.sobol_freq_err_at_conv);

                    const overallStepsExpected = (sobolBaseline != null && measurements != null && measurements < sobolBaseline);
                    const sbedOverallErrExpected = (absErr != null && uncert != null && absErr < uncert);
                    const sobolOverallErrExpected = (sobolOverallErr != null && sobolOverallUncert != null && sobolOverallErr < sobolOverallUncert);
                    const overallUncertDiffExpected = (sobolOverallUncert != null && uncert != null && sobolOverallUncert - uncert > 0);
                    const overallErrDiffExpected = (sobolOverallErr != null && absErr != null && sobolOverallErr - absErr > 0);

                    return [
                        // Row 1: Steps
                        [
                            { label: 'Sobol baseline', val: sobolBaseline != null ? formatCount(sobolBaseline) : 'N/A', tip: 'Measurements required by a simple Sobol sweep with Bayesian convergence to resolve this distribution.', cardClass: overallStepsExpected ? 'expected-card' : '' },
                            { label: 'Sbed overall steps', val: measurements != null ? formatCount(measurements) : 'N/A', tip: 'Total measurements taken during Sbed active locator run.', cardClass: overallStepsExpected ? 'expected-card' : '' },
                            { label: 'Overall savings', val: (sobolBaseline != null && measurements != null) ? formatCount(sobolBaseline - measurements) : 'N/A', tip: 'Difference in total measurements required for overall convergence between simple Sobol baseline and Sbed.', cardClass: overallStepsExpected ? 'expected-card' : '' }
                        ],
                        // Row 2: Uncertainty
                        [
                            { label: 'Sobol overall uncertainty', val: sobolOverallUncert != null ? formatFrequency(sobolOverallUncert) : 'N/A', tip: 'Final estimated standard deviation of Sobol baseline frequency estimate.', cardClass: sobolOverallErrExpected ? 'expected-card' : '' },
                            { label: 'Sbed overall uncertainty', val: uncert != null ? formatFrequency(uncert) : 'N/A', tip: 'Final estimated standard deviation of Sbed frequency estimate.', cardClass: sbedOverallErrExpected ? 'expected-card' : '' },
                            { label: 'Overall uncert difference', val: (sobolOverallUncert != null && uncert != null) ? formatFrequency(sobolOverallUncert - uncert) : 'N/A', tip: 'Difference in final frequency estimate uncertainty (positive = SBED was more confident).', cardClass: overallUncertDiffExpected ? 'expected-card' : '' }
                        ],
                        // Row 3: Absolute Error
                        [
                            { label: 'Sobol overall error', val: sobolOverallErr != null ? formatFrequency(sobolOverallErr) : 'N/A', tip: 'Final absolute frequency error of Sobol baseline.', cardClass: sobolOverallErrExpected ? 'expected-card' : '' },
                            { label: 'Sbed overall error', val: absErr != null ? formatFrequency(absErr) : 'N/A', tip: 'Final absolute frequency error of Sbed.', cardClass: sbedOverallErrExpected ? 'expected-card' : '' },
                            { label: 'Overall error difference', val: (sobolOverallErr != null && absErr != null) ? formatFrequency(sobolOverallErr - absErr) : 'N/A', tip: 'Difference in final absolute frequency error (positive = SBED was more accurate).', cardClass: overallErrDiffExpected ? 'expected-card' : '' }
                        ]
                    ];
                }

                function buildEarlyStopComparisonRows(phaseData) {
                    if (!phaseData) return [];
                    const metrics = phaseData.metrics || {};
                    
                    const measurements = phaseData.measurements != null ? phaseData.measurements : metrics.measurements;
                    const stepsToFb = phaseData.steps_to_fb != null ? phaseData.steps_to_fb : metrics.steps_to_fb;
                    
                    const uncert = phaseData.uncert != null ? phaseData.uncert : metrics.uncert;
                    const uncertFb = phaseData.uncert_fb_at_milestone != null ? phaseData.uncert_fb_at_milestone : metrics.uncert_fb_at_milestone;
                    
                    const absErr = phaseData.abs_err_x != null ? phaseData.abs_err_x : metrics.abs_err_x;
                    const errFb = phaseData.err_fb_at_milestone != null ? phaseData.err_fb_at_milestone : metrics.err_fb_at_milestone;

                    const earlyStopStepsExpected = (measurements != null && stepsToFb != null && stepsToFb < measurements);
                    const earlyStopUncertExpected = (uncert != null && uncertFb != null && uncert < uncertFb);
                    const earlyStopErrExpected = (absErr != null && errFb != null && absErr < errFb);

                    return [
                        // Row 1: Steps
                        [
                            { label: 'Sbed overall steps', val: measurements != null ? formatCount(measurements) : 'N/A', tip: 'Total measurements taken during Sbed active locator run.', cardClass: earlyStopStepsExpected ? 'expected-card' : '' },
                            { label: 'Sbed freq convergence', val: stepsToFb != null ? formatCount(stepsToFb) : 'N/A', tip: 'Steps needed for Sbed frequency uncertainty to drop below threshold.', cardClass: earlyStopStepsExpected ? 'expected-card' : '' },
                            { label: 'Early stopping savings', val: (measurements != null && stepsToFb != null) ? formatCount(measurements - stepsToFb) : 'N/A', tip: 'Measurements saved by stopping active locator immediately after frequency converges.', cardClass: earlyStopStepsExpected ? 'expected-card' : '' }
                        ],
                        // Row 2: Uncertainty
                        [
                            { label: 'Sbed final uncertainty', val: uncert != null ? formatFrequency(uncert) : 'N/A', tip: 'Final frequency estimate uncertainty (standard deviation) at locator termination.', cardClass: earlyStopUncertExpected ? 'expected-card' : '' },
                            { label: 'Sbed freq uncertainty', val: uncertFb != null ? formatFrequency(uncertFb) : 'N/A', tip: 'Frequency estimate uncertainty (standard deviation) at the moment fb converged.', cardClass: earlyStopUncertExpected ? 'expected-card' : '' },
                            { label: 'Freq to final uncert diff', val: (uncert != null && uncertFb != null) ? formatFrequency(uncertFb - uncert) : 'N/A', tip: 'Uncertainty reduction achieved by continuing to run from fb convergence until locator termination.', cardClass: earlyStopUncertExpected ? 'expected-card' : '' }
                        ],
                        // Row 3: Absolute Error
                        [
                            { label: 'Sbed final error', val: absErr != null ? formatFrequency(absErr) : 'N/A', tip: 'Final absolute frequency error vs ground truth at locator termination.', cardClass: earlyStopErrExpected ? 'expected-card' : '' },
                            { label: 'Sbed freq error', val: errFb != null ? formatFrequency(errFb) : 'N/A', tip: 'Absolute frequency error vs ground truth at the moment fb converged.', cardClass: earlyStopErrExpected ? 'expected-card' : '' },
                            { label: 'Freq to final error diff', val: (absErr != null && errFb != null) ? formatFrequency(errFb - absErr) : 'N/A', tip: 'Absolute error reduction achieved by continuing to run from fb convergence until locator termination.', cardClass: earlyStopErrExpected ? 'expected-card' : '' }
                        ]
                    ];
                }

                function renderRowsToHtml(rows) {
                    if (!rows || rows.length === 0) return '';
                    let html = '';
                    for (const row of rows) {
                        if (row && row.length > 0) {
                            html += '<div class="scan-metrics-panel" style="margin-bottom:0.5em;">' + renderItemsToHtml(row) + '</div>';
                        }
                    }
                    return html;
                }

                if (plot.coarse && plot.fine) {
                    const totalMeasurements = plot.measurements || (plot.coarse.measurements + plot.fine.measurements);
                    scanMetrics.className = 'scan-metrics-wrapper';
                    let html =
                        '<div style="margin-bottom:0.4em;font-weight:600;color:#334155;font-size:0.85em;">' + escapeHtml(plot.coarse.label) + '</div>' +
                        '<div class="scan-metrics-panel">' + renderItemsToHtml(buildScanItems(plot.coarse, false, totalMeasurements)) + '</div>' +
                        '<div style="margin-top:0.75em;margin-bottom:0.4em;font-weight:600;color:#334155;font-size:0.85em;">' + escapeHtml(plot.fine.label) + '</div>' +
                        '<div class="scan-metrics-panel">' + renderItemsToHtml(buildScanItems(plot.fine, true, totalMeasurements)) + '</div>';

                    if (plot.true_params) {
                        html += '<div style="margin-top:0.75em;margin-bottom:0.4em;font-weight:600;color:#334155;font-size:0.85em;">' + escapeHtml(plot.true_params.label) + '</div>' +
                            '<div class="scan-metrics-panel">' + renderItemsToHtml(buildTrueParamItems(plot.true_params, plot), true) + '</div>';
                    }
                    scanMetrics.innerHTML = html;
                    setupRepeatComparisonUI(plot);
                } else {
                    scanMetrics.className = 'scan-metrics-wrapper';
                    let html = '<div class="scan-metrics-panel">' + renderItemsToHtml(buildScanItems(plot, true)) + '</div>';

                    if (plot.true_params) {
                        html += '<div style="margin-top:0.75em;margin-bottom:0.4em;font-weight:600;color:#334155;font-size:0.85em;">' + escapeHtml(plot.true_params.label) + '</div>' +
                            '<div class="scan-metrics-panel">' + renderItemsToHtml(buildTrueParamItems(plot.true_params, plot), true) + '</div>';
                    }
                    scanMetrics.innerHTML = html;
                    setupRepeatComparisonUI(plot);
                }
                renderSweepMetricsPanel(scanSweepMetrics, plot.metrics || {});
                updateBayesView(plot);
                updateBayesStatsView(plot);
                updateBayesInteractiveView(plot);
                updateBayesTabs();
            } else {
                scanIframe.src = '';
                scanMetrics.className = '';
                scanMetrics.innerHTML = '';
                const scanCmpSel = document.getElementById('scan-comparison-selector');
                if (scanCmpSel) scanCmpSel.innerHTML = '';
                const scanCmpCards = document.getElementById('scan-comparison-cards');
                if (scanCmpCards) scanCmpCards.innerHTML = '';
                if (scanSweepMetrics) { scanSweepMetrics.hidden = true; scanSweepMetrics.innerHTML = ''; }
                updateBayesView(null);
                updateBayesStatsView(null);
                updateBayesInteractiveView(null);
                updateBayesTabs();
            }
        } else {
            scanIframe.src = '';
            scanMetrics.textContent = '';
            const scanCmpSel2 = document.getElementById('scan-comparison-selector');
            if (scanCmpSel2) scanCmpSel2.innerHTML = '';
            const scanCmpCards2 = document.getElementById('scan-comparison-cards');
            if (scanCmpCards2) scanCmpCards2.innerHTML = '';
            if (scanSweepMetrics) { scanSweepMetrics.hidden = true; scanSweepMetrics.innerHTML = ''; }
            updateBayesView(null);
            updateBayesStatsView(null);
            updateBayesInteractiveView(null);
            updateBayesTabs();
        }

        // Check view mode and render summary if needed
        const activeViewModeBtn = document.querySelector('#scan-view-mode button.is-active');
        if (activeViewModeBtn && activeViewModeBtn.dataset.value === 'summary') {
            renderRepeatsSummary(scanGeneratorValue, scanNoiseValue, scanStrategyValue);
        }
    }

    function buildTrueParamItems(trueData, plot) {
        const items = [];
        const params = trueData.params || {};
        const bounds = trueData.bounds || {};
        // Preferred order for common parameters
        const preferred = ['frequency', 'linewidth', 'fwhm_total', 'split', 'dip_depth', 'k_np', 'lorentz_frac'];
        const keys = Object.keys(params).sort((a, b) => {
            const ia = preferred.indexOf(a);
            const ib = preferred.indexOf(b);
            if (ia !== -1 && ib !== -1) return ia - ib;
            if (ia !== -1) return -1;
            if (ib !== -1) return 1;
            return a.localeCompare(b);
        });

        for (const name of keys) {
            const val = params[name];
            const b = bounds[name];
            let label = name.replace(/_/g, ' ');
            // Capitalize first letter
            label = label.charAt(0).toUpperCase() + label.slice(1);

            let formatted = val;
            let fmtLo = b ? b[0] : null;
            let fmtHi = b ? b[1] : null;

            const lowName = name.toLowerCase();
            const isFreqLike = lowName.includes('freq') || lowName.includes('linewidth') || lowName.includes('split') || lowName === 'fwhm_total';

            if (typeof val === 'number') {
                if (isFreqLike) {
                    formatted = formatFrequency(val);
                    if (b) {
                        fmtLo = formatFrequency(b[0]);
                        fmtHi = formatFrequency(b[1]);
                    }
                } else if (lowName === 'dip_depth' || lowName === 'k_np' || lowName === 'lorentz_frac') {
                    formatted = val.toFixed(3);
                    if (b) {
                        fmtLo = b[0].toFixed(3);
                        fmtHi = b[1].toFixed(3);
                    }
                } else {
                    formatted = formatMetricValue(val);
                    if (b) {
                        fmtLo = formatMetricValue(b[0]);
                        fmtHi = formatMetricValue(b[1]);
                    }
                }
            }

            // Extract final estimate and milestone fb if available
            let finalEst = null;
            if (plot && plot.metrics) {
                finalEst = plot.metrics["final_est_" + name];
            }

            let fbAtMilestone = null;
            if (plot && plot.metrics && name === 'frequency') {
                fbAtMilestone = plot.metrics["fb_at_milestone"];
            }

            items.push({
                label: label,
                val: formatted,
                bounds: b,
                rawVal: val,
                fmtLo: fmtLo,
                fmtHi: fmtHi,
                name: name,
                finalEst: finalEst,
                fbAtMilestone: fbAtMilestone
            });
        }
        return items;
    }

    // ---- Covariance ellipse renderer (JS-native, from JSON data) ----

    function registerTimelineAdapter(adapter) {
        if (!adapter || !adapter.contentWindow) return;
        activeIframes.add(adapter);
        updateGlobalTimelineMetadata();
    }

    /**
     * Compute 2σ covariance ellipse points for a 2×2 sub-covariance,
     * centered at (cx, cy). Replicates the Python eigensystem approach exactly.
     */
    function computeEllipse(cov2d, cx, cy) {
        const std_i = Math.sqrt(Math.max(cov2d[0][0], 1e-30));
        const std_j = Math.sqrt(Math.max(cov2d[1][1], 1e-30));
        let rho = cov2d[0][1] / (std_i * std_j);
        if (!isFinite(rho)) rho = 0;
        rho = Math.max(-0.9999, Math.min(0.9999, rho));
        // Eigensystem of [[1, ρ], [ρ, 1]] — eigh returns ascending eigenvalues.
        // For ρ≥0:  λ_a=1-ρ (v_a=[1,-1]/√2),  λ_b=1+ρ (v_b=[1,1]/√2)
        // For ρ<0:  λ_a=1+ρ (v_a=[1, 1]/√2),   λ_b=1-ρ (v_b=[1,-1]/√2)
        const va = rho >= 0 ? [1 / Math.SQRT2, -1 / Math.SQRT2] : [1 / Math.SQRT2, 1 / Math.SQRT2];
        const vb = rho >= 0 ? [1 / Math.SQRT2,  1 / Math.SQRT2] : [1 / Math.SQRT2, -1 / Math.SQRT2];
        const a_half = 2 * Math.sqrt(Math.max(1 - Math.abs(rho), 0));
        const b_half = 2 * Math.sqrt(Math.max(1 + Math.abs(rho), 0));
        const N = 100;
        const xs = new Array(N + 1), ys = new Array(N + 1);
        for (let k = 0; k <= N; k++) {
            const theta = (2 * Math.PI * k) / N;
            const ex = a_half * Math.cos(theta);
            const ey = b_half * Math.sin(theta);
            // Rotate by eigenvector matrix (rot.T in Python notation)
            xs[k] = cx + (ex * va[0] + ey * vb[0]) * std_i;
            ys[k] = cy + (ex * va[1] + ey * vb[1]) * std_j;
        }
        return { x: xs, y: ys };
    }

    /**
     * Render covariance ellipses from a JSON data file.
     * Creates one independent Plotly chart per parameter pair, side by side.
     * Calls onReady(adapter) once rendering is done so the caller can register
     * the adapter with the global timeline.
     */
    async function renderCovarianceEllipses(container, jsonPath, onReady) {
        container.innerHTML = '<div style="padding:2em;color:#64748b;text-align:center;">Loading covariance data…</div>';

        let data;
        try {
            const resp = await fetch(jsonPath, { cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            data = await resp.json();
        } catch (e) {
            container.innerHTML = `<div style="padding:2em;color:#ef4444;">Failed to load covariance data: ${escapeHtml(String(e.message))}</div>`;
            return;
        }

        if (data.schema !== 'covariance_ellipses_v1' || !data.steps || !data.pairs) {
            container.innerHTML = '<div style="padding:2em;color:#64748b;">No covariance data available.</div>';
            return;
        }

        // Guard against stale renders: if container was cleared while we were fetching, bail out.
        if (!container.isConnected) return;

        const { param_names, param_units, pairs, steps, true_params } = data;
        const nSteps = steps.length;

        await ensurePlotly();
        if (!container.isConnected) return;

        // One div per pair, arranged in a flex row
        container.innerHTML = '';
        container.style.cssText = 'display:flex; gap:8px; flex-wrap:wrap; margin-top:8px;';

        const plotDivs = [];
        for (const [i, j] of pairs) {
            const pi = param_names[i];
            const pj = param_names[j];
            const piLabel = pi + (param_units[pi] || '');
            const pjLabel = pj + (param_units[pj] || '');

            const wrapper = document.createElement('div');
            wrapper.style.cssText = 'flex:1; min-width:220px; max-width:480px;';

            const titleEl = document.createElement('div');
            titleEl.style.cssText = 'text-align:center; font-size:0.78em; color:#64748b; font-weight:600; margin-bottom:2px;';
            titleEl.textContent = `${piLabel} vs ${pjLabel}`;
            wrapper.appendChild(titleEl);

            const plotDiv = document.createElement('div');
            wrapper.appendChild(plotDiv);
            container.appendChild(wrapper);
            plotDivs.push({ div: plotDiv, i, j, piLabel, pjLabel, pi, pj });
        }

        function updateEllipseStep(stepIdx) {
            const step = steps[Math.max(0, Math.min(stepIdx, nSteps - 1))];
            const cov = step.covariance;
            const means = step.means || {};
            const progress = stepIdx / Math.max(1, nSteps - 1);
            const alpha = 0.25 + 0.55 * progress;
            const fillColor = `rgba(0, 100, 200, ${(alpha * 0.6).toFixed(2)})`;
            const lineColor = `rgba(0, 100, 200, ${alpha.toFixed(2)})`;

            for (const { div, i, j, piLabel, pjLabel, pi, pj } of plotDivs) {
                const cov2d = [[cov[i][i], cov[i][j]], [cov[j][i], cov[j][j]]];
                const cx = (means[pi] != null) ? means[pi] : 0;
                const cy = (means[pj] != null) ? means[pj] : 0;
                const { x, y } = computeEllipse(cov2d, cx, cy);

                const traces = [{
                    type: 'scatter', x, y,
                    mode: 'lines',
                    line: { color: lineColor, width: 2 },
                    fill: 'toself',
                    fillcolor: fillColor,
                    hoverinfo: 'skip',
                    showlegend: false,
                }];

                if (true_params) {
                    const ti = true_params[pi], tj = true_params[pj];
                    if (ti != null && tj != null) {
                        traces.push({
                            type: 'scatter', x: [ti], y: [tj],
                            mode: 'markers',
                            marker: { symbol: 'x', color: 'red', size: 10, line: { color: 'red', width: 2 } },
                            name: 'True', showlegend: false,
                        });
                    }
                }

                const layout = {
                    template: 'plotly_white',
                    margin: { l: 55, r: 10, t: 10, b: 50 },
                    xaxis: { title: { text: piLabel, font: { size: 11 } }, autorange: true },
                    yaxis: { title: { text: pjLabel, font: { size: 11 } }, autorange: true },
                    height: 260,
                    hovermode: false,
                };

                Plotly.react(div, traces, layout, { displayModeBar: false });
            }
        }

        updateEllipseStep(0);

        const adapter = {
            contentWindow: {
                showFrame: updateEllipseStep,
                totalFrames: nSteps,
                stepValues: steps.map((_, k) => k),
            },
            addEventListener: () => {},
        };

        if (onReady) onReady(adapter);
    }

    // ---- End covariance ellipse renderer ----

    // ---- JSON-based render functions (posterior, convergence, Fisher) ----

    /** Weighted Gaussian KDE at nGrid evenly-spaced points over [lo, hi]. */
    function _weightedKDE(values, weights, lo, hi, nGrid) {
        const n = values.length;
        let wSum = 0;
        for (let i = 0; i < n; i++) wSum += weights[i];
        if (wSum <= 0 || n === 0) return { grid: [], kde: [] };

        let mean = 0;
        for (let i = 0; i < n; i++) mean += (weights[i] / wSum) * values[i];
        let variance = 0;
        for (let i = 0; i < n; i++) {
            const d = values[i] - mean;
            variance += (weights[i] / wSum) * d * d;
        }
        const sigma = Math.sqrt(Math.max(variance, 1e-30));
        const bw = Math.max(1.06 * sigma * Math.pow(n, -0.2), (hi - lo) * 1e-4);

        const grid = [], kde = [];
        const step = (hi - lo) / (nGrid - 1);
        const inv_bw_sqrt2pi = 1.0 / (bw * Math.sqrt(2 * Math.PI));
        const inv_bw = 1.0 / bw;
        for (let g = 0; g < nGrid; g++) {
            const x = lo + g * step;
            grid.push(x);
            let density = 0;
            for (let i = 0; i < n; i++) {
                const z = (x - values[i]) * inv_bw;
                density += (weights[i] / wSum) * Math.exp(-0.5 * z * z);
            }
            kde.push(density * inv_bw_sqrt2pi);
        }
        return { grid, kde };
    }

    async function renderPosterior(container, jsonPath, onReady) {
        container.innerHTML = '<div style="padding:2em;color:#64748b;text-align:center;">Loading posterior data…</div>';

        let data;
        try {
            const resp = await fetch(jsonPath, { cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            data = await resp.json();
        } catch (e) {
            container.innerHTML = `<div style="padding:2em;color:#ef4444;">Failed to load posterior data: ${escapeHtml(String(e.message))}</div>`;
            return;
        }

        if (data.schema !== 'posterior_v1' || !data.steps || !data.param_names) {
            container.innerHTML = '<div style="padding:2em;color:#64748b;">No posterior data available.</div>';
            return;
        }
        if (!container.isConnected) return;

        const { param_names, param_units, physical_bounds, true_params, resampled_steps, steps } = data;
        const nSteps = steps.length;
        const nParams = param_names.length;

        await ensurePlotly();
        if (!container.isConnected) return;

        container.innerHTML = '';
        container.style.cssText = 'display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;';

        const plotDivs = [];
        for (const param of param_names) {
            const unit = param_units[param] ? ` (${param_units[param]})` : '';
            const wrapper = document.createElement('div');
            wrapper.style.cssText = 'flex:1; min-width:200px; max-width:420px;';
            const title = document.createElement('div');
            title.style.cssText = 'text-align:center; font-size:0.78em; color:#64748b; font-weight:600; margin-bottom:2px;';
            title.textContent = param + unit;
            wrapper.appendChild(title);
            const plotDiv = document.createElement('div');
            wrapper.appendChild(plotDiv);
            container.appendChild(wrapper);
            plotDivs.push({ div: plotDiv, param });
        }

        function updateStep(stepIdx) {
            const step = steps[Math.max(0, Math.min(stepIdx, nSteps - 1))];
            const isResampled = resampled_steps && resampled_steps.includes(stepIdx);

            for (const { div, param } of plotDivs) {
                const entry = step[param];
                if (!entry) continue;

                const unit = param_units[param] ? ` (${param_units[param]})` : '';
                const bounds = physical_bounds[param];
                const traces = [];

                if (entry.type === 'particles') {
                    const lo = bounds ? bounds[0] : Math.min(...entry.values);
                    const hi = bounds ? bounds[1] : Math.max(...entry.values);
                    const { grid, kde } = _weightedKDE(entry.values, entry.weights, lo, hi, 120);
                    traces.push({
                        type: 'scatter', x: grid, y: kde,
                        mode: 'lines',
                        fill: 'tozeroy',
                        fillcolor: isResampled ? 'rgba(255,140,0,0.18)' : 'rgba(0,100,200,0.18)',
                        line: { color: isResampled ? 'rgba(255,140,0,0.8)' : 'rgba(0,100,200,0.8)', width: 1.5 },
                        hoverinfo: 'skip', showlegend: false,
                    });
                } else if (entry.type === 'grid') {
                    traces.push({
                        type: 'bar', x: entry.axis, y: entry.posterior,
                        marker: { color: 'rgba(0,100,200,0.5)' },
                        hoverinfo: 'skip', showlegend: false,
                    });
                }

                if (true_params && true_params[param] != null) {
                    const tv = true_params[param];
                    traces.push({
                        type: 'scatter', x: [tv, tv], y: [0, 1],
                        mode: 'lines', yaxis: 'y',
                        line: { color: 'rgba(200,30,30,0.8)', width: 1.5, dash: 'dash' },
                        hoverinfo: 'skip', showlegend: false,
                    });
                }

                const xlo = bounds ? bounds[0] : undefined;
                const xhi = bounds ? bounds[1] : undefined;
                const layout = {
                    template: 'plotly_white',
                    margin: { l: 10, r: 10, t: 4, b: 36 },
                    xaxis: { title: { text: param + unit, font: { size: 10 } }, range: [xlo, xhi], autorange: xlo == null },
                    yaxis: { visible: false },
                    height: 180,
                    hovermode: false,
                };
                Plotly.react(div, traces, layout, { displayModeBar: false });
            }
        }

        updateStep(0);

        const adapter = {
            contentWindow: {
                showFrame: updateStep,
                totalFrames: nSteps,
                stepValues: steps.map((_, k) => k),
            },
            addEventListener: () => {},
        };
        if (onReady) onReady(adapter);
    }

    async function renderParameterConvergence(container, jsonPath) {
        container.innerHTML = '<div style="padding:1em;color:#64748b;text-align:center;">Loading…</div>';

        let data;
        try {
            const resp = await fetch(jsonPath, { cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            data = await resp.json();
        } catch (e) {
            container.innerHTML = `<div style="padding:1em;color:#ef4444;">Failed to load convergence data: ${escapeHtml(String(e.message))}</div>`;
            return;
        }

        if (data.schema !== 'parameter_convergence_v1' || !data.steps) {
            container.innerHTML = '<div style="padding:1em;color:#64748b;">No convergence data.</div>';
            return;
        }
        if (!container.isConnected) return;

        await ensurePlotly();
        if (!container.isConnected) return;

        const { param_names, param_units, true_params, steps } = data;
        const xs = steps.map((_, i) => i);
        const COLORS = ['#1e90ff', '#e05c00', '#00a878', '#9b59b6', '#e74c3c', '#2ecc71', '#f39c12'];

        const traces = [];
        param_names.forEach((param, ci) => {
            const color = COLORS[ci % COLORS.length];
            const unit = param_units[param] ? ` (${param_units[param]})` : '';
            const uncerts = steps.map((s) => s.uncertainties[param] ?? null);
            traces.push({
                type: 'scatter', x: xs, y: uncerts,
                mode: 'lines', name: `${param}${unit} σ`,
                line: { color, width: 2 },
            });
            if (true_params && true_params[param] != null) {
                traces.push({
                    type: 'scatter', x: [xs[0], xs[xs.length - 1]], y: [true_params[param], true_params[param]],
                    mode: 'lines', name: `${param} true`,
                    line: { color, width: 1, dash: 'dot' },
                    showlegend: false,
                });
            }
        });

        const layout = {
            template: 'plotly_white',
            margin: { l: 60, r: 20, t: 20, b: 50 },
            xaxis: { title: 'Step' },
            yaxis: { title: 'Uncertainty' },
            legend: { orientation: 'h', y: -0.25 },
            height: 320,
        };

        container.innerHTML = '';
        const plotDiv = document.createElement('div');
        container.appendChild(plotDiv);
        Plotly.react(plotDiv, traces, layout, { responsive: true, displayModeBar: false });
    }

    async function renderConvergenceMetrics(container, jsonPath) {
        container.innerHTML = '<div style="padding:1em;color:#64748b;text-align:center;">Loading…</div>';

        let data;
        try {
            const resp = await fetch(jsonPath, { cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            data = await resp.json();
        } catch (e) {
            container.innerHTML = `<div style="padding:1em;color:#ef4444;">Failed to load metrics: ${escapeHtml(String(e.message))}</div>`;
            return;
        }

        if (data.schema !== 'convergence_metrics_v1' || !data.steps) {
            container.innerHTML = '<div style="padding:1em;color:#64748b;">No convergence metrics.</div>';
            return;
        }
        if (!container.isConnected) return;

        await ensurePlotly();
        if (!container.isConnected) return;

        const { param_names, param_units, convergence_threshold, steps } = data;
        const xs = steps.map((s) => s.step);
        const COLORS = ['#1e90ff', '#e05c00', '#00a878', '#9b59b6', '#e74c3c', '#2ecc71', '#f39c12'];

        const traces = [];
        // Threshold line at 1.0 (uncertainties are already relative to the threshold)
        traces.push({
            type: 'scatter', x: [xs[0], xs[xs.length - 1]], y: [1, 1],
            mode: 'lines', name: 'Threshold',
            line: { color: 'rgba(200,30,30,0.6)', width: 1.5, dash: 'dash' },
        });

        param_names.forEach((param, ci) => {
            const color = COLORS[ci % COLORS.length];
            const unit = param_units[param] ? ` (${param_units[param]})` : '';
            const vals = steps.map((s) => s.uncertainties[param] ?? null);
            traces.push({
                type: 'scatter', x: xs, y: vals,
                mode: 'lines', name: `${param}${unit}`,
                line: { color, width: 2 },
            });
        });

        // Shade converged region
        const convergedStart = steps.findIndex((s) => s.convergence_achieved);
        if (convergedStart >= 0) {
            traces.push({
                type: 'scatter',
                x: [xs[convergedStart], xs[xs.length - 1], xs[xs.length - 1], xs[convergedStart]],
                y: [0, 0, Math.max(...steps.map((s) => Math.max(...param_names.map((p) => s.uncertainties[p] ?? 0)))), 0],
                fill: 'toself', fillcolor: 'rgba(0,180,80,0.08)',
                mode: 'none', hoverinfo: 'skip', showlegend: false,
            });
        }

        const layout = {
            template: 'plotly_white',
            margin: { l: 60, r: 20, t: 20, b: 50 },
            xaxis: { title: 'Step' },
            yaxis: { title: 'Relative uncertainty', rangemode: 'tozero' },
            legend: { orientation: 'h', y: -0.25 },
            height: 300,
        };

        container.innerHTML = '';
        const plotDiv = document.createElement('div');
        container.appendChild(plotDiv);
        Plotly.react(plotDiv, traces, layout, { responsive: true, displayModeBar: false });
    }

    async function renderFisher(container, jsonPath) {
        container.innerHTML = '<div style="padding:1em;color:#64748b;text-align:center;">Loading…</div>';

        let data;
        try {
            const resp = await fetch(jsonPath, { cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            data = await resp.json();
        } catch (e) {
            container.innerHTML = `<div style="padding:1em;color:#ef4444;">Failed to load Fisher data: ${escapeHtml(String(e.message))}</div>`;
            return;
        }

        if (data.schema !== 'fisher_v1' || !data.steps) {
            container.innerHTML = '<div style="padding:1em;color:#64748b;">No Fisher information data.</div>';
            return;
        }
        if (!container.isConnected) return;

        await ensurePlotly();
        if (!container.isConnected) return;

        const { param_names, param_units, steps } = data;
        const xs = steps.map((_, i) => i);
        const COLORS = ['#1e90ff', '#e05c00', '#00a878', '#9b59b6', '#e74c3c', '#2ecc71', '#f39c12'];

        // One subplot per parameter: CRLB (lower bound) vs actual uncertainty
        container.innerHTML = '';
        container.style.cssText = 'display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;';

        for (const [ci, param] of param_names.entries()) {
            const unit = param_units[param] ? ` (${param_units[param]})` : '';
            const color = COLORS[ci % COLORS.length];

            const crlb = steps.map((s) => s.fisher_bounds[param] ?? null);
            const actual = steps.map((s) => s.actual_uncertainty[param] ?? null);

            const traces = [
                {
                    type: 'scatter', x: xs, y: actual,
                    mode: 'lines', name: 'Actual σ',
                    line: { color, width: 2 },
                },
                {
                    type: 'scatter', x: xs, y: crlb,
                    mode: 'lines', name: 'CRLB',
                    line: { color, width: 1.5, dash: 'dash' },
                    fill: 'tonexty', fillcolor: `${color}22`,
                },
            ];

            const layout = {
                template: 'plotly_white',
                margin: { l: 55, r: 10, t: 28, b: 40 },
                title: { text: param + unit, font: { size: 11 }, y: 0.96 },
                xaxis: { title: { text: 'Step', font: { size: 10 } } },
                yaxis: { title: { text: 'Uncertainty', font: { size: 10 } }, rangemode: 'tozero' },
                showlegend: ci === 0,
                legend: { orientation: 'h', y: -0.3 },
                height: 240,
            };

            const wrapper = document.createElement('div');
            wrapper.style.cssText = 'flex:1; min-width:200px; max-width:420px;';
            container.appendChild(wrapper);
            Plotly.react(wrapper, traces, layout, { displayModeBar: false, responsive: true });
        }
    }

    // ---- End JSON-based render functions ----

    function renderJitterView(container, jitterPlot) {
        if (!container || !jitterPlot) return;

        const jitter = jitterPlot.jitter || {};
        const variances = jitterPlot.variances || {};
        const correlations = jitterPlot.correlations || {};
        const paramNames = Object.keys(jitter);

        const jitterItems = paramNames.map(name => ({
            label: name,
            val: formatMetricValue(jitter[name]),
            tip: 'Standard deviation of estimates over last 20 steps (physical units).',
            rawVal: jitter[name]
        }));

        const varItems = Object.keys(variances).length > 0
            ? paramNames.map(name => ({ label: name, val: formatMetricValue(variances[name]), rawVal: variances[name] }))
            : [];

        // Jitter cards row
        let html = '<h4>Jitter (last 20 steps)</h4>';
        html += '<div class="jitter-cards">' + renderItemsToHtml(jitterItems) + '</div>';

        // Variance cards row
        if (varItems.length > 0) {
            html += '<h4>Final Variances (diag Σ)</h4>';
            html += '<div class="jitter-cards">' + renderItemsToHtml(varItems) + '</div>';
        }

        // Correlation matrix — full-width block, not inside the card grid
        if (Object.keys(correlations).length > 0) {
            html += '<h4>Correlation Matrix <span class="help-icon" tabindex="0" title="Measures how tightly parameters are coupled. +1/-1 = perfect correlation; 0 = independent.">?</span></h4>';
            html += '<div class="correlation-table-wrapper"><table class="correlation-table">';
            html += '<thead><tr><th></th>' + paramNames.map(n => `<th>${escapeHtml(n)}</th>`).join('') + '</tr></thead>';
            html += '<tbody>';
            for (const ni of paramNames) {
                html += `<tr><th>${escapeHtml(ni)}</th>`;
                for (const nj of paramNames) {
                    const val = (correlations[ni] && correlations[ni][nj] !== undefined) ? correlations[ni][nj] : 0;
                    let color, textColor;
                    if (ni === nj) {
                        color = 'rgba(71, 85, 105, 0.15)';
                        textColor = '#1e293b';
                    } else if (val > 0) {
                        color = `rgba(30, 144, 255, ${0.1 + 0.9 * val})`;
                        textColor = val > 0.4 ? '#ffffff' : '#1e3a8a';
                    } else {
                        color = `rgba(239, 68, 68, ${0.1 + 0.9 * Math.abs(val)})`;
                        textColor = Math.abs(val) > 0.4 ? '#ffffff' : '#7f1d1d';
                    }
                    html += `<td class="corr-cell" style="background-color:${color}; color:${textColor};">${val.toFixed(2)}</td>`;
                }
                html += '</tr>';
            }
            html += '</tbody></table></div>';
        }

        container.innerHTML = html;
    }

    function renderItemsToHtml(items, useSliders = false) {
        return items.map(it => {
            const tipAttr = it.tip ? ' title="' + it.tip.replace(/"/g, '&quot;') + '"' : '';
            const icon = it.tip ? '<span class="help-icon" tabindex="0"' + tipAttr + '>?</span>' : '';

            let valueHtml = '<div class="metric-value">' + it.val + '</div>';

            if (useSliders && it.bounds && typeof it.rawVal === 'number') {
                const lo = it.bounds[0];
                const hi = it.bounds[1];
                const percent = Math.min(100, Math.max(0, (it.rawVal - lo) / (hi - lo) * 100));

                let markersHtml = '<div class="param-range-marker" title="True Value: ' + it.val + '" style="left: ' + percent + '%; background-color: #2563eb; z-index: 10;"></div>';

                if (typeof it.finalEst === 'number' && Number.isFinite(it.finalEst)) {
                    const finalPct = Math.min(100, Math.max(0, (it.finalEst - lo) / (hi - lo) * 100));
                    const formattedFinal = it.name && (it.name.toLowerCase().includes('freq') || it.name.toLowerCase().includes('linewidth') || it.name.toLowerCase().includes('split') || it.name === 'fwhm_total') ? formatFrequency(it.finalEst) : it.finalEst.toFixed(3);
                    markersHtml += '<div class="param-range-marker" title="Final Inferred: ' + formattedFinal + '" style="left: ' + finalPct + '%; background-color: #ef4444; width: 8px; height: 8px; z-index: 9;"></div>';
                }

                if (typeof it.fbAtMilestone === 'number' && Number.isFinite(it.fbAtMilestone)) {
                    const fbPct = Math.min(100, Math.max(0, (it.fbAtMilestone - lo) / (hi - lo) * 100));
                    const formattedFb = formatFrequency(it.fbAtMilestone);
                    markersHtml += '<div class="param-range-marker" title="Milestone Conv Freq: ' + formattedFb + '" style="left: ' + fbPct + '%; background-color: #f59e0b; width: 8px; height: 8px; z-index: 8;"></div>';
                }

                valueHtml =
                    '<div class="metric-value">' + it.val + '</div>' +
                    '<div class="param-range-container">' +
                    '<div class="param-range-track">' +
                    markersHtml +
                    '</div>' +
                    '<div class="param-range-bounds">' +
                    '<span>' + it.fmtLo + '</span>' +
                    '<span>' + it.fmtHi + '</span>' +
                    '</div>' +
                    '</div>';
            }

            const extraClass = it.cardClass ? ' ' + it.cardClass : '';
            return '<div class="metric-item' + extraClass + '">' +
                '<div class="metric-label">' + it.label + icon + '</div>' +
                valueHtml +
                '</div>';
        }).join('');
    }

    function findSobolBaselineForPlot(plot) {
        if (!window.MANIFEST || !plot) return null;
        return window.MANIFEST.find(p =>
            p.strategy === "SimpleSobol" &&
            p.generator === plot.generator &&
            p.noise === plot.noise &&
            p.repeat === plot.repeat
        );
    }

    function findOtherStrategyPlots(plot) {
        if (!window.MANIFEST || !plot) return [];
        return window.MANIFEST.filter(p =>
            p.type === 'scan' &&
            p.generator === plot.generator &&
            p.noise === plot.noise &&
            p.repeat === plot.repeat &&
            p.strategy !== plot.strategy &&
            p.strategy !== 'SimpleSobol'
        );
    }

    let lastSummaryKey = null;

    // ── Entity helpers ────────────────────────────────────────────────────────
    function _phaseData(p) {
        return (p && p.coarse && p.fine) ? p.fine : p;
    }
    function _mv(obj, ...keys) {
        if (!obj) return null;
        const m = obj.metrics || {};
        for (const k of keys) {
            if (obj[k] != null) return obj[k];
            if (m[k] != null) return m[k];
        }
        return null;
    }

    function buildSummaryEntities(generator, noise) {
        const all = (window.MANIFEST || []).filter(p =>
            p.generator === generator && p.noise === noise && p.type === 'scan'
        );
        const map = new Map();
        for (const p of all) {
            const d = _phaseData(p);
            if (!map.has(p.strategy)) {
                map.set(p.strategy, {
                    id: p.strategy, label: p.strategy,
                    steps: [], uncert: [], err: [],
                    steps_to_fb: [], uncert_at_fb: [], err_at_fb: [],
                    stepsType: 'measurements',
                });
            }
            const e = map.get(p.strategy);
            const push = (arr, v) => { if (v != null) arr.push(v); };
            push(e.steps,        _mv(d, 'measurements'));
            push(e.uncert,       _mv(d, 'uncert'));
            push(e.err,          _mv(d, 'abs_err_x'));
            push(e.steps_to_fb,  _mv(d, 'steps_to_fb'));
            push(e.uncert_at_fb, _mv(d, 'uncert_fb_at_milestone'));
            push(e.err_at_fb,    _mv(d, 'err_fb_at_milestone'));
        }
        const entities = [...map.values()].filter(e => e.steps.length || e.uncert.length);
        // Synthesize Sobol baseline from embedded fields if not in manifest as own strategy
        if (!map.has('SimpleSobol')) {
            const sobol = {
                id: '__sobol__', label: 'Sobol (baseline)',
                steps: [], uncert: [], err: [],
                steps_to_fb: [], uncert_at_fb: [], err_at_fb: [],
                stepsType: 'measurements',
            };
            for (const p of all) {
                const d = _phaseData(p);
                const sp = findSobolBaselineForPlot(p);
                const push = (arr, v) => { if (v != null) arr.push(v); };
                push(sobol.steps,        _mv(d, 'sobol_baseline_steps'));
                push(sobol.steps_to_fb,  _mv(d, 'sobol_freq_steps'));
                push(sobol.uncert_at_fb, _mv(d, 'sobol_freq_uncert_at_conv'));
                push(sobol.err_at_fb,    _mv(d, 'sobol_freq_err_at_conv'));
                // Prefer the explicit final uncert/err captured during the Sobol
                // baseline run; fall back to a separate SimpleSobol manifest entry
                // if one exists (older runs without the embedded fields).
                const sFinUncert = _mv(d, 'sobol_baseline_uncert');
                const sFinErr    = _mv(d, 'sobol_baseline_err');
                push(sobol.uncert, sFinUncert != null ? sFinUncert : (sp ? sp.uncert : null));
                push(sobol.err,    sFinErr    != null ? sFinErr    : (sp ? sp.abs_err_x : null));
            }
            if (sobol.steps.length) entities.push(sobol);
        }
        return entities;
    }

    function buildRepeatEntities(plot) {
        const d = _phaseData(plot);
        const entities = [];
        entities.push({
            id: 'main', label: plot.strategy || 'Active locator',
            steps:        _mv(d, 'measurements'),
            uncert:       _mv(d, 'uncert'),
            err:          _mv(d, 'abs_err_x'),
            steps_to_fb:  _mv(d, 'steps_to_fb'),
            uncert_at_fb: _mv(d, 'uncert_fb_at_milestone'),
            err_at_fb:    _mv(d, 'err_fb_at_milestone'),
            stepsType: 'measurements',
        });
        const sobolSteps = _mv(d, 'sobol_baseline_steps');
        if (sobolSteps != null) {
            const sp = findSobolBaselineForPlot(plot);
            const sFinUncert = _mv(d, 'sobol_baseline_uncert');
            const sFinErr    = _mv(d, 'sobol_baseline_err');
            entities.push({
                id: 'sobol', label: 'Sobol (baseline)',
                steps: sobolSteps,
                uncert: sFinUncert != null ? sFinUncert : (sp ? sp.uncert : null),
                err:    sFinErr    != null ? sFinErr    : (sp ? sp.abs_err_x : null),
                steps_to_fb:  _mv(d, 'sobol_freq_steps'),
                uncert_at_fb: _mv(d, 'sobol_freq_uncert_at_conv'),
                err_at_fb:    _mv(d, 'sobol_freq_err_at_conv'),
                stepsType: 'measurements',
            });
        }
        const stFb = _mv(d, 'steps_to_fb');
        if (stFb != null) {
            entities.push({
                id: 'main_fb', label: (plot.strategy || 'Locator') + ' @ freq conv.',
                steps: stFb,
                uncert: _mv(d, 'uncert_fb_at_milestone'),
                err:    _mv(d, 'err_fb_at_milestone'),
                steps_to_fb: null, uncert_at_fb: null, err_at_fb: null,
                stepsType: 'steps',
            });
        }
        // Add any other strategy runs for the same repeat (e.g. SimpleSweep ↔ SBED).
        for (const other of findOtherStrategyPlots(plot)) {
            const od = _phaseData(other);
            if (_mv(od, 'uncert') == null && _mv(od, 'abs_err_x') == null) continue;
            entities.push({
                id: `other_${other.strategy}`,
                label: other.strategy,
                steps:        _mv(od, 'measurements'),
                uncert:       _mv(od, 'uncert'),
                err:          _mv(od, 'abs_err_x'),
                steps_to_fb:  _mv(od, 'steps_to_fb'),
                uncert_at_fb: _mv(od, 'uncert_fb_at_milestone'),
                err_at_fb:    _mv(od, 'err_fb_at_milestone'),
                stepsType: 'measurements',
            });
        }
        return entities;
    }

    // ── Pairwise metric rows ──────────────────────────────────────────────────
    function buildPairwiseRows(eA, eB) {
        const isArr = v => Array.isArray(v);
        const has = v => isArr(v) ? v.length > 0 : v != null;
        const sub = (a, b) => {
            if (!has(a) || !has(b)) return null;
            if (isArr(a) && isArr(b)) {
                const n = Math.min(a.length, b.length);
                return Array.from({length: n}, (_, i) => b[i] - a[i]);
            }
            return (!isArr(a) && !isArr(b)) ? b - a : null;
        };
        // Δ card header — use the full entity labels. CSS (text-overflow:
        // ellipsis) handles any overflow at render time based on the card's
        // actual width, instead of hard-clipping in JS.
        const deltaLabel = `Δ  ${eB.label} − ${eA.label}`;
        const rows = [];
        const addRow = (metric, aVal, bVal, cA, cB, cD, type) => {
            // Every metric should be present for every entity by construction
            // (the backend captures both "at convergence" and "final" snapshots).
            // If you see a row skipped here, it means the backend data-capture
            // gap has reappeared — investigate rather than hide.
            if (!has(aVal) || !has(bVal)) return;
            rows.push({ label: metric, cards: [
                { label: eA.label, sublabel: metric, data: aVal, color: cA, type },
                { label: eB.label, sublabel: metric, data: bVal, color: cB, type },
                { label: deltaLabel, data: sub(aVal, bVal), color: cD, type },
            ]});
        };
        addRow('Steps to completion',       eA.steps,        eB.steps,        '#f472b6','#60a5fa','#22c55e', eA.stepsType || 'measurements');
        addRow('Final frequency uncertainty',eA.uncert,       eB.uncert,       '#a78bfa','#34d399','#f59e0b','frequency');
        addRow('Final frequency error',      eA.err,          eB.err,          '#c084fc','#10b981','#6366f1','frequency');
        addRow('Steps to freq. convergence', eA.steps_to_fb,  eB.steps_to_fb,  '#fb923c','#38bdf8','#a3e635','steps');
        addRow('Uncertainty @ freq. conv.',  eA.uncert_at_fb, eB.uncert_at_fb, '#818cf8','#2dd4bf','#fbbf24','frequency');
        addRow('Error @ freq. conv.',        eA.err_at_fb,    eB.err_at_fb,    '#d946ef','#4ade80','#f43f5e','frequency');
        return rows;
    }

    // ── Shared selector + card renderer ──────────────────────────────────────
    function buildTwoDropdownSelector(entities, onPairChange) {
        const wrapper = document.createElement('div');
        wrapper.className = 'control-row';
        wrapper.style.cssText = 'gap:0.5em;margin:0.5em 0 0.75em;flex-wrap:wrap;align-items:center;';
        const lbl = document.createElement('span');
        lbl.className = 'control-label';
        lbl.textContent = 'Compare:';
        const selA = document.createElement('select');
        selA.className = 'control-select';
        const vs = document.createElement('span');
        vs.style.cssText = 'padding:0 0.3em;color:#64748b;font-size:0.9em;';
        vs.textContent = 'with';
        const selB = document.createElement('select');
        selB.className = 'control-select';
        for (const e of entities) {
            [selA, selB].forEach(sel => {
                const opt = document.createElement('option');
                opt.value = e.id;
                opt.textContent = e.label;
                sel.appendChild(opt);
            });
        }
        if (entities.length >= 2) selB.selectedIndex = 1;
        const notify = () => {
            const eA = entities.find(e => e.id === selA.value);
            const eB = entities.find(e => e.id === selB.value);
            if (eA && eB) onPairChange(eA, eB);
        };
        selA.addEventListener('change', notify);
        selB.addEventListener('change', notify);
        wrapper.appendChild(lbl);
        wrapper.appendChild(selA);
        wrapper.appendChild(vs);
        wrapper.appendChild(selB);
        return { wrapper, notify };
    }

    function renderPairwiseCards(rows, container, isSummary) {
        container.innerHTML = '';
        if (!rows || !rows.length) {
            const p = document.createElement('p');
            p.style.cssText = 'color:#94a3b8;padding:1em 0;font-size:0.9em;';
            p.textContent = 'No shared metrics for this pair.';
            container.appendChild(p);
            return;
        }
        for (const row of rows) {
            const hdr = document.createElement('div');
            hdr.style.cssText = 'margin-top:1em;margin-bottom:0.4em;font-weight:600;color:#334155;font-size:0.85em;';
            hdr.textContent = row.label;
            container.appendChild(hdr);
            const rowDiv = document.createElement('div');
            rowDiv.className = 'scan-metrics-panel' + (isSummary ? ' summary-grid' : '');
            rowDiv.style.marginBottom = '0.75em';
            container.appendChild(rowDiv);
            for (const card of row.cards) {
                const cardDiv = document.createElement('div');
                cardDiv.className = 'metric-item';
                rowDiv.appendChild(cardDiv);
                const lblDiv = document.createElement('div');
                lblDiv.className = 'metric-label';
                // For entity cards (not the Δ card) show the metric as secondary text
                // so it's always clear what's being shown even when scrolling past the heading
                if (card.sublabel) {
                    lblDiv.title = `${card.label} — ${card.sublabel}`;
                }
                lblDiv.textContent = card.label;
                cardDiv.appendChild(lblDiv);
                if (isSummary) {
                    createCardHistogram(cardDiv, card.data, card.label, card.color, card.type);
                } else {
                    const valEl = document.createElement('div');
                    valEl.className = 'metric-value';
                    const v = card.data;
                    if (v == null) { valEl.textContent = 'N/A'; valEl.style.color = '#94a3b8'; }
                    else if (card.type === 'frequency') valEl.textContent = formatFrequency(v);
                    else valEl.textContent = formatCount(v);
                    cardDiv.appendChild(valEl);
                }
            }
        }
    }

    function setupRepeatComparisonUI(plot) {
        const selContainer = document.getElementById('scan-comparison-selector');
        const cardContainer = document.getElementById('scan-comparison-cards');
        if (!selContainer || !cardContainer) return;
        selContainer.innerHTML = '';
        cardContainer.innerHTML = '';
        const entities = buildRepeatEntities(plot);
        if (entities.length < 2) return;
        const { wrapper, notify } = buildTwoDropdownSelector(entities, (eA, eB) => {
            renderPairwiseCards(buildPairwiseRows(eA, eB), cardContainer, false);
        });
        selContainer.appendChild(wrapper);
        notify();
    }

    function createCardHistogram(cardContainer, data, name, color, unitType = null) {
        if (!data || data.length === 0) {
            const empty = document.createElement('div');
            empty.style.cssText = 'height:140px;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:0.9em;font-weight:500;';
            empty.textContent = 'N/A';
            cardContainer.appendChild(empty);
            return;
        }
        const plotDiv = document.createElement('div');
        plotDiv.style.height = '140px';
        plotDiv.style.width = '100%';
        cardContainer.appendChild(plotDiv);
        let scaledData = data;
        let unit = '';
        if (unitType === 'frequency') {
            let maxAbs = 0;
            for (let i = 0; i < data.length; i++) { const a = Math.abs(data[i]); if (a > maxAbs) maxAbs = a; }
            let factor = 1;
            unit = 'Hz';
            if (maxAbs >= 1e9) { factor = 1e9; unit = 'GHz'; }
            else if (maxAbs >= 1e6) { factor = 1e6; unit = 'MHz'; }
            else if (maxAbs >= 1e3) { factor = 1e3; unit = 'kHz'; }
            scaledData = data.map(v => v / factor);
        } else if (unitType === 'steps') { unit = 'steps'; }
        else if (unitType === 'measurements') { unit = 'meas.'; }
        Plotly.newPlot(plotDiv, [{
            x: scaledData, type: 'histogram', name,
            marker: { color, line: { color: 'rgba(255,255,255,0.6)', width: 0.5 } },
            opacity: 0.85, autobinx: true,
        }], {
            margin: { l: 25, r: 10, t: 10, b: 25 },
            xaxis: { title: { text: unit, font: { size: 9, color: '#64748b', family: 'system-ui, sans-serif' } }, tickfont: { size: 8, color: '#64748b' }, showgrid: true, gridcolor: '#f1f5f9', zeroline: true, zerolinecolor: '#cbd5e1' },
            yaxis: { tickfont: { size: 8, color: '#64748b' }, showgrid: true, gridcolor: '#f1f5f9' },
            showlegend: false, plot_bgcolor: 'transparent', paper_bgcolor: 'transparent',
            bargap: 0.05, dragmode: false, hovermode: 'x',
        }, { displayModeBar: false, responsive: true });
    }

    function renderRepeatsSummary(generator, noise /*, strategy — unused: comparison is now across all strategies */) {
        const selContainer = document.getElementById('summary-comparison-selector');
        const container = document.getElementById('summary-subjects-container');
        const currentKey = `${generator}|${noise}`;
        if (currentKey === lastSummaryKey) {
            window.dispatchEvent(new Event('resize'));
            return;
        }
        lastSummaryKey = currentKey;
        if (selContainer) selContainer.innerHTML = '';
        if (container) container.innerHTML = '';

        const entities = buildSummaryEntities(generator, noise);
        if (entities.length < 2) {
            if (container) {
                const ph = document.createElement('div');
                ph.style.cssText = 'padding:3em 2em;color:#64748b;text-align:center;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:8px;margin:1em 0;';
                ph.innerHTML = '💡 <strong>Not enough data to compare.</strong><br><span style="font-size:0.9em;color:#94a3b8;display:block;margin-top:0.5em;">Run at least two strategies (or one strategy with its Sobol baseline) for this generator and noise configuration.</span>';
                container.appendChild(ph);
            }
            return;
        }

        ensurePlotly().then(() => {
            if (!selContainer || !container) return;

            const { wrapper: sumWrapper, notify: sumNotify } = buildTwoDropdownSelector(entities, (eA, eB) => {
                renderPairwiseCards(buildPairwiseRows(eA, eB), container, true);
            });
            selContainer.appendChild(sumWrapper);
            sumNotify();
        });
    }


    // Toggle setup
    const scanViewMode = document.getElementById('scan-view-mode');
    if (scanViewMode) {
        scanViewMode.addEventListener('click', (e) => {
            if (e.target.tagName === 'BUTTON') {
                scanViewMode.querySelectorAll('button').forEach(b => {
                    b.classList.remove('is-active');
                    b.setAttribute('aria-checked', 'false');
                    b.tabIndex = -1;
                });
                e.target.classList.add('is-active');
                e.target.setAttribute('aria-checked', 'true');
                e.target.tabIndex = 0;

                const mode = e.target.dataset.value;
                const repeatView = document.getElementById('scan-repeat-view');
                const summaryView = document.getElementById('scan-summary-view');

                if (mode === 'single') {
                    repeatView.style.display = 'block';
                    summaryView.style.display = 'none';
                } else {
                    repeatView.style.display = 'none';
                    summaryView.style.display = 'block';
                    renderRepeatsSummary(controlValue(scanGenerator), getEffectiveScanNoise(), controlValue(scanStrategy));
                }
            }
        });
        scanViewMode.addEventListener('keydown', (e) => {
            if (e.target.tagName !== 'BUTTON') return;
            const buttons = Array.from(scanViewMode.querySelectorAll('button'));
            const index = buttons.indexOf(e.target);
            let nextIndex = null;
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                nextIndex = (index + 1) % buttons.length;
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                nextIndex = (index - 1 + buttons.length) % buttons.length;
            }
            if (nextIndex !== null) {
                e.preventDefault();
                buttons[nextIndex].focus();
                buttons[nextIndex].click();
            }
        });
    }

    const summaryTabBar = document.getElementById('summary-tab-bar');
    if (summaryTabBar) {
        summaryTabBar.addEventListener('click', (e) => {
            if (e.target.tagName === 'BUTTON') {
                summaryTabBar.querySelectorAll('button').forEach(b => {
                    b.classList.remove('is-active');
                    b.setAttribute('aria-selected', 'false');
                    b.tabIndex = -1;
                    const panel = document.getElementById(b.dataset.tab);
                    if (panel) panel.classList.remove('is-active');
                });
                e.target.classList.add('is-active');
                e.target.setAttribute('aria-selected', 'true');
                e.target.tabIndex = 0;
                const activePanel = document.getElementById(e.target.dataset.tab);
                if (activePanel) activePanel.classList.add('is-active');

                // Trigger resize so Plotly fits correctly if it was hidden
                window.dispatchEvent(new Event('resize'));
            }
        });
        summaryTabBar.addEventListener('keydown', (e) => {
            if (e.target.tagName !== 'BUTTON') return;
            const buttons = Array.from(summaryTabBar.querySelectorAll('button'));
            const index = buttons.indexOf(e.target);
            let nextIndex = null;
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                nextIndex = (index + 1) % buttons.length;
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                nextIndex = (index - 1 + buttons.length) % buttons.length;
            }
            if (nextIndex !== null) {
                e.preventDefault();
                buttons[nextIndex].focus();
                buttons[nextIndex].click();
            }
        });
    }

    if (scanIframe) {
        scanIframe.addEventListener('load', () => {
            // Keep legend preference when switching locator strategy/repeat (new iframe src).
            applyMeasurementDistributionPreferenceInScanIframe();
            bindScanIframeLegendPreferenceSync();
            // Parse and render narrowed param bounds from the scan figure meta.
            renderNarrowedBoundsFromIframe();
        });
    }

    const narrowedBoundsPanel = document.getElementById('narrowed-bounds-panel');

    function formatBoundValue(v) {
        if (typeof v !== 'number' || !Number.isFinite(v)) return '?';
        if (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) return v.toExponential(2);
        return v.toPrecision(4);
    }

    function isFrequencyVariable(name) {
        if (typeof name !== 'string') return false;
        const n = name.toLowerCase();
        return n.includes('frequency') || n.includes('linewidth') || n.includes('split') || n.includes('span');
    }

    function formatHzValue(name, v) {
        if (typeof v !== 'number' || !Number.isFinite(v)) return '?';
        if (!isFrequencyVariable(name)) {
            return formatBoundValue(v);
        }
        let unit = ' Hz';
        let factor = 1;
        const absV = Math.abs(v);
        if (absV >= 1e9) {
            unit = ' GHz';
            factor = 1e9;
        } else if (absV >= 1e6) {
            unit = ' MHz';
            factor = 1e6;
        } else if (absV >= 1e3) {
            unit = ' kHz';
            factor = 1e3;
        }
        const scaled = v / factor;
        return parseFloat(scaled.toFixed(3)) + unit;
    }

    function renderNarrowedBoundsPanel(narrowedBounds) {
        if (!narrowedBoundsPanel) return;
        if (!narrowedBounds || typeof narrowedBounds !== 'object' || Object.keys(narrowedBounds).length === 0) {
            narrowedBoundsPanel.hidden = true;
            narrowedBoundsPanel.innerHTML = '';
            return;
        }
        const entries = Object.entries(narrowedBounds)
            .filter(([, range]) => Array.isArray(range) && range.length === 2)
            .map(([name, [lo, hi]]) => ({ name, lo, hi }));
        if (entries.length === 0) {
            narrowedBoundsPanel.hidden = true;
            narrowedBoundsPanel.innerHTML = '';
            return;
        }

        let cardsHtml = '';
        for (const { name, lo, hi } of entries) {
            let selectedVal = undefined;
            if (currentPlot && currentPlot.true_params && currentPlot.true_params.params) {
                const params = currentPlot.true_params.params;
                if (params[name] !== undefined) {
                    selectedVal = params[name];
                } else {
                    const lowerName = name.toLowerCase();
                    const foundKey = Object.keys(params).find(k => k.toLowerCase() === lowerName);
                    if (foundKey) {
                        selectedVal = params[foundKey];
                    }
                }
            }
            if (selectedVal === undefined || typeof selectedVal !== 'number') {
                selectedVal = (lo + hi) / 2;
            }

            let percent = 50;
            if (hi > lo) {
                percent = Math.min(100, Math.max(0, (selectedVal - lo) / (hi - lo) * 100));
            }

            const fmtLo = formatHzValue(name, lo);
            const fmtHi = formatHzValue(name, hi);
            const fmtVal = formatHzValue(name, selectedVal);

            cardsHtml +=
                '<div class="param-slider-card" title="' + name + ': [' + lo + ', ' + hi + '] — Selected: ' + selectedVal + '">' +
                '<div class="param-slider-label">🔍 ' + escapeHtml(name) + '</div>' +
                '<div class="param-slider-wrapper">' +
                '<div class="param-slider-track-bg"></div>' +
                '<div class="param-slider-track-fill" style="width: ' + percent + '%"></div>' +
                '<div class="param-slider-handle" style="left: ' + percent + '%"></div>' +
                '<div class="param-slider-value" style="left: ' + percent + '%">' + fmtVal + '</div>' +
                '</div>' +
                '<div class="param-slider-bounds">' +
                '<span class="param-slider-bound-lo">' + fmtLo + '</span>' +
                '<span class="param-slider-bound-hi">' + fmtHi + '</span>' +
                '</div>' +
                '</div>';
        }

        narrowedBoundsPanel.innerHTML =
            '<div class="param-bounds-header">Sweep-narrowed priors:</div>' +
            '<div class="param-bounds-grid">' + cardsHtml + '</div>';
        narrowedBoundsPanel.hidden = false;
    }

    function renderNarrowedBoundsFromIframe() {
        if (!scanIframe || !narrowedBoundsPanel) return;
        const frameDoc = scanIframe.contentDocument;
        if (!frameDoc) { renderNarrowedBoundsPanel(null); return; }
        const html = frameDoc.documentElement ? frameDoc.documentElement.outerHTML : '';
        if (!html) { renderNarrowedBoundsPanel(null); return; }
        try {
            const m = html.match(/Plotly\.newPlot\(\s*"[^"]+",\s*/);
            if (!m) { renderNarrowedBoundsPanel(null); return; }
            const pos = m.index + m[0].length;
            const dataStr = html.slice(pos);
            let depth = 0, end = 0;
            for (let i = 0; i < dataStr.length; i++) {
                if (dataStr[i] === '[') depth++;
                else if (dataStr[i] === ']') {
                    depth--;
                    if (depth === 0) { end = i + 1; break; }
                }
            }
            let layoutStart = pos + end;
            while (layoutStart < html.length && /[\s,]/.test(html[layoutStart])) layoutStart++;
            if (html[layoutStart] !== '{') { renderNarrowedBoundsPanel(null); return; }
            let ldepth = 0, lend = 0;
            for (let i = layoutStart; i < html.length; i++) {
                if (html[i] === '{') ldepth++;
                else if (html[i] === '}') {
                    ldepth--;
                    if (ldepth === 0) { lend = i + 1; break; }
                }
            }
            if (lend <= layoutStart) { renderNarrowedBoundsPanel(null); return; }
            const layout = JSON.parse(html.slice(layoutStart, lend));
            const meta = layout && layout.meta;
            renderNarrowedBoundsPanel(meta && meta.narrowed_param_bounds);
        } catch (e) {
            renderNarrowedBoundsPanel(null);
        }
    }

    function setupTabs() {
        const tabBar = document.querySelector('.tab-bar');
        const tabPanels = document.querySelectorAll('.tab-panel');

        tabBar.style.display = 'flex';
        tabBar.innerHTML = ''; // Clear existing

        const hasScans = scanPlots.length > 0;

        if (hasScans) {
            const button = document.createElement('button');
            button.className = 'tab-button';
            button.textContent = 'Scan measurements';
            button.dataset.tab = 'scan-section';
            button.setAttribute('role', 'tab');
            button.setAttribute('id', 'tab-scan');
            button.setAttribute('aria-controls', 'scan-section');
            button.setAttribute('aria-selected', 'false');
            tabBar.appendChild(button);
        }

        if (hasScans) {
            const button = document.createElement('button');
            button.className = 'tab-button';
            button.textContent = 'Head to head';
            button.dataset.tab = 'scan-comparison-section';
            button.setAttribute('role', 'tab');
            button.setAttribute('id', 'tab-scan-comparison');
            button.setAttribute('aria-controls', 'scan-comparison-section');
            button.setAttribute('aria-selected', 'false');
            tabBar.appendChild(button);
        }


        const tabButtons = Array.from(tabBar.querySelectorAll('.tab-button'));
        if (tabButtons.length > 0) {
            tabButtons[0].classList.add('is-active');
            tabButtons[0].setAttribute('aria-selected', 'true');
            tabButtons[0].tabIndex = 0;
            for (let i = 1; i < tabButtons.length; i++) {
                tabButtons[i].tabIndex = -1;
            }
            const initialTabId = tabButtons[0].dataset.tab;
            tabPanels.forEach(panel => {
                if (panel.id === initialTabId) {
                    panel.classList.remove('is-hidden');
                } else {
                    panel.classList.add('is-hidden');
                }
            });
        } else {
            tabBar.style.display = 'none';
        }

        tabBar.addEventListener('click', (e) => {
            const target = e.target;
            if (!target.matches('.tab-button')) {
                return;
            }

            tabButtons.forEach(button => {
                button.classList.remove('is-active');
                button.setAttribute('aria-selected', 'false');
                button.tabIndex = -1;
            });
            target.classList.add('is-active');
            target.setAttribute('aria-selected', 'true');
            target.tabIndex = 0;

            tabPanels.forEach(panel => {
                if (panel.id === target.dataset.tab) {
                    panel.classList.remove('is-hidden');
                } else {
                    panel.classList.add('is-hidden');
                }
            });
        });

        tabBar.addEventListener('keydown', (e) => {
            const target = e.target;
            if (!target.matches('.tab-button')) {
                return;
            }
            const index = tabButtons.indexOf(target);
            let nextIndex = null;
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                nextIndex = (index + 1) % tabButtons.length;
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                nextIndex = (index - 1 + tabButtons.length) % tabButtons.length;
            }
            if (nextIndex !== null) {
                e.preventDefault();
                tabButtons[nextIndex].focus();
                tabButtons[nextIndex].click();
            }
        });
    }

    // --- Strategy metrics (model_comparison bar charts) ---
    const allAggregatePlots = plots.filter(p => p.type === 'model_comparison' || p.type === 'milestone');

    function updateCompPlots() {
        const compIframeAbsErr = document.getElementById('comp-iframe-abs-err');
        const compIframeMeasurements = document.getElementById('comp-iframe-measurements');
        const compIframeDuration = document.getElementById('comp-iframe-duration');
        if (!compIframeAbsErr || !compIframeMeasurements || !compIframeDuration) return;

        const scanGeneratorEl = document.getElementById('scan-generator');
        const scanNoiseEl = document.getElementById('scan-noise');
        if (!scanGeneratorEl || !scanNoiseEl) return;
        const gen = controlValue(scanGeneratorEl);
        const noise = controlValue(scanNoiseEl);

        if (!gen || !noise) {
            compIframeAbsErr.src = '';
            compIframeMeasurements.src = '';
            compIframeDuration.src = '';
            const compIframeSavings = document.getElementById('comp-iframe-savings');
            if (compIframeSavings) compIframeSavings.src = '';
            const mIframeSteps = document.getElementById('milestone-iframe-steps');
            const mIframeErrFc = document.getElementById('milestone-iframe-err-fc');
            const mIframeDeltaFc = document.getElementById('milestone-iframe-delta-fc');
            if (mIframeSteps) mIframeSteps.src = '';
            if (mIframeErrFc) mIframeErrFc.src = '';
            if (mIframeDeltaFc) mIframeDeltaFc.src = '';
            const spanPerNoiseIframe = document.getElementById('comp-iframe-span-per-noise');
            if (spanPerNoiseIframe) spanPerNoiseIframe.src = '';
            return;
        }

        const absErrPlot = allAggregatePlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'abs_err_x');
        const measurementsPlot = allAggregatePlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'measurements');
        const durationPlot = allAggregatePlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'duration_ms');
        const savingsPlot = allAggregatePlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'sobol_difference');
        const spanPerNoisePlot = allAggregatePlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'savings_vs_span_per_noise');

        compIframeAbsErr.src = absErrPlot ? absErrPlot.path : '';
        compIframeMeasurements.src = measurementsPlot ? measurementsPlot.path : '';
        compIframeDuration.src = durationPlot ? durationPlot.path : '';

        const compIframeSavings = document.getElementById('comp-iframe-savings');
        const compSavingsContainer = document.getElementById('comp-savings-container');
        if (compIframeSavings) {
            compIframeSavings.src = savingsPlot ? savingsPlot.path : '';
            if (compSavingsContainer) {
                compSavingsContainer.style.display = savingsPlot ? 'block' : 'none';
            }
        }

        const spanPerNoiseIframe = document.getElementById('comp-iframe-span-per-noise');
        const spanPerNoiseContainer = document.getElementById('comp-span-per-noise-container');
        if (spanPerNoiseIframe) {
            spanPerNoiseIframe.src = spanPerNoisePlot ? spanPerNoisePlot.path : '';
            if (spanPerNoiseContainer) {
                spanPerNoiseContainer.style.display = spanPerNoisePlot ? 'block' : 'none';
            }
        }

        // Update Trends Across Configurations (Summary plots)
        const summaryErrIframe = document.getElementById('comp-iframe-summary-err');
        const summaryMeasIframe = document.getElementById('comp-iframe-summary-meas');
        const summarySavingsIframe = document.getElementById('comp-iframe-summary-savings');
        const summarySpanIframe = document.getElementById('comp-iframe-summary-span');
        
        if (summaryErrIframe && summaryMeasIframe && summarySavingsIframe && summarySpanIframe) {
            const summaryPlots = plots.filter(p => p.type === 'summary' && p.generator === gen);
            const errPlot = summaryPlots.find(p => p.metric === 'pair_rmse') || summaryPlots.find(p => p.metric === 'abs_err_x');
            const measPlot = summaryPlots.find(p => p.metric === 'measurements');
            const savPlot = summaryPlots.find(p => p.metric === 'savings');
            const spanPlot = summaryPlots.find(p => p.metric === 'savings_vs_span');
            
            summaryErrIframe.src = errPlot ? errPlot.path : '';
            summaryMeasIframe.src = measPlot ? measPlot.path : '';
            summarySavingsIframe.src = savPlot ? savPlot.path : '';
            summarySpanIframe.src = spanPlot ? spanPlot.path : '';
        }

        // Update milestone plots
        const milestonePlots = plots.filter(p => p.type === 'milestone');
        const stepsPlot = milestonePlots.find(p => p.generator === gen && p.noise === noise && p.path.includes('steps_to_fb'));
        const errFcPlot = milestonePlots.find(p => p.generator === gen && p.noise === noise && p.path.includes('error_comparison_fc'));
        const deltaFcPlot = milestonePlots.find(p => p.generator === gen && p.noise === noise && p.path.includes('error_delta_fc'));

        const mIframeSteps = document.getElementById('milestone-iframe-steps');
        const mIframeErrFc = document.getElementById('milestone-iframe-err-fc');
        const mIframeDeltaFc = document.getElementById('milestone-iframe-delta-fc');
        const mContainer = document.getElementById('milestone-plots-container');

        if (mIframeSteps) mIframeSteps.src = stepsPlot ? stepsPlot.path : '';
        if (mIframeErrFc) mIframeErrFc.src = errFcPlot ? errFcPlot.path : '';
        if (mIframeDeltaFc) mIframeDeltaFc.src = deltaFcPlot ? deltaFcPlot.path : '';

        if (mContainer) {
            mContainer.style.display = (stepsPlot || errFcPlot || deltaFcPlot) ? 'flex' : 'none';
        }
    }

    const toggleNoiseMetricsBtn = document.getElementById('toggle-noise-metrics-btn');
    const noiseMetricsView = document.getElementById('noise-metrics-view');
    if (toggleNoiseMetricsBtn && noiseMetricsView) {
        toggleNoiseMetricsBtn.addEventListener('click', () => {
            if (noiseMetricsView.hidden) {
                noiseMetricsView.hidden = false;
                updateCompPlots();
            } else {
                noiseMetricsView.hidden = true;
            }
        });
    }

    // Call updateCompPlots when scan controls change, if panel is visible
    const scanGeneratorEl = document.getElementById('scan-generator');
    const scanNoiseEl = document.getElementById('scan-noise');
    if (scanGeneratorEl) {
        scanGeneratorEl.addEventListener('controlchange', () => {
            if (noiseMetricsView && !noiseMetricsView.hidden) {
                updateCompPlots();
            }
        });
    }
    if (scanNoiseEl) {
        scanNoiseEl.addEventListener('controlchange', () => {
            if (noiseMetricsView && !noiseMetricsView.hidden) {
                updateCompPlots();
            }
        });
    }

    // --- Scan Comparison: same signal (generator / noise / repeat), two locators ---
    function setupScanComparison() {
        const cmpGen = document.getElementById('cmp-shared-generator');
        const cmpNoise = document.getElementById('cmp-shared-noise');
        const cmpRepeat = document.getElementById('cmp-shared-repeat');
        const leftStrat = document.getElementById('left-strategy');
        const rightStrat = document.getElementById('right-strategy');
        const headToHeadEl = document.getElementById('head-to-head-plot');
        const leftMetrics = document.getElementById('left-metrics');
        const rightMetrics = document.getElementById('right-metrics');

        if (
            !cmpGen ||
            !cmpNoise ||
            !cmpRepeat ||
            !leftStrat ||
            !rightStrat ||
            !headToHeadEl
        ) {
            return;
        }

        function updateCmpSharedSignalControls() {
            const genItems = [...new Set(scanPlots.map((p) => p.generator))].sort();
            const selGen = renderSegmentedControl(cmpGen, genItems, controlValue(cmpGen));
            const noiseItems = scanPlots
                .filter((p) => p.generator === selGen)
                .map((p) => p.noise);
            renderSegmentedControl(cmpNoise, noiseItems, controlValue(cmpNoise));
        }

        function updateCmpStrategyControls() {
            const selGen = controlValue(cmpGen);
            const selNoise = controlValue(cmpNoise);
            const availableFromPlots = new Set(
                scanPlots
                    .filter((p) => p.generator === selGen && p.noise === selNoise)
                    .map((p) => p.strategy),
            );
            const gridStrategies = (window.STRATEGY_GRID && window.STRATEGY_GRID[selGen]) || [];
            const stratItems = [...new Set([...gridStrategies, ...availableFromPlots])];
            const disabledItems = new Set(
                stratItems.filter((strategy) => !availableFromPlots.has(strategy)),
            );
            const opts = { disabledItems };
            renderSegmentedControl(leftStrat, stratItems, controlValue(leftStrat), opts);
            renderSegmentedControl(rightStrat, stratItems, controlValue(rightStrat), opts);
        }

        function repeatStringsFor(g, n, strat) {
            return scanPlots
                .filter(
                    (p) =>
                        p.generator === g &&
                        p.noise === n &&
                        p.strategy === strat,
                )
                .map((p) => String(p.repeat ?? p.attempt ?? 1));
        }

        function updateCmpRepeatControl() {
            const g = controlValue(cmpGen);
            const n = controlValue(cmpNoise);
            const sl = controlValue(leftStrat);
            const sr = controlValue(rightStrat);
            const repsL = repeatStringsFor(g, n, sl);
            const repsR = repeatStringsFor(g, n, sr);
            const setR = new Set(repsR);
            let common = [...new Set(repsL)].filter((r) => setR.has(r));
            common.sort((a, b) => Number(a) - Number(b));
            if (common.length === 0) {
                common = [...new Set([...repsL, ...repsR])].sort((a, b) => Number(a) - Number(b));
            }
            const { value: selRep } = renderSelectControl(
                cmpRepeat,
                common,
                controlValue(cmpRepeat) || cmpRepeat.dataset.value || '',
            );
            if (selRep) {
                cmpRepeat.dataset.value = selRep;
            }
        }

        function updateAllCmpControls() {
            updateCmpSharedSignalControls();
            updateCmpStrategyControls();
            updateCmpRepeatControl();
        }

        function applyMetrics(el, plot) {
            if (!el) {
                return;
            }
            if (plot) {
                const absErr = formatFrequency(plot.abs_err_x);
                const uncertainty = formatFrequency(plot.uncert);
                const measurements = formatCount(plot.measurements);
                const duration = formatDuration(plot.duration_ms);
                let text = `Measurements: ${measurements} • Duration: ${duration} • Abs Error: ${absErr} • Uncertainty: ${uncertainty}`;
                const expUniform = plot.metrics && plot.metrics.expected_uniform_points;
                if (expUniform != null && Number.isFinite(expUniform)) {
                    text += ` • Exp. uniform: ${formatCount(expUniform)}`;
                }
                const sobolBaseline = plot.sobol_baseline_steps;
                if (sobolBaseline != null && Number.isFinite(sobolBaseline)) {
                    text += ` • Sobol baseline: ${formatCount(sobolBaseline)}`;
                }
                const sobolFreqBaseline = plot.sobol_freq_steps;
                if (sobolFreqBaseline != null && Number.isFinite(sobolFreqBaseline)) {
                    text += ` • Sobol freq baseline: ${formatCount(sobolFreqBaseline)}`;
                }
                if (sobolBaseline != null && sobolFreqBaseline != null && Number.isFinite(sobolBaseline) && Number.isFinite(sobolFreqBaseline)) {
                    const diffVal = plot.sobol_conv_diff != null ? plot.sobol_conv_diff : (sobolBaseline - sobolFreqBaseline);
                    text += ` • Sobol diff: ${formatCount(diffVal)}`;
                }
                el.textContent = text;
            } else {
                el.textContent = '';
            }
        }

        function clearHeadToHeadPlot(message) {
            if (window.Plotly) {
                try {
                    window.Plotly.purge(headToHeadEl);
                } catch (e) {
                    /* ignore */
                }
            }
            headToHeadEl.innerHTML = message
                ? `<p class="metrics">${message}</p>`
                : '';
        }

        async function updateCmpPlots() {
            const vGen = controlValue(cmpGen);
            const vNoise = controlValue(cmpNoise);
            const vStratL = controlValue(leftStrat);
            const vStratR = controlValue(rightStrat);
            const repStr = controlValue(cmpRepeat);
            const vRep = repStr ? parseInt(repStr, 10) : NaN;

            if (!vGen || !vNoise || !vStratL || !vStratR || !Number.isFinite(vRep)) {
                clearHeadToHeadPlot('');
                applyMetrics(leftMetrics, null);
                applyMetrics(rightMetrics, null);
                return;
            }

            const plotL = scanPlots.find(
                (p) =>
                    p.generator === vGen &&
                    p.noise === vNoise &&
                    p.strategy === vStratL &&
                    p.repeat === vRep,
            );
            const plotR = scanPlots.find(
                (p) =>
                    p.generator === vGen &&
                    p.noise === vNoise &&
                    p.strategy === vStratR &&
                    p.repeat === vRep,
            );

            applyMetrics(leftMetrics, plotL);
            applyMetrics(rightMetrics, plotR);

            if (!plotL || !plotR) {
                clearHeadToHeadPlot('No scan data for this selection.');
                return;
            }

            // Load plot data on-demand from scan HTML files
            const [pdL, pdR] = await Promise.all([
                plotL.plot_data ? Promise.resolve(plotL.plot_data) : loadPlotDataFromScanHtml(plotL),
                plotR.plot_data ? Promise.resolve(plotR.plot_data) : loadPlotDataFromScanHtml(plotR)
            ]);

            if (!pdL || !pdR || !pdL.x_dense || !pdR.x_dense) {
                clearHeadToHeadPlot(
                    'Could not load plot data from scan files.',
                );
                return;
            }

            try {
                await ensurePlotly();
                if (window.Plotly) {
                    try {
                        window.Plotly.purge(headToHeadEl);
                    } catch (e) {
                        /* ignore */
                    }
                }
                headToHeadEl.innerHTML = '';
                const traces = buildHeadToHeadTraces(pdL, pdR, vStratL, vStratR);
                const focusShapes = buildHeadToHeadFocusShapes(pdL, pdR);
                const layout = {
                    title: 'Head to head: same signal, two strategies',
                    template: 'plotly_white',
                    xaxis: { title: 'frequency' },
                    yaxis: { title: 'intensity (photon count)' },
                    legend: {
                        orientation: 'h',
                        yanchor: 'top',
                        y: -0.2,
                        xanchor: 'center',
                        x: 0.5,
                    },
                    margin: { t: 48, b: 120, l: 56, r: 24 },
                    shapes: focusShapes,
                };
                await window.Plotly.react(headToHeadEl, traces, layout, { responsive: true });
            } catch (err) {
                console.error(err);
                clearHeadToHeadPlot('Could not render combined plot (Plotly failed to load or draw).');
            }
        }

        function onSharedChange() {
            updateCmpSharedSignalControls();
            updateCmpRepeatControl();
            updateCmpPlots();
        }

        cmpGen.addEventListener('controlchange', onSharedChange);
        cmpNoise.addEventListener('controlchange', onSharedChange);
        leftStrat.addEventListener('controlchange', () => {
            updateCmpRepeatControl();
            updateCmpPlots();
        });
        rightStrat.addEventListener('controlchange', () => {
            updateCmpRepeatControl();
            updateCmpPlots();
        });
        cmpRepeat.addEventListener('change', () => {
            cmpRepeat.dataset.value = cmpRepeat.value || '';
            updateCmpPlots();
        });

        if (scanDefault) {
            cmpGen.dataset.value = scanDefault.generator ?? '';
            cmpNoise.dataset.value = scanDefault.noise ?? '';
            cmpRepeat.dataset.value =
                scanDefault.repeat === undefined ? '' : String(scanDefault.repeat);
            const strats = [
                ...new Set(scanPlots.filter((p) => p.generator === scanDefault.generator && p.noise === scanDefault.noise).map((p) => p.strategy)),
            ].sort();
            if (strats.length >= 2) {
                leftStrat.dataset.value = strats[0];
                rightStrat.dataset.value = strats[1];
            } else if (strats.length === 1) {
                leftStrat.dataset.value = strats[0];
                rightStrat.dataset.value = strats[0];
            }
        }

        updateAllCmpControls();
        updateCmpPlots();
    }

    if (document.getElementById('scan-comparison-section')) {
        setupScanComparison();
    }

    scanGenerator.addEventListener('controlchange', () => {
        updateAllScanControls();
        findAndDisplayPlot();
    });
    scanNoise.addEventListener('controlchange', () => {
        updateGaussStdSlider();
        updateScanStrategyControl();
        updateScanRepeatControl();
        findAndDisplayPlot();
    });
    scanStrategy.addEventListener('controlchange', () => {
        updateScanRepeatControl();
        findAndDisplayPlot();
    });
    scanRepeat.addEventListener('change', () => {
        scanRepeat.dataset.value = scanRepeat.value || '';
        updateRepeatNavButtons();
        findAndDisplayPlot();
    });

    if (gaussStdSlider) {
        gaussStdSlider.addEventListener('input', (e) => {
            const idx = parseInt(e.target.value, 10);
            gaussStdSlider.dataset.index = idx;
            if (currentGaussSigmas[idx] !== undefined) {
                gaussStdValue.textContent = currentGaussSigmas[idx].toFixed(4).replace(/\.?0+$/, '');
                gaussStdPrev.disabled = idx === 0;
                gaussStdNext.disabled = idx === currentGaussSigmas.length - 1;
            }
            updateScanStrategyControl();
            updateScanRepeatControl();
            findAndDisplayPlot();
        });
    }

    if (gaussStdPrev) {
        gaussStdPrev.addEventListener('click', () => {
            const currentIdx = parseInt(gaussStdSlider.value, 10);
            if (currentIdx > 0) {
                const nextIdx = currentIdx - 1;
                gaussStdSlider.value = nextIdx;
                gaussStdSlider.dataset.index = nextIdx;
                gaussStdValue.textContent = currentGaussSigmas[nextIdx].toFixed(4).replace(/\.?0+$/, '');
                gaussStdPrev.disabled = nextIdx === 0;
                gaussStdNext.disabled = nextIdx === currentGaussSigmas.length - 1;
                
                updateScanStrategyControl();
                updateScanRepeatControl();
                findAndDisplayPlot();
            }
        });
    }

    if (gaussStdNext) {
        gaussStdNext.addEventListener('click', () => {
            const currentIdx = parseInt(gaussStdSlider.value, 10);
            if (currentIdx < currentGaussSigmas.length - 1) {
                const nextIdx = currentIdx + 1;
                gaussStdSlider.value = nextIdx;
                gaussStdSlider.dataset.index = nextIdx;
                gaussStdValue.textContent = currentGaussSigmas[nextIdx].toFixed(4).replace(/\.?0+$/, '');
                gaussStdPrev.disabled = nextIdx === 0;
                gaussStdNext.disabled = nextIdx === currentGaussSigmas.length - 1;
                
                updateScanStrategyControl();
                updateScanRepeatControl();
                findAndDisplayPlot();
            }
        });
    }

    if (scanRepeatPrev) {
        scanRepeatPrev.addEventListener('click', () => {
            selectRepeatByOffset(-1);
        });
    }
    if (scanRepeatNext) {
        scanRepeatNext.addEventListener('click', () => {
            selectRepeatByOffset(1);
        });
    }

    if (scanDefault) {
        scanGenerator.dataset.value = scanDefault.generator ?? '';
        
        const defaultNoise = scanDefault.noise ?? '';
        if (defaultNoise.includes('Gauss')) {
            scanNoise.dataset.value = 'Gauss';
        } else {
            scanNoise.dataset.value = defaultNoise;
        }

        scanStrategy.dataset.value = scanDefault.strategy ?? '';
        scanRepeat.dataset.value =
            scanDefault.repeat === undefined || scanDefault.repeat === null
                ? ''
                : String(scanDefault.repeat);
    }

    try {
        setupTabs();
        updateAllScanControls();
        updateCompPlots();
        findAndDisplayPlot();
    } catch (error) {
        console.error('Error initializing UI controls:', error);
        // Show an error message to the user
        const errorDiv = document.createElement('div');
        errorDiv.setAttribute('role', 'alert');
        errorDiv.style.padding = '20px';
        errorDiv.style.margin = '20px';
        errorDiv.style.border = '1px solid #f5c6cb';
        errorDiv.style.backgroundColor = '#f8d7da';
        errorDiv.style.color = '#721c24';
        errorDiv.style.borderRadius = '4px';
        errorDiv.innerHTML = '<h3>Error: Failed to initialize UI</h3><p>There was a problem initializing the user interface. Please check the console for more details.</p>';
        document.body.appendChild(errorDiv);
    }
}

// Keyboard shortcut: 'r' to reload/recalculate results
let _reloadInProgress = false;

async function triggerReload() {
    if (_reloadInProgress) {
        console.log('Reload already in progress...');
        return;
    }
    _reloadInProgress = true;
    console.log('Reloading results...');
    // Show notification
    const notif = document.createElement('div');
    notif.setAttribute('role', 'status');
    notif.setAttribute('aria-live', 'polite');
    notif.id = 'reload-notification';
    notif.style.cssText = 'position:fixed;top:20px;right:20px;padding:15px 25px;background:#2196F3;color:white;border-radius:4px;z-index:9999;font-family:sans-serif;font-weight:bold;box-shadow:0 2px 10px rgba(0,0,0,0.3);';
    notif.textContent = 'Reloading results...';
    document.body.appendChild(notif);

    try {
        const response = await fetch('/api/reload', { method: 'POST' });
        const data = await response.json();
        console.log('Reload response:', data);

        if (data.status === 'started') {
            notif.style.background = '#4CAF50';
            notif.textContent = 'Recalculating... (this may take a moment)';
            // Poll for completion
            const pollInterval = setInterval(async () => {
                try {
                    const statusResp = await fetch('/api/status');
                    const status = await statusResp.json();
                    if (!status.reload_running) {
                        clearInterval(pollInterval);
                        notif.style.background = '#4CAF50';
                        notif.textContent = 'Done! Reloading page...';
                        setTimeout(() => window.location.reload(), 1000);
                    }
                } catch (e) {
                    console.error('Poll error:', e);
                }
            }, 1000);
        } else if (data.status === 'already_running') {
            notif.style.background = '#FF9800';
            notif.textContent = 'Reload already in progress';
            setTimeout(() => notif.remove(), 3000);
            _reloadInProgress = false;
        }
    } catch (error) {
        console.error('Reload failed:', error);
        notif.style.background = '#f44336';
        notif.textContent = 'Reload failed (see console)';
        setTimeout(() => notif.remove(), 5000);
        _reloadInProgress = false;
    }
}

document.addEventListener('keydown', async (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        return;
    }
    if (e.key === 'r' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        triggerReload();
    }
});

const recalcBtn = document.getElementById('recalc-btn');
if (recalcBtn) {
    recalcBtn.addEventListener('click', triggerReload);
}

const init = () => {
    const lastRunEl = document.getElementById('last-run-time');
    if (lastRunEl) {
        lastRunEl.textContent = 'Last run: ' + new Date().toLocaleString();
    }
    window.NVISION_BOOTSTRAP
        .then(() => {
            main();
        })
        .catch((error) => {
            console.error('Failed to initialize UI assets:', error);
            const errorDiv = document.createElement('div');
            errorDiv.setAttribute('role', 'alert');
            errorDiv.style.padding = '20px';
            errorDiv.style.margin = '20px';
            errorDiv.style.border = '1px solid #f5c6cb';
            errorDiv.style.backgroundColor = '#f8d7da';
            errorDiv.style.color = '#721c24';
            errorDiv.style.borderRadius = '4px';
            errorDiv.innerHTML = '<h3>Error: Failed to initialize UI assets</h3><p>Could not load manifest/settings data files.</p>';
            document.body.appendChild(errorDiv);
        });
};

if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
