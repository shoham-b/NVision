window.NVISION_ASSET_PREFIX = '';
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

function main() {
        let plots = [];
        try {
            plots = window.MANIFEST;
            if (!Array.isArray(plots)) {
                throw new Error('Invalid manifest format');
            }
        } catch (error) {
            console.error('Error reading plots manifest from window.MANIFEST:', error);
            // Show an error message to the user
            const errorDiv = document.createElement('div');
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

        plots.forEach((p) => {
            const gen = p.generator || '';
            const strat = p.strategy || '';
            if (gen.startsWith('NVCenter-')) {
                p.generator_type = strat.includes('Bayesian') ? 'NV Center (Bayesian)' : 'NV Center';
            } else {
                p.generator_type = 'Supplemental';
            }
        });

        const scanPlots = plots.filter((p) => p.type === 'scan' && p.generator_type);
        const bayesSection = document.getElementById('bayes-section-container');
        const bayesImage = document.getElementById('bayes-image');
        const bayesPlots = plots.filter((p) => p.type === 'bayesian');
        const bayesInteractivePlots = plots.filter((p) => p.type === 'bayesian_interactive');
        const bayesInteractiveSection = document.getElementById('bayes-interactive-section');
        const bayesInteractiveIframe = document.getElementById('bayes-interactive-iframe');
        const bayesConvergenceSection = document.getElementById('bayes-convergence-section');
        const bayesConvergenceIframe = document.getElementById('bayes-convergence-iframe');
        const bayesStatsPlots = plots.filter((p) => p.type === 'bayesian_stats');
        const bayesStatsSection = document.getElementById('bayes-stats-section');
        const posteriorHistoryImage = document.getElementById('posterior-history-image');
        const convergenceStatsImage = document.getElementById('convergence-stats-image');
        const hasBayesArtifacts =
            bayesPlots.length > 0 ||
            bayesInteractivePlots.length > 0 ||
            bayesStatsPlots.length > 0 ||
            plots.some((p) => p.type === 'bayesian_parameter_convergence');
        if (bayesSection) {
            bayesSection.hidden = !hasBayesArtifacts;
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

        function updateBayesInteractiveView(selectedPlot) {
            if (!bayesInteractiveSection || !bayesInteractiveIframe) {
                return;
            }

            if (!selectedPlot) {
                bayesInteractiveSection.hidden = true;
                bayesInteractiveIframe.src = '';
                bayesConvergenceSection.hidden = true;
                bayesConvergenceIframe.src = '';
                return;
            }

            const interactivePlot = bayesInteractivePlots.find(
                (p) =>
                    p.generator === selectedPlot.generator &&
                    p.noise === selectedPlot.noise &&
                    p.strategy === selectedPlot.strategy &&
                    p.repeat === selectedPlot.repeat
            );

            if (interactivePlot) {
                bayesInteractiveIframe.src = interactivePlot.path;
                bayesInteractiveSection.hidden = false;
            } else {
                bayesInteractiveSection.hidden = true;
                bayesInteractiveIframe.src = '';
            }

            const convergencePlot = plots.find(
                (p) =>
                    p.type === 'bayesian_parameter_convergence' &&
                    p.generator === selectedPlot.generator &&
                    p.noise === selectedPlot.noise &&
                    p.strategy === selectedPlot.strategy &&
                    p.repeat === selectedPlot.repeat
            );

            if (convergencePlot) {
                bayesConvergenceIframe.src = convergencePlot.path;
                bayesConvergenceSection.hidden = false;
            } else {
                bayesConvergenceSection.hidden = true;
                bayesConvergenceIframe.src = '';
            }
        }

        function updateBayesStatsView(selectedPlot) {
            if (!bayesStatsSection || !posteriorHistoryImage || !convergenceStatsImage) {
                return;
            }

            if (!selectedPlot) {
                bayesStatsSection.hidden = true;
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
                bayesStatsSection.hidden = false;
            } else {
                bayesStatsSection.hidden = true;
            }
        }

        const nvCenterDefault = scanPlots.find(p => p.generator_type === 'NV Center (Bayesian)');
        const scanDefault = nvCenterDefault || (scanPlots.length > 0 ? scanPlots[0] : null);

        const scanGeneratorType = document.getElementById('scan-generator-type');
        const scanGenerator = document.getElementById('scan-generator');
        const scanNoise = document.getElementById('scan-noise');
        const scanStrategy = document.getElementById('scan-strategy');
        const scanRepeat = document.getElementById('scan-repeat');
        const scanRepeatPrev = document.getElementById('scan-repeat-prev');
        const scanRepeatNext = document.getElementById('scan-repeat-next');
        const scanIframe = document.getElementById('scan-iframe');
        const scanMetrics = document.getElementById('scan-metrics');
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

        function formatCount(value) {
            if (typeof value === 'number' && Number.isFinite(value)) {
                return Math.round(value).toString();
            }
            return 'N/A';
        }

        function formatDuration(value) {
            if (typeof value === 'number' && Number.isFinite(value)) {
                return value.toFixed(1) + ' ms';
            }
            return 'N/A';
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
            let coarse_x = [], coarse_y = [], fine_x = [], fine_y = [], fine_step = [];
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

            if (coarse_x.length || fine_x.length) {
                out.measurements = {
                    mode: 'phases',
                    coarse_x, coarse_y: coarse_y.map(y => y == null ? null : Number(y)),
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

        function setControlValue(control, value, {silent = false} = {}) {
            const normalized = value ?? '';
            control.dataset.value = normalized;
            const buttons = control.querySelectorAll('button');
            for (const button of buttons) {
                const isActive = button.dataset.value === normalized;
                button.classList.toggle('is-active', isActive);
                button.setAttribute('aria-checked', String(isActive));
            }
            if (!silent) {
                control.dispatchEvent(
                    new CustomEvent('controlchange', {
                        bubbles: false,
                        detail: {value: normalized},
                    }),
                );
            }
        }

        function renderSegmentedControl(control, items, previousValue) {
            const uniqueItems = [
                ...new Set(
                    items.filter((item) => item !== undefined && item !== null)
                ),
            ]
                .map((item) => String(item))
                .sort((a, b) => {
                    if (a === 'NV Center (Bayesian)') return -1;
                    if (b === 'NV Center (Bayesian)') return 1;
                    return a.localeCompare(b);
                });

            control.innerHTML = '';

            for (const item of uniqueItems) {
                const button = document.createElement('button');
                button.type = 'button';
                button.dataset.value = item;
                button.setAttribute('role', 'radio');
                button.setAttribute('aria-checked', 'false');
                button.textContent = item;
                button.addEventListener('click', () => {
                    setControlValue(control, item);
                });
                control.appendChild(button);
            }

            let nextValue = '';
            if (uniqueItems.length > 0) {
                if (previousValue && uniqueItems.includes(previousValue)) {
                    nextValue = previousValue;
                } else {
                    nextValue = uniqueItems[0];
                }
            }

            setControlValue(control, nextValue, {silent: true});
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
            return {value: nextValue, items: uniqueItems};
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

        /** Left column: target, generator, noise (signal / experiment path). */
        function updateScanSignalControls() {
            const selectedScanGeneratorType = renderSegmentedControl(
                scanGeneratorType,
                [...new Set(scanPlots.map((p) => p.generator_type))].sort(),
                controlValue(scanGeneratorType),
            );

            const scanGeneratorItems = scanPlots
                .filter((p) => p.generator_type === selectedScanGeneratorType)
                .map((p) => p.generator);
            const selectedScanGenerator = renderSegmentedControl(
                scanGenerator,
                scanGeneratorItems,
                controlValue(scanGenerator),
            );

            const scanNoiseItems = scanPlots
                .filter(
                    (p) =>
                        p.generator_type === selectedScanGeneratorType &&
                        p.generator === selectedScanGenerator
                )
                .map((p) => p.noise);
            renderSegmentedControl(
                scanNoise,
                scanNoiseItems,
                controlValue(scanNoise),
            );
        }

        /** Right column: strategies for this target type only — not filtered by generator or noise. */
        function updateScanStrategyControl() {
            const selectedScanGeneratorType = controlValue(scanGeneratorType);
            const scanStrategyItems = [
                ...new Set(
                    scanPlots
                        .filter((p) => p.generator_type === selectedScanGeneratorType)
                        .map((p) => p.strategy)
                ),
            ];
            renderSegmentedControl(
                scanStrategy,
                scanStrategyItems,
                controlValue(scanStrategy),
            );
        }

        /** Repeat options for the full (signal × locator) selection. */
        function updateScanRepeatControl() {
            const selectedScanGeneratorType = controlValue(scanGeneratorType);
            const selectedScanGenerator = controlValue(scanGenerator);
            const selectedScanNoise = controlValue(scanNoise);
            const selectedScanStrategy = controlValue(scanStrategy);

            const scanRepeatItems = scanPlots
                .filter(
                    (p) =>
                        p.generator_type === selectedScanGeneratorType &&
                        p.generator === selectedScanGenerator &&
                        p.noise === selectedScanNoise &&
                        p.strategy === selectedScanStrategy
                )
                .map((p) => String(p.repeat ?? p.attempt ?? 1));
            const {value: selectedRepeat, items: repeatItems} = renderSelectControl(
                scanRepeat,
                scanRepeatItems,
                controlValue(scanRepeat) || scanRepeat.dataset.value || '',
            );
            currentRepeatItems = repeatItems;
            if (selectedRepeat) {
                scanRepeat.dataset.value = selectedRepeat;
            }
            updateRepeatNavButtons();
        }

        function updateAllScanControls() {
            updateScanSignalControls();
            updateScanStrategyControl();
            updateScanRepeatControl();
        }

        function findAndDisplayPlot() {
            const scanGeneratorTypeValue = controlValue(scanGeneratorType);
            const scanGeneratorValue = controlValue(scanGenerator);
            const scanNoiseValue = controlValue(scanNoise);
            const scanStrategyValue = controlValue(scanStrategy);
            const scanRepeatValue = controlValue(scanRepeat);

            if (
                scanGeneratorTypeValue &&
                scanGeneratorValue &&
                scanNoiseValue &&
                scanStrategyValue &&
                scanRepeatValue
            ) {
                const repeatNumber = parseInt(scanRepeatValue, 10);
                const plot = scanPlots.find(
                    (p) =>
                        p.generator_type === scanGeneratorTypeValue &&
                        p.generator === scanGeneratorValue &&
                        p.noise === scanNoiseValue &&
                        p.strategy === scanStrategyValue &&
                        p.repeat === repeatNumber
                );
                scanIframe.src = plot ? plot.path : '';
                if (plot) {
                    const absErr = formatMetricValue(plot.abs_err_x);
                    const uncertainty = formatMetricValue(plot.uncert);
                    const measurements = formatCount(plot.measurements);
                    const duration = formatDuration(plot.duration_ms);
                    const repeatTotal = plot.repeat_total ?? null;
                    const attemptLabel = repeatTotal
                        ? 'Attempt ' + plot.repeat + ' of ' + repeatTotal
                        : 'Attempt ' + plot.repeat;
                    scanMetrics.textContent =
                        attemptLabel +
                        ' • Measurements: ' + measurements +
                        ' • Duration: ' + duration +
                        ' • Abs error: ' +
                        absErr +
                        ' • Uncertainty: ' +
                        uncertainty;
                    updateBayesView(plot);
                    updateBayesStatsView(plot);
                    updateBayesInteractiveView(plot);
                } else {
                    scanIframe.src = '';
                    scanMetrics.textContent = '';
                    updateBayesView(null);
                    updateBayesStatsView(null);
                    updateBayesInteractiveView(null);
                }
            } else {
                scanIframe.src = '';
                scanMetrics.textContent = '';
                scanMetrics.textContent = '';
                updateBayesView(null);
                updateBayesStatsView(null);
                updateBayesInteractiveView(null);
            }
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
            const chips = entries.map(({ name, lo, hi }) =>
                '<span class="param-bound-chip" title="' + name + ': [' + lo + ', ' + hi + ']">' +
                '\uD83D\uDD0D\u00A0<strong>' + name + '</strong>\u00A0[' + formatBoundValue(lo) + ',\u00A0' + formatBoundValue(hi) + ']</span>'
            ).join('');
            narrowedBoundsPanel.innerHTML =
                '<span class="param-bound-label">Sweep-narrowed priors:</span>' + chips;
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
                tabBar.appendChild(button);
            }

            if (hasScans) {
                const button = document.createElement('button');
                button.className = 'tab-button';
                button.textContent = 'Head to head';
                button.dataset.tab = 'scan-comparison-section';
                tabBar.appendChild(button);
            }

            const strategyButton = document.createElement('button');
            strategyButton.className = 'tab-button';
            strategyButton.textContent = 'Strategy metrics';
            strategyButton.dataset.tab = 'strategy-comparison-section';
            tabBar.appendChild(strategyButton);

            const tabButtons = tabBar.querySelectorAll('.tab-button');
            if (tabButtons.length > 0) {
                tabButtons[0].classList.add('is-active');
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

                tabButtons.forEach(button => button.classList.remove('is-active'));
                target.classList.add('is-active');

                tabPanels.forEach(panel => {
                    if (panel.id === target.dataset.tab) {
                        panel.classList.remove('is-hidden');
                    } else {
                        panel.classList.add('is-hidden');
                    }
                });
            });
        }

        // --- Strategy metrics (model_comparison bar charts) ---
        const compGeneratorType = document.getElementById('comp-generator-type');
        const compGenerator = document.getElementById('comp-generator');
        const compNoise = document.getElementById('comp-noise');
        const compIframeAbsErr = document.getElementById('comp-iframe-abs-err');
        const compIframeMeasurements = document.getElementById('comp-iframe-measurements');
        const compIframeDuration = document.getElementById('comp-iframe-duration');

        const modelCompPlots = plots.filter(p => p.type === 'model_comparison');

        function updateCompControls() {
            if (!compGeneratorType || !compGenerator || !compNoise) return;
            const uniqueGenTypes = [...new Set(modelCompPlots.map(p => p.generator_type || (p.generator.startsWith('NVCenter-') ? 'NV Center' : 'Complementary')))].sort();
            const selectedGenType = renderSegmentedControl(compGeneratorType, uniqueGenTypes, controlValue(compGeneratorType));

            const uniqueGenerators = [...new Set(
                modelCompPlots
                    .filter(p => (p.generator_type || (p.generator.startsWith('NVCenter-') ? 'NV Center' : 'Complementary')) === selectedGenType)
                    .map(p => p.generator)
            )].sort();
            const selectedGen = renderSegmentedControl(compGenerator, uniqueGenerators, controlValue(compGenerator));

            const uniqueNoises = [...new Set(
                modelCompPlots
                    .filter(p => p.generator === selectedGen)
                    .map(p => p.noise)
            )].sort();

            renderSegmentedControl(compNoise, uniqueNoises, controlValue(compNoise));
        }

        function updateCompPlots() {
            if (!compIframeAbsErr || !compIframeMeasurements || !compIframeDuration) return;
            const gen = controlValue(compGenerator);
            const noise = controlValue(compNoise);

            if (!gen || !noise) {
                compIframeAbsErr.src = '';
                compIframeMeasurements.src = '';
                compIframeDuration.src = '';
                return;
            }

            const absErrPlot = modelCompPlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'abs_err_x');
            const measurementsPlot = modelCompPlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'measurements');
            const durationPlot = modelCompPlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'duration_ms');

            compIframeAbsErr.src = absErrPlot ? absErrPlot.path : '';
            compIframeMeasurements.src = measurementsPlot ? measurementsPlot.path : '';
            compIframeDuration.src = durationPlot ? durationPlot.path : '';
        }

        if (compGeneratorType && compGenerator && compNoise) {
            compGeneratorType.addEventListener('controlchange', () => {
                updateCompControls();
                updateCompPlots();
            });

            compGenerator.addEventListener('controlchange', () => {
                updateCompControls();
                updateCompPlots();
            });

            compNoise.addEventListener('controlchange', () => {
                updateCompPlots();
            });
        }

        // --- Scan Comparison: same signal (target / generator / noise / repeat), two locators ---
        function setupScanComparison() {
            const cmpGenType = document.getElementById('cmp-shared-generator-type');
            const cmpGen = document.getElementById('cmp-shared-generator');
            const cmpNoise = document.getElementById('cmp-shared-noise');
            const cmpRepeat = document.getElementById('cmp-shared-repeat');
            const leftStrat = document.getElementById('left-strategy');
            const rightStrat = document.getElementById('right-strategy');
            const headToHeadEl = document.getElementById('head-to-head-plot');
            const leftMetrics = document.getElementById('left-metrics');
            const rightMetrics = document.getElementById('right-metrics');

            if (
                !cmpGenType ||
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
                const selGenType = renderSegmentedControl(
                    cmpGenType,
                    [...new Set(scanPlots.map((p) => p.generator_type))].sort(),
                    controlValue(cmpGenType),
                );
                const genItems = scanPlots.filter((p) => p.generator_type === selGenType).map((p) => p.generator);
                const selGen = renderSegmentedControl(cmpGen, genItems, controlValue(cmpGen));
                const noiseItems = scanPlots
                    .filter((p) => p.generator_type === selGenType && p.generator === selGen)
                    .map((p) => p.noise);
                renderSegmentedControl(cmpNoise, noiseItems, controlValue(cmpNoise));
            }

            function updateCmpStrategyControls() {
                const selGenType = controlValue(cmpGenType);
                const stratItems = [
                    ...new Set(scanPlots.filter((p) => p.generator_type === selGenType).map((p) => p.strategy)),
                ];
                renderSegmentedControl(leftStrat, stratItems, controlValue(leftStrat));
                renderSegmentedControl(rightStrat, stratItems, controlValue(rightStrat));
            }

            function repeatStringsFor(gt, g, n, strat) {
                return scanPlots
                    .filter(
                        (p) =>
                            p.generator_type === gt &&
                            p.generator === g &&
                            p.noise === n &&
                            p.strategy === strat,
                    )
                    .map((p) => String(p.repeat ?? p.attempt ?? 1));
            }

            function updateCmpRepeatControl() {
                const gt = controlValue(cmpGenType);
                const g = controlValue(cmpGen);
                const n = controlValue(cmpNoise);
                const sl = controlValue(leftStrat);
                const sr = controlValue(rightStrat);
                const repsL = repeatStringsFor(gt, g, n, sl);
                const repsR = repeatStringsFor(gt, g, n, sr);
                const setR = new Set(repsR);
                let common = [...new Set(repsL)].filter((r) => setR.has(r));
                common.sort((a, b) => Number(a) - Number(b));
                if (common.length === 0) {
                    common = [...new Set([...repsL, ...repsR])].sort((a, b) => Number(a) - Number(b));
                }
                const {value: selRep} = renderSelectControl(
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
                    const absErr = formatMetricValue(plot.abs_err_x);
                    const uncertainty = formatMetricValue(plot.uncert);
                    const measurements = formatCount(plot.measurements);
                    const duration = formatDuration(plot.duration_ms);
                    el.textContent = `Measurements: ${measurements} • Duration: ${duration} • Abs Error: ${absErr} • Uncertainty: ${uncertainty}`;
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
                const vGenType = controlValue(cmpGenType);
                const vGen = controlValue(cmpGen);
                const vNoise = controlValue(cmpNoise);
                const vStratL = controlValue(leftStrat);
                const vStratR = controlValue(rightStrat);
                const repStr = controlValue(cmpRepeat);
                const vRep = repStr ? parseInt(repStr, 10) : NaN;

                if (!vGenType || !vGen || !vNoise || !vStratL || !vStratR || !Number.isFinite(vRep)) {
                    clearHeadToHeadPlot('');
                    applyMetrics(leftMetrics, null);
                    applyMetrics(rightMetrics, null);
                    return;
                }

                const plotL = scanPlots.find(
                    (p) =>
                        p.generator_type === vGenType &&
                        p.generator === vGen &&
                        p.noise === vNoise &&
                        p.strategy === vStratL &&
                        p.repeat === vRep,
                );
                const plotR = scanPlots.find(
                    (p) =>
                        p.generator_type === vGenType &&
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

            cmpGenType.addEventListener('controlchange', () => {
                updateCmpSharedSignalControls();
                updateCmpStrategyControls();
                updateCmpRepeatControl();
                updateCmpPlots();
            });
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
                cmpGenType.dataset.value = scanDefault.generator_type ?? '';
                cmpGen.dataset.value = scanDefault.generator ?? '';
                cmpNoise.dataset.value = scanDefault.noise ?? '';
                cmpRepeat.dataset.value =
                    scanDefault.repeat === undefined ? '' : String(scanDefault.repeat);
                const types = [...new Set(scanPlots.map((p) => p.generator_type))];
                const defType = scanDefault.generator_type ?? types[0];
                const strats = [
                    ...new Set(scanPlots.filter((p) => p.generator_type === defType).map((p) => p.strategy)),
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


        scanGeneratorType.addEventListener('controlchange', () => {
            updateScanSignalControls();
            updateScanStrategyControl();
            updateScanRepeatControl();
            findAndDisplayPlot();
        });
        scanGenerator.addEventListener('controlchange', () => {
            updateScanSignalControls();
            updateScanRepeatControl();
            findAndDisplayPlot();
        });
        scanNoise.addEventListener('controlchange', () => {
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
            scanGeneratorType.dataset.value = scanDefault.generator_type ?? '';
            scanGenerator.dataset.value = scanDefault.generator ?? '';
            scanNoise.dataset.value = scanDefault.noise ?? '';
            scanStrategy.dataset.value = scanDefault.strategy ?? '';
            scanRepeat.dataset.value =
                scanDefault.repeat === undefined || scanDefault.repeat === null
                    ? ''
                    : String(scanDefault.repeat);
        }

        try {
            setupTabs();
            updateCompControls();
            updateCompPlots();
            updateAllScanControls();
            findAndDisplayPlot();
        } catch (error) {
            console.error('Error initializing UI controls:', error);
            // Show an error message to the user
            const errorDiv = document.createElement('div');
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

    window.addEventListener('DOMContentLoaded', () => {
        window.NVISION_BOOTSTRAP
            .then(() => {
                main();
            })
            .catch((error) => {
                console.error('Failed to initialize UI assets:', error);
                const errorDiv = document.createElement('div');
                errorDiv.style.padding = '20px';
                errorDiv.style.margin = '20px';
                errorDiv.style.border = '1px solid #f5c6cb';
                errorDiv.style.backgroundColor = '#f8d7da';
                errorDiv.style.color = '#721c24';
                errorDiv.style.borderRadius = '4px';
                errorDiv.innerHTML = '<h3>Error: Failed to initialize UI assets</h3><p>Could not load manifest/settings data files.</p>';
                document.body.appendChild(errorDiv);
            });
    });
