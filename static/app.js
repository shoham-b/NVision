window.NVISION_ASSET_PREFIX = '';
        window.NVISION_BOOTSTRAP = (async () => {
            async function loadScript(candidates, onLoaded) {
                for (const candidate of candidates) {
                    const loaded = await new Promise((resolve) => {
                        const script = document.createElement('script');
                        script.src = candidate.src;
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
                throw new Error('Could not load manifest.js from current directory or ../artifacts/');
            }

            const settingsCandidate = await loadScript([
                { src: 'settings.js' },
                { src: './settings.js' },
                { src: '../artifacts/settings.js' },
            ]);
            if (!settingsCandidate) {
                throw new Error('Could not load settings.js from current directory or ../artifacts/');
            }
        })();

function main() {
        let plots = [];
        try {
            plots = window.MANIFEST;
            if (!Array.isArray(plots) || plots.length === 0) {
                throw new Error('No plots data available');
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

        function updateControls() {
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
            const selectedScanNoise = renderSegmentedControl(
                scanNoise,
                scanNoiseItems,
                controlValue(scanNoise),
            );

            const scanStrategyItems = [
                ...new Set(
                    scanPlots
                        .filter((p) => p.generator_type === selectedScanGeneratorType)
                        .map((p) => p.strategy)
                ),
            ];

            const selectedScanStrategy = renderSegmentedControl(
                scanStrategy,
                scanStrategyItems,
                controlValue(scanStrategy),
            );

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

        function setupTabs() {
            const tabBar = document.querySelector('.tab-bar');
            const tabPanels = document.querySelectorAll('.tab-panel');

            tabBar.style.display = 'flex';
            tabBar.innerHTML = ''; // Clear existing

            const hasScans = scanPlots.length > 0;
            // Check for generic comparison plots
            const hasModelComp = plots.some(p => p.type === 'model_comparison');

            if (hasScans) {
                const button = document.createElement('button');
                button.className = 'tab-button';
                button.textContent = 'Scan Measurements';
                button.dataset.tab = 'scan-section';
                tabBar.appendChild(button);
            }

            if (hasModelComp) {
                const button = document.createElement('button');
                button.className = 'tab-button';
                button.textContent = 'Strategy Comparison';
                button.dataset.tab = 'strategy-comparison-section';
                tabBar.appendChild(button);
            }

            if (hasScans) { // Reuse hasScans for locator comparison as it relies on same data
                const button = document.createElement('button');
                button.className = 'tab-button';
                button.textContent = 'Scan Comparison';
                button.dataset.tab = 'scan-comparison-section';
                tabBar.appendChild(button);
            }

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

        // --- Strategy Comparison Logic ---
        const compGeneratorType = document.getElementById('comp-generator-type');
        const compGenerator = document.getElementById('comp-generator');
        const compNoise = document.getElementById('comp-noise');
        const compIframeAbsErr = document.getElementById('comp-iframe-abs-err');
        const compIframeMeasurements = document.getElementById('comp-iframe-measurements');
        const compIframeDuration = document.getElementById('comp-iframe-duration');

        const modelCompPlots = plots.filter(p => p.type === 'model_comparison');

        function updateCompControls() {
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
            const gen = controlValue(compGenerator);
            const noise = controlValue(compNoise);

            if (!gen || !noise) return;

            const absErrPlot = modelCompPlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'abs_err_x');
            const measurementsPlot = modelCompPlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'measurements');
            const durationPlot = modelCompPlots.find(p => p.generator === gen && p.noise === noise && p.metric === 'duration_ms');

            compIframeAbsErr.src = absErrPlot ? absErrPlot.path : '';
            compIframeMeasurements.src = measurementsPlot ? measurementsPlot.path : '';
            compIframeDuration.src = durationPlot ? durationPlot.path : '';
        }

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

        // --- Scan Comparison Logic ---
        function setupSide(prefix) {
            const genType = document.getElementById(prefix + '-generator-type');
            const gen = document.getElementById(prefix + '-generator');
            const noise = document.getElementById(prefix + '-noise');
            const strat = document.getElementById(prefix + '-strategy');
            const rep = document.getElementById(prefix + '-repeat');
            const iframe = document.getElementById(prefix + '-iframe');
            const metrics = document.getElementById(prefix + '-metrics');

            let currentRepItems = [];

            function updateSideControls() {
                const selGenType = renderSegmentedControl(genType, [...new Set(scanPlots.map(p => p.generator_type))].sort(), controlValue(genType));

                const genItems = scanPlots.filter(p => p.generator_type === selGenType).map(p => p.generator);
                const selGen = renderSegmentedControl(gen, genItems, controlValue(gen));

                const noiseItems = scanPlots.filter(p => p.generator_type === selGenType && p.generator === selGen).map(p => p.noise);
                const selNoise = renderSegmentedControl(noise, noiseItems, controlValue(noise));

                const stratItems = [...new Set(scanPlots.filter(p => p.generator_type === selGenType && p.generator === selGen && p.noise === selNoise).map(p => p.strategy))];
                const selStrat = renderSegmentedControl(strat, stratItems, controlValue(strat));

                const repItems = scanPlots.filter(p =>
                    p.generator_type === selGenType &&
                    p.generator === selGen &&
                    p.noise === selNoise &&
                    p.strategy === selStrat
                ).map(p => String(p.repeat ?? p.attempt ?? 1));

                const {
                    value: selRep,
                    items
                } = renderSelectControl(rep, repItems, controlValue(rep) || rep.dataset.value || '');
                currentRepItems = items;
                if (selRep) rep.dataset.value = selRep;
            }

            function updateSidePlot() {
                const vGenType = controlValue(genType);
                const vGen = controlValue(gen);
                const vNoise = controlValue(noise);
                const vStrat = controlValue(strat);
                const vRep = parseInt(controlValue(rep), 10);

                if (vGenType && vGen && vNoise && vStrat && vRep) {
                    const plot = scanPlots.find(p =>
                        p.generator_type === vGenType &&
                        p.generator === vGen &&
                        p.noise === vNoise &&
                        p.strategy === vStrat &&
                        p.repeat === vRep
                    );

                    iframe.src = plot ? plot.path : '';
                    if (plot) {
                        const absErr = formatMetricValue(plot.abs_err_x);
                        const uncertainty = formatMetricValue(plot.uncert);
                        const measurements = formatCount(plot.measurements);
                        const duration = formatDuration(plot.duration_ms);
                        metrics.textContent = `Measurements: ${measurements} • Duration: ${duration} • Abs Error: ${absErr} • Uncertainty: ${uncertainty}`;
                    } else {
                        metrics.textContent = '';
                    }
                } else {
                    iframe.src = '';
                    metrics.textContent = '';
                }
            }

            function onControlChange() {
                updateSideControls();
                updateSidePlot();
            }

            genType.addEventListener('controlchange', onControlChange);
            gen.addEventListener('controlchange', onControlChange);
            noise.addEventListener('controlchange', onControlChange);
            strat.addEventListener('controlchange', onControlChange);
            rep.addEventListener('change', () => {
                rep.dataset.value = rep.value || '';
                updateSidePlot();
            });

            // Init
            if (scanDefault) {
                genType.dataset.value = scanDefault.generator_type ?? '';
                gen.dataset.value = scanDefault.generator ?? '';
                noise.dataset.value = scanDefault.noise ?? '';
                strat.dataset.value = scanDefault.strategy ?? '';
                rep.dataset.value = scanDefault.repeat === undefined ? '' : String(scanDefault.repeat);
            }
            updateSideControls();
            updateSidePlot();
        }

        if (document.getElementById('scan-comparison-section')) {
            setupSide('left');
            setupSide('right');
        }


        scanGeneratorType.addEventListener('controlchange', () => {
            updateControls();
            findAndDisplayPlot();
        });
        scanGenerator.addEventListener('controlchange', () => {
            updateControls();
            findAndDisplayPlot();
        });
        scanNoise.addEventListener('controlchange', () => {
            updateControls();
            findAndDisplayPlot();
        });
        scanStrategy.addEventListener('controlchange', () => {
            updateControls();
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
            updateControls();
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
