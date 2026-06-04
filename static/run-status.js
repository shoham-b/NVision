// Run status banner — polls run_status.json and updates the banner.
// Also handles help-toggle accordion buttons (aria-expanded / hidden).

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
