// Recalculate / reload — keyboard shortcut 'r', recalc button, and app init.
// Calls POST /api/reload then polls /api/status until the run finishes.
// Also owns the DOMContentLoaded → main() bootstrap sequence.

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
