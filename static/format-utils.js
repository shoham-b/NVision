// Pure formatting utilities — no DOM or Plotly dependencies.
// All functions are global so app.js can call them directly.

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

// Standard NV-center electron gyromagnetic ratio (Hz/Tesla). zeeman_split is the
// half-separation between the two Zeeman groups, i.e. gamma * B.
const NV_GYROMAGNETIC_HZ_PER_TESLA = 28.025e9;

function zeemanSplitToMagneticFieldTesla(zeemanSplitHz) {
    if (typeof zeemanSplitHz !== 'number' || !Number.isFinite(zeemanSplitHz)) return null;
    return zeemanSplitHz / NV_GYROMAGNETIC_HZ_PER_TESLA;
}

function formatMagneticField(teslaValue) {
    if (typeof teslaValue !== 'number' || !Number.isFinite(teslaValue)) return 'N/A';
    const absVal = Math.abs(teslaValue);
    if (absVal >= 1) {
        return teslaValue.toFixed(3) + ' T';
    } else if (absVal >= 1e-3) {
        return (teslaValue * 1e3).toFixed(2) + ' mT';
    } else if (absVal >= 1e-6 || absVal === 0) {
        return (teslaValue * 1e6).toFixed(2) + ' µT';
    } else {
        return (teslaValue * 1e6).toPrecision(3) + ' µT';
    }
}

function formatCount(value) {
    if (typeof value === 'number' && Number.isFinite(value)) {
        return Math.round(value).toString();
    }
    return 'N/A';
}

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

// Canonical single-symbol labels for signal-model parameters (see nvision/spectra/
// numba_kernels.py, nv_center.py, voigt_zeeman.py for the formulas these match).
// Used in the signal equation panel and wherever else a parameter name is shown in
// the UI, so the same parameter always reads with the same symbol everywhere.
const PARAM_LETTERS = {
    frequency: 'f_B',
    linewidth: 'w',
    dip_depth: 'A',
    background: 'B',
    split: 'δ',
    k_np: 'k',
    c_total: 'C',
    fwhm_total: 'Γ',
    lorentz_frac: 'r',
    zeeman_split: 'Δ',
    saturation: 's',
    sigma_inhom: 'σ',
    c_max: 'C₀',
    homogeneous_linewidth: 'γ',
};

function paramLetter(param) {
    return PARAM_LETTERS[param] || null;
}

// e.g. " [f]" — appended after an existing "name (unit)" label; '' when the
// parameter has no assigned letter (not part of a documented signal equation).
function paramLetterSuffix(param) {
    const l = paramLetter(param);
    return l ? ` [${l}]` : '';
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
