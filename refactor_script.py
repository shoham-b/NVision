import os
import re

NVISION_DIR = r"C:\Users\shoha\git\NVision\nvision"

# 1. Delete jax_kernels.py
jax_file = os.path.join(NVISION_DIR, "spectra", "jax_kernels.py")
if os.path.exists(jax_file):
    os.remove(jax_file)
    print(f"Deleted {jax_file}")

# 2. Update nv_center_generator.py
generator_file = os.path.join(NVISION_DIR, "sim", "gen", "nv_center_generator.py")
with open(generator_file, "r") as f:
    gen_content = f.read()

# Replace dip_depth with c_total in generator
# "unit_dip_depth" becomes "c_total"
# "dip_depth=dip_depth," becomes "c_total=c_total,"
gen_content = gen_content.replace(
"""            unit_dip_depth = rng.uniform(0.3, 0.95)
            lw2 = linewidth**2
            model = NVCenterLorentzianModel()
            # Scale a desired contrast onto the true peak-shape maximum
            xs = np.linspace(center_freq - split, center_freq + split, 200)
            g = (
                (lw2 / k_np**2) / ((xs - (center_freq - split)) ** 2 + lw2)
                + (lw2 / k_np) / ((xs - center_freq) ** 2 + lw2)
                + lw2 / ((xs - (center_freq + split)) ** 2 + lw2)
            )
            dip_depth = unit_dip_depth / float(g.max())
            typed_params = NVCenterLorentzianSpectrum(
                frequency=center_freq,
                linewidth=linewidth,
                split=split,
                k_np=k_np,
                dip_depth=dip_depth,
            )
            bounds = nv_center_lorentzian_bounds_for_domain(self.x_min, self.x_max)
            
            # Generate Gaussian priors for all parameters except frequency
            prior_split = max(MIN_SPLIT, min(MAX_SPLIT, rng.gauss(split, 0.1e6)))
            prior_linewidth = max(MIN_LINEWIDTH, min(MAX_LINEWIDTH, rng.gauss(linewidth, 10e3)))
            prior_k_np = max(MIN_K_NP, min(MAX_K_NP, rng.gauss(k_np, 0.1)))
            prior_dip_depth = max(0.1, min(1.0, rng.gauss(dip_depth, 0.05)))
            
            bounds["_priors"] = {
                "split": (prior_split, 0.1e6),
                "linewidth": (prior_linewidth, 10e3),
                "k_np": (prior_k_np, 0.1),
                "dip_depth": (prior_dip_depth, 0.05),
                "frequency": ("sin^2", np.pi / width),
            }""",
"""            c_total = rng.uniform(0.1, 0.4)
            model = NVCenterLorentzianModel()
            
            typed_params = NVCenterLorentzianSpectrum(
                frequency=center_freq,
                linewidth=linewidth,
                split=split,
                k_np=k_np,
                c_total=c_total,
            )
            bounds = nv_center_lorentzian_bounds_for_domain(self.x_min, self.x_max)
            
            # Generate Gaussian priors for all parameters except frequency
            prior_split = max(MIN_SPLIT, min(MAX_SPLIT, rng.gauss(split, 0.1e6)))
            prior_linewidth = max(MIN_LINEWIDTH, min(MAX_LINEWIDTH, rng.gauss(linewidth, 10e3)))
            prior_k_np = max(MIN_K_NP, min(MAX_K_NP, rng.gauss(k_np, 0.1)))
            prior_c_total = max(0.1, min(0.4, rng.gauss(c_total, 0.05)))
            
            bounds["_priors"] = {
                "split": (prior_split, 0.1e6),
                "linewidth": (prior_linewidth, 10e3),
                "k_np": (prior_k_np, 0.1),
                "c_total": (prior_c_total, 0.05),
                "frequency": ("sin^2", np.pi / width),
            }""")

with open(generator_file, "w") as f:
    f.write(gen_content)
print("Updated nv_center_generator.py")

# 3. Update nv_center.py
nv_center_file = os.path.join(NVISION_DIR, "spectra", "nv_center.py")
with open(nv_center_file, "r") as f:
    nvc_content = f.read()

# Remove JAX imports
nvc_content = re.sub(r"from nvision\.spectra\.jax_kernels import [^\n]+\n", "", nvc_content)
nvc_content = re.sub(r"    def compute_jax\(self.*?return [^\n]+\n", "", nvc_content, flags=re.DOTALL)
# Wait, jax is 9-10 lines per compute_jax method
nvc_content = re.sub(r"    def compute_jax\(self, x: float, params:[^\)]+\) -> Any:\n(?: {8}[^\n]+\n)*", "", nvc_content)

# Remove gradient_vectorized_many
nvc_content = re.sub(r"    def gradient_vectorized_many\(self, x_phys_array: Sequence\[float\], samples_phys: NVCenterLorentzianSpectrumSamples\) -> np\.ndarray:\n(?: {8}[^\n]+\n)*", "", nvc_content)

# Rename dip_depth to c_total for Lorentzian classes
nvc_content = nvc_content.replace(
"""@dataclass(frozen=True)
class NVCenterLorentzianSpectrum:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    dip_depth: float

    @property
    def physical_amplitude(self) -> float:
        \"\"\"Physical Hz² amplitude (numerator): right peak depth × linewidth².\"\"\"
        return (self.dip_depth / self.k_np) * self.linewidth**2""",
"""@dataclass(frozen=True)
class NVCenterLorentzianSpectrum:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    c_total: float

    @property
    def physical_amplitude(self) -> float:
        \"\"\"Physical Hz² amplitude (numerator): approximation.\"\"\"
        return (self.c_total / self.k_np) * self.linewidth**2""")

nvc_content = nvc_content.replace(
"""@dataclass(frozen=True)
class NVCenterLorentzianSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    dip_depth: np.ndarray""",
"""@dataclass(frozen=True)
class NVCenterLorentzianSpectrumSamples:
    frequency: np.ndarray
    linewidth: np.ndarray
    split: np.ndarray
    k_np: np.ndarray
    c_total: np.ndarray""")

nvc_content = nvc_content.replace(
"""@dataclass(frozen=True)
class NVCenterLorentzianSpectrumUncertainty:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    dip_depth: float""",
"""@dataclass(frozen=True)
class NVCenterLorentzianSpectrumUncertainty:
    frequency: float
    linewidth: float
    split: float
    k_np: float
    c_total: float""")

# Update NVCenterLorentzianModel
nvc_content = nvc_content.replace(
"""    def compute_nvcenter_lorentzian_model(
        x: float,
        frequency: float,
        linewidth: float,
        split: float,
        k_np: float,
        dip_depth: float,
    ) -> float:""",
"""    def compute_nvcenter_lorentzian_model(
        x: float,
        frequency: float,
        linewidth: float,
        split: float,
        k_np: float,
        c_total: float,
    ) -> float:""")

nvc_content = nvc_content.replace(
"""        return nv_center_lorentzian_eval(
            float(x),
            float(frequency),
            float(linewidth),
            float(split),
            float(k_np),
            float(dip_depth),
            1.0,
        )""",
"""        return nv_center_lorentzian_eval(
            float(x),
            float(frequency),
            float(linewidth),
            float(split),
            float(k_np),
            float(c_total),
            1.0,
        )""")

nvc_content = nvc_content.replace(
"""    def compute_nvcenter_lorentzian_model_vectorized(
        self,
        x: float,
        frequency: np.ndarray,
        linewidth: np.ndarray,
        split: np.ndarray,
        k_np: np.ndarray,
        dip_depth: np.ndarray,
    ) -> np.ndarray:""",
"""    def compute_nvcenter_lorentzian_model_vectorized(
        self,
        x: float,
        frequency: np.ndarray,
        linewidth: np.ndarray,
        split: np.ndarray,
        k_np: np.ndarray,
        c_total: np.ndarray,
    ) -> np.ndarray:""")

nvc_content = nvc_content.replace(
"""            np.asarray(dip_depth, dtype=FLOAT_DTYPE),
            bg,
            out,
        )""",
"""            np.asarray(c_total, dtype=FLOAT_DTYPE),
            bg,
            out,
        )""")

nvc_content = nvc_content.replace(
"""    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "dip_depth")

    def parameter_weights(self) -> dict[str, float]:
        return {"frequency": 2.0, "linewidth": 1.0, "split": 1.0, "k_np": 1.0, "dip_depth": 1.0}""",
"""    def is_scale_parameter(self, name: str) -> bool:
        return name in ("linewidth", "c_total")

    def parameter_weights(self) -> dict[str, float]:
        return {"frequency": 2.0, "linewidth": 1.0, "split": 1.0, "k_np": 1.0, "c_total": 1.0}""")

nvc_content = nvc_content.replace(
"""    def compute(self, x: float, params: NVCenterLorentzianSpectrum) -> float:
        return self.compute_nvcenter_lorentzian_model(
            float(x),
            params.frequency,
            params.linewidth,
            params.split,
            params.k_np,
            params.dip_depth / params.k_np,
        )""",
"""    def compute(self, x: float, params: NVCenterLorentzianSpectrum) -> float:
        return self.compute_nvcenter_lorentzian_model(
            float(x),
            params.frequency,
            params.linewidth,
            params.split,
            params.k_np,
            params.c_total,
        )""")

nvc_content = nvc_content.replace(
"""            samples.k_np,
            samples.dip_depth,
        )""",
"""            samples.k_np,
            samples.c_total,
        )""")

nvc_content = nvc_content.replace(
"""            np.asarray(samples_phys.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples_phys.dip_depth, dtype=FLOAT_DTYPE),""",
"""            np.asarray(samples_phys.k_np, dtype=FLOAT_DTYPE),
            np.asarray(samples_phys.c_total, dtype=FLOAT_DTYPE),""")

nvc_content = nvc_content.replace(
"""        "k_np": (MIN_K_NP, MAX_K_NP),
        "dip_depth": (0.1, 1.0),
        "_signal_max_span": (0.0, max_span),""",
"""        "k_np": (MIN_K_NP, MAX_K_NP),
        "c_total": (0.1, 0.4),
        "_signal_max_span": (0.0, max_span),""")

with open(nv_center_file, "w") as f:
    f.write(nvc_content)
print("Updated nv_center.py")

# 4. Update numba_kernels.py
numba_file = os.path.join(NVISION_DIR, "spectra", "numba_kernels.py")
with open(numba_file, "r") as f:
    numba_content = f.read()

# Delete gradient methods from numba_kernels
numba_content = re.sub(r"@njit\(cache=True, fastmath=True\)\ndef nv_center_lorentzian_gradient\(.*?grad_phys\[4\] = -\(tl / k2 \+ \(lw2 \* iden_c\) / k_safe \+ tr\)\n\n\n", "", numba_content, flags=re.DOTALL)
numba_content = re.sub(r"@njit\(cache=True, parallel=True\)\ndef nv_center_lorentzian_gradient_vectorized_many\(.*?nv_center_lorentzian_gradient\(x, f, lw, s, knp, d, grad_phys_out\[i, j\]\)\n\n\n", "", numba_content, flags=re.DOTALL)

# In numba_kernels, change dip_depth to c_total everywhere in lorentzian functions
numba_content = numba_content.replace(
"""def nv_center_lorentzian_eval(
    x: float,
    freq: float,
    linewidth: float,
    split: float,
    k_np: float,
    dip_depth: float,
    background: float,
) -> float:""",
"""def nv_center_lorentzian_eval(
    x: float,
    freq: float,
    linewidth: float,
    split: float,
    k_np: float,
    c_total: float,
    background: float,
) -> float:""")

numba_content = numba_content.replace("    c_total = dip_depth\n", "    # c_total is passed directly\n")

numba_content = numba_content.replace(
"""def nv_center_lorentzian_vectorized_many(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    dip_depth: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:""",
"""def nv_center_lorentzian_vectorized_many(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:""")

numba_content = numba_content.replace(
"""def nv_center_lorentzian_vectorized_one(
    x: float,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    dip_depth: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:""",
"""def nv_center_lorentzian_vectorized_one(
    x: float,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:""")

numba_content = numba_content.replace(
"""def nv_center_lorentzian_vectorized_many_fast(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    dip_depth: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:""",
"""def nv_center_lorentzian_vectorized_many_fast(
    xs: np.ndarray,
    freq: np.ndarray,
    linewidth: np.ndarray,
    split: np.ndarray,
    k_np: np.ndarray,
    c_total: np.ndarray,
    background: np.ndarray,
    out: np.ndarray,
) -> None:""")

def update_vectorized_body(content, func_name):
    pattern = rf"(def {func_name}\(.*?\):\n(?:.*?)for j in .*?:\n)(.*?)(?=            out\[)"
    
    def repl(m):
        header = m.group(1)
        return header + """            lw = linewidth[j]
            f = freq[j]
            s = split[j]
            k = k_np[j]
            c = c_total[j]
            bg = background[j]

            omega = lw if lw > 1e-10 else 1e-10
            k_safe = k if k > 1e-10 else 1e-10
            
            x_dim = (x - f) / omega
            alpha = s / omega
            p_sum = k_safe + 1.0 + (1.0 / k_safe)
            
            p_0 = c / p_sum
            p_l = c * (k_safe / p_sum)
            p_r = c * ((1.0 / k_safe) / p_sum)

"""
    return re.sub(pattern, repl, content, flags=re.DOTALL)

def update_vectorized_many_body(content, func_name):
    pattern = rf"(def {func_name}\(.*?\):\n(?:.*?)(?:    for i in prange\(m\):\n        x = xs\[i\]\n        for j in range\(n\):\n)(.*?)(?=            out\[))"
    
    def repl(m):
        header = m.group(1)
        body = """            lw = linewidth[j]
            f = freq[j]
            s = split[j]
            k = k_np[j]
            c = c_total[j]
            bg = background[j]

            omega = lw if lw > 1e-10 else 1e-10
            k_safe = k if k > 1e-10 else 1e-10
            
            x_dim = (x - f) / omega
            alpha = s / omega
            p_sum = k_safe + 1.0 + (1.0 / k_safe)
            
            p_0 = c / p_sum
            p_l = c * (k_safe / p_sum)
            p_r = c * ((1.0 / k_safe) / p_sum)

"""
        # We need to replace only the body part inside the match
        full = m.group(0)
        return full.replace(m.group(2), body)
    return re.sub(pattern, repl, content, flags=re.DOTALL)


numba_content = update_vectorized_many_body(numba_content, "nv_center_lorentzian_vectorized_many")
numba_content = update_vectorized_many_body(numba_content, "nv_center_lorentzian_vectorized_many_fast")

pattern_one = rf"(def nv_center_lorentzian_vectorized_one\(.*?\):\n(?:.*?)    for j in prange\(n\):\n)(.*?)(?=        out\[)"
def repl_one(m):
    return m.group(1) + """        lw = linewidth[j]
        f = freq[j]
        s = split[j]
        k = k_np[j]
        c = c_total[j]
        bg = background[j]

        omega = lw if lw > 1e-10 else 1e-10
        k_safe = k if k > 1e-10 else 1e-10
        
        x_dim = (x - f) / omega
        alpha = s / omega
        p_sum = k_safe + 1.0 + (1.0 / k_safe)
        
        p_0 = c / p_sum
        p_l = c * (k_safe / p_sum)
        p_r = c * ((1.0 / k_safe) / p_sum)

"""
numba_content = re.sub(pattern_one, repl_one, numba_content, flags=re.DOTALL)

# Fix the out assignment to use p_l, p_0, p_r
numba_content = numba_content.replace(
    "out[i, j] = bg - (amp_l / denom_l + amp_c / denom_c + amp_r / denom_r)",
    "out[i, j] = bg - (p_l / ((x_dim + alpha)**2 + 1.0) + p_0 / (x_dim**2 + 1.0) + p_r / ((x_dim - alpha)**2 + 1.0))"
)
numba_content = numba_content.replace(
    "out[j] = bg - (amp_l / denom_l + amp_c / denom_c + amp_r / denom_r)",
    "out[j] = bg - (p_l / ((x_dim + alpha)**2 + 1.0) + p_0 / (x_dim**2 + 1.0) + p_r / ((x_dim - alpha)**2 + 1.0))"
)

with open(numba_file, "w") as f:
    f.write(numba_content)
print("Updated numba_kernels.py")

# Also need to remove jax imports from other files if any
for file in ["gaussian.py", "lorentzian.py", "voigt_zeeman.py", "signal.py", "composite.py"]:
    filepath = os.path.join(NVISION_DIR, "spectra", file)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
        content = re.sub(r"from nvision\.spectra\.jax_kernels import [^\n]+\n", "", content)
        content = re.sub(r"    def compute_jax\(self.*?return [^\n]+\n", "", content, flags=re.DOTALL)
        content = re.sub(r"    def compute_jax\(self, x: float, params:[^\)]+\) -> Any:\n(?: {8}[^\n]+\n)*", "", content)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Cleaned JAX from {file}")
