"""Extract plot data using Plotly's JSON format."""
from pathlib import Path
import re, json, base64, struct

html = Path('artifacts/graphs/scans/nvcenter-voigt-zeeman_gauss-0-01_genericsweep_r1.html').read_text()

# Find Plotly.newPlot
m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[.*?\])\s*,\s*(\{.*?\})\s*\)', html, re.DOTALL)
if not m:
    print('No Plotly.newPlot found')
    exit(1)

data_str = m.group(1)

# Decode base64 arrays
def decode_b64(match):
    dtype = match.group(1)
    b64 = match.group(2)
    raw = base64.b64decode(b64)
    if dtype == 'f8':
        arr = struct.unpack('<' + 'd' * (len(raw) // 8), raw)
        return json.dumps(list(arr))
    elif dtype == 'i4':
        arr = struct.unpack('<' + 'i' * (len(raw) // 4), raw)
        return json.dumps(list(arr))
    return match.group(0)

# Replace base64 encoded arrays
data_fixed = re.sub(r'"dtype":"([^"]+)","bdata":"([^"]+)"', decode_b64, data_str)

try:
    data = json.loads(data_fixed)
except json.JSONDecodeError as e:
    print(f'JSON error: {e}')
    # Try to find the error position
    ctx_start = max(0, e.pos - 200)
    ctx_end = min(len(data_fixed), e.pos + 200)
    print(f'Context: {data_fixed[ctx_start:ctx_end]}')
    exit(1)

for trace in data:
    name = trace.get('name', 'unnamed')
    xs = trace.get('x', [])
    ys = trace.get('y', [])
    if isinstance(xs, list) and len(xs) > 0:
        print(f'{name}: x=[{min(xs):.6e}, {max(xs):.6e}], y=[{min(ys):.4f}, {max(ys):.4f}], count={len(xs)}')
        if 'measurement' in name.lower() or 'coarse' in name.lower():
            print(f'  Num y < 0.9: {sum(1 for y in ys if y < 0.9)}')
            print(f'  Num y < 0.5: {sum(1 for y in ys if y < 0.5)}')
    else:
        print(f'{name}: x type={type(xs).__name__}, y type={type(ys).__name__}')
