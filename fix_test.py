import numpy as np

v1 = np.float32(0.9998598)
v2 = 1.0

print("np.allclose(v1, v2):", np.allclose(v1, v2))
print("np.allclose(v1, v2, atol=1e-3):", np.allclose(v1, v2, atol=1e-3))
