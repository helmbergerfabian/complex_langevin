### src/complex_langevin/config.py
import numpy as np
import os

CL_PRECISION = os.getenv("CL_PRECISION", "single").lower()

if CL_PRECISION == "double":
    CL_REAL = np.float64
    CL_COMPLEX = np.complex128
    CL_INT = np.int64
else:
    CL_REAL = np.float32
    CL_COMPLEX = np.complex64
    CL_INT = np.int32