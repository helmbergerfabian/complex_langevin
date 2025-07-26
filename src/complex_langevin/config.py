### src/complex_langevin/config.py
from numpy import float32, float64, complex64, complex128, int64, int32
import os

CL_PRECISION = os.getenv("CL_PRECISION", "single").lower()

if CL_PRECISION == "double":
    CL_REAL = float64
    CL_COMPLEX = complex128
    CL_INT = int64
else:
    CL_REAL = float32
    CL_COMPLEX = complex64
    CL_INT = int32


from time import time
import os
if os.getenv("VERBOSE", "False") == "True":
    def log(self, sender, message):
        print(f"{hex(id(self))}\t[{sender}]\t{message}\t({time()-float(os.environ.get('RUNNER_START_TIME', 0))})")
else:
    def log(self, sender, message): pass