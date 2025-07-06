### complex_langevin/models/base.py
# import os
# target = os.environ.get('MY_NUMBA_TARGET', 'numba').lower()

# if target == 'cuda': from numba import cuda; CLArray = cuda.devicearray.DeviceNDArray
# elif target == 'numba': CLArray = np.ndarray
# elif target == 'python': CLArray = np.ndarray
# else: raise ValueError(f"Unknown target: {target}")

import numpy as np

from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    @abstractmethod
    def drift(self, x):
        pass