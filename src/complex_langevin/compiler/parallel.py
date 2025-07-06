
### complex_langevin/compiler/parallel.py
from .factory import get_backend

_backend = get_backend()

parallel_loop = _backend.parallel_loop
act_parallel_loop = _backend.act_parallel_loop