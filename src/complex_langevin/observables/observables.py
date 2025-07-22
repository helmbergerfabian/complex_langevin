import numpy as np
from queue import Queue
from threading import Event

from complex_langevin.config import CL_COMPLEX, CL_INT
from complex_langevin.compiler import backend
from complex_langevin.simulation.state import SimState
from complex_langevin.observables.daq import DAQThread

class MomentObservable():
    def __init__(self, state: SimState, order: int, name = None, buffer_size = None):
        self.name = name
        self.backend = backend
        self.buffer_size = buffer_size or int(1e3)
        self.obs_running_sum = np.zeros(shape = state.n_seeds, dtype=CL_COMPLEX)
        self._buffer_idx = 0
        self.meas_count = np.zeros(shape = state.n_seeds, dtype=CL_INT)
        self.state = state
        self.accumulator = []
        self.kernel = self.generate_kernel()
        self.order = order
        self.out_q = Queue()
        self.daq_active = False

        if backend.use_cuda: 
            from complex_langevin.utils.gpu_handler import GPU_handler
            self.handler = GPU_handler(self)
            self.to_device()
        
    def to_device(self):
        self.handler.to_device()
        print(f"Copied {self.name} arrays to device")

    def to_host(self):
        self.handler.to_host()
        print(f"Copied {self.name} arrays to host")  

    def start_daq(self):
        self.stop_event = Event()
        self.daq = DAQThread(self.out_q, self.stop_event)
        self.daq.start()
        self.daq_active = True

    @property
    def phi(self):
        return self.state.phi_read
       
    def generate_kernel(self):
        @self.backend.kernel
        def _moment_kernel(idx, phi_arr, obs_running_sum, order):
            phi_idx = phi_arr[idx]
            obs = phi_idx**order
            obs_running_sum[idx] += obs
        return _moment_kernel
    
    def observe(self):
        self.backend.parallel_loop(self.kernel, self.state.n_seeds, self.state.phi_read, self.obs_running_sum, self.order)
        self._buffer_idx += 1

        if self.daq_active:
            if self._buffer_idx%self.buffer_size == 0: 
                if backend.use_cuda: 
                    data_out = self.obs_running_sum.copy_to_host()
                else: 
                    data_out = self.obs_running_sum.copy()
                self.out_q.put((data_out, self._buffer_idx))
    
    def finish(self):
        self.stop_event.set()
        self.daq.join()
