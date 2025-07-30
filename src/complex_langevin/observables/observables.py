from numpy import zeros
from queue import Queue
from threading import Event

from complex_langevin.config import CL_COMPLEX, CL_INT, CL_REAL, log
from complex_langevin.compiler import backend
from complex_langevin.simulation.state import SimState
from complex_langevin.observables.daq import DAQThread


from numba import njit, prange
import numpy as np

@njit(parallel=True)
# def mark_flush_mask(tau, tau_last_flush, flush_mask, delta_tau_block = 1, max_tau = 10, ):
def mark_flush_mask(tau, tau_last_flush, flush_mask, block_counts, delta_tau_block = 1, num_blocks=20):
    for i in prange(tau.shape[0]):
        # if tau[i] >= max_tau:
        #     flush_mask[i] = False
        #     continue
        if block_counts[i] > num_blocks:
            flush_mask[i] = False
            continue
        flush_mask[i] = (tau[i] - tau_last_flush[i]) > delta_tau_block

@njit(parallel=True)
def pack_block_means(obs_sum, count, tau, tau_last_flush, flush_mask, block_means):
    """
    Based on a flush_mask, block_means are calculated and running sum is reset. 
    """
    for i in prange(obs_sum.shape[0]):
        if flush_mask[i]:
            block_means[i] = obs_sum[i] / count[i]
            obs_sum[i] = 0.0
            count[i] = 0
            tau_last_flush[i] = tau[i]
        else:
            block_means[i] = np.nan 



class MomentObservable():
    def __init__(self, state: SimState, 
                 order: int, name = None, 
                 buffer_size = None, delta_tau_block = None, 
                 num_blocks = 50):
        import numpy as np
        self.name = name
        self.backend = backend
        self.buffer_size = buffer_size or int(1e3)
        self.obs_running_sum = zeros(shape = state.n_seeds, dtype=CL_COMPLEX)
        self.block_means = np.zeros_like(self.obs_running_sum)
        self.count = zeros(shape = state.n_seeds, dtype=CL_INT)

        self.flush_mask = zeros(shape = state.n_seeds, dtype=bool)
        self.last_flush = zeros(shape = state.n_seeds, dtype=CL_REAL)
        self.last_meas = zeros(shape = state.n_seeds, dtype=CL_REAL)
        self.cold = np.full(shape = state.n_seeds, fill_value=True)

        # self.flush_times = [[] for _ in range(state.n_seeds)]
        # self.hist = [[] for _ in range(state.n_seeds)]
        self.delta_tau_block = delta_tau_block or 1
        self.step_counter = 0
        # self.max_tau = max_tau
        self.num_blocks = num_blocks

        # self.meas_count = zeros(shape = state.n_seeds, dtype=CL_INT)
        self.state = state
        self.kernel = self.generate_kernel()
        self.order = order
        self.out_q = Queue()
        self.daq_active = False

        # prepare DAQ
        self.stop_event = Event()
        self.daq = DAQThread(self.state.n_seeds, self.out_q, self.stop_event)

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
        self.daq.start()
        self.daq_active = True

    @property
    def phi(self):
        return self.state.phi_read
       
    def generate_kernel(self):
        @self.backend.kernel
        def _moment_kernel(idx, phi_arr, obs_running_sum, count, order, langevin_time, last_meas):
            phi_idx = phi_arr[idx]
            obs = phi_idx**order
            obs_running_sum[idx] += obs
            count[idx] += 1
            last_meas[idx] = langevin_time[idx]
        return _moment_kernel
    
    def observe(self):
        # self.backend.parallel_loop(self.kernel, self.state.n_seeds, self.state.phi_read, self.obs_running_sum, self.count, self.order)
        self.cold = (self.state.langevin_time - self.last_meas) > self.state.dt_base/10
        # self.log(f"lt: {self.state.langevin_time}")
        # self.log(f"last meas: {self.last_meas}")
        # self.log(f"cold:{self.cold}")

        self.backend.act_parallel_loop(self.kernel, self.cold, self.state.n_seeds, self.state.phi_read, self.obs_running_sum, self.count, self.order, self.state.langevin_time, self.last_meas)
        
        # self.log(f"Field: {self.state.phi_read}")
        # self.log("Kernel executed: ")
        # self.log(f"Langevin times: {self.state.langevin_time}")
        # self.log(f"Last fluash: {self.last_flush}")
        # self.log(f"Old flush mask: {self.flush_mask}")

        if self.daq_active:
            mark_flush_mask(
                self.state.langevin_time, self.last_flush,
                self.flush_mask, self.daq.block_counts, self.delta_tau_block,
                self.num_blocks
            )
            self.log(f"New Flush Mask: {self.flush_mask}")

            if np.any(self.flush_mask):  # only flush if needed
                pack_block_means(
                    self.obs_running_sum, self.count,
                    self.state.langevin_time, self.last_flush,
                    self.flush_mask, self.block_means,
                )

                valid_idx = np.where(~np.isnan(self.block_means))[0]
                valid_vals = self.block_means[valid_idx]
                if valid_idx.shape[0] > 0:
                    self.log(f"flusing seed {valid_idx}")
                    self.out_q.put((valid_idx, valid_vals))

    def reset(self):
        self.obs_running_sum[:] = 0.0
        self.step_counter = 0
        # if self.daq_active:
        #     self.daq.block_means.clear()finish

    def finish(self):
        self.stop_event.set()
        self.daq.join()

    def log(self, message): log(self, self.name, message)