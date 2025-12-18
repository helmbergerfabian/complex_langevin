from numpy import zeros
from queue import Queue
from threading import Event

from complex_langevin.config import CL_COMPLEX, CL_INT, CL_REAL, log
from complex_langevin.compiler import backend
from complex_langevin.simulation.state import SimState
from complex_langevin.observables.daq import DAQThread


from numba import njit, prange
import numpy as np
import os
target = os.environ["MY_NUMBA_TARGET"]


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



# @njit(parallel=True)
# def pack_block_means(obs_sum, count, tau, tau_last_flush, flush_mask, block_means):
#     """
#     Based on a flush_mask, block_means are calculated and running sum is reset. 
#     """
#     for i in prange(obs_sum.shape[0]):
#         if flush_mask[i]:
#             block_means[i] = obs_sum[i] / count[i]
#             obs_sum[i] = 0.0
#             count[i] = 0
#             tau_last_flush[i] = tau[i]
#         else:
#             block_means[i] = np.nan 


############################
########ExperiMental########

# python mode
@backend.kernel
def extract_cold_indices_kernel(idx, idx_list, lt, last_meas, auto_corr, out_indices, out_count):
    _idx = idx_list[idx]
    if (lt[_idx] - last_meas[_idx]) > auto_corr:
        out_indices[out_count[0]] = _idx
        out_count[0] += 1

@backend.kernel
def pack_block_means(idx, cold_indices, obs_sum, count, tau, tau_last_flush, block_means):
    """
    block_means are calculated and running sum is reset. 
    """
    _idx = cold_indices[idx]
    block_means[_idx] = obs_sum[_idx] / count[_idx]
    obs_sum[_idx] = 0.0
    count[_idx] = 0
    tau_last_flush[_idx] = tau[_idx]

if backend.use_cuda: 
    from numba import cuda # type: ignore
    # @backend.kernel
    # def extract_cold_indices_kernel_1(idx, langevin_time, last_meas, dt_base, out_indices, out_count):
    #     if idx == 0:
    #         out_count[0] = 0
    #     cuda.syncthreads()

    #     if (langevin_time[idx] - last_meas[idx]) > (dt_base / 10):
    #         j = cuda.atomic.add(out_count, 0, 1)
    #         out_indices[j] = idx

    # @backend.kernel
    # def extract_cold_indices_kernel_2(idx, langevin_time, last_flush, delta_tau_block, out_indices, out_count):
    #     if idx == 0:
    #         out_count[0] = 0
    #     cuda.syncthreads()

    #     if (langevin_time[idx] - last_flush[idx]) > delta_tau_block:
    #         j = cuda.atomic.add(out_count, 0, 1)
    #         out_indices[j] = idx

    # @backend.kernel
    # def extract_cold_indices_kernel(idx, lt, last_action, lt_span, out_indices, out_count):
    #     if idx == 0:
    #         out_count[0] = 0
    #     cuda.syncthreads()

    #     if (lt[idx] - last_action[idx]) > lt_span:
    #         j = cuda.atomic.add(out_count, 0, 1)
    #         out_indices[j] = idx

    @backend.kernel
    def pack_block_means(idx, cold_indices, obs_sum, count, tau, tau_last_flush, block_means):
        """
        block_means are calculated and running sum is reset. 
        """
        _idx = cold_indices[idx]
        block_means[_idx] = obs_sum[_idx] / count[_idx]
        obs_sum[_idx] = 0.0
        count[_idx] = 0
        tau_last_flush[_idx] = tau[_idx]

############################
############################


class MomentObservable():
    def __init__(self, state: SimState, 
                 order: int, name = None, 
                 buffer_size = None, delta_tau_block = None, 
                 num_blocks = 5):
        import numpy as np
        self.name = name
        self.backend = backend
        self.state = state

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
        self.num_blocks = np.ones(self.state.n_seeds, dtype = CL_REAL) * num_blocks

        # self.meas_count = zeros(shape = state.n_seeds, dtype=CL_INT)
        self.kernel = self.generate_kernel()
        self.order = order
        self.out_q = Queue()
        self.daq_active = False

        # prepare DAQ
        self.stop_event = Event()
        self.daq = DAQThread(self.state.n_seeds, self.out_q, self.stop_event, max_blocks = num_blocks)

        self.cold_indices = np.zeros(self.state.n_seeds, dtype = CL_INT)
        self.cold_count = np.array([1], dtype = CL_INT)  # like a counter
        self.zero_buffer = np.array([0], dtype = CL_INT)
    
        self.cold_indices_2 = np.zeros(self.state.n_seeds, dtype = CL_INT)
        self.cold_count_2 = np.array([1], dtype = CL_INT)  # like a counter


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
       
    # def generate_kernel(self):
    #     @self.backend.kernel
    #     def _moment_kernel(idx, phi_arr, obs_running_sum, count, order, langevin_time, last_meas):
    #         phi_idx = phi_arr[idx]
    #         obs = phi_idx**order
    #         obs_running_sum[idx] += obs
    #         count[idx] += 1
    #         last_meas[idx] = langevin_time[idx]
    #     return _moment_kernel
    
    def generate_kernel(self):
        @self.backend.kernel
        def _moment_kernel(idx, idx_list, phi_arr, obs_running_sum, count, order, langevin_time, last_meas):
            _idx = idx_list[idx]
            phi_idx = phi_arr[_idx]
            obs = phi_idx**order
            obs_running_sum[_idx] += obs
            count[_idx] += 1
            last_meas[_idx] = langevin_time[_idx]
        return _moment_kernel
    
    def observe(self):

        # self.cold = (self.state.langevin_time - self.last_meas) > self.state.dt_base/10
        # self.backend.act_parallel_loop(self.kernel, self.cold, self.state.n_seeds, self.state.phi_read, self.obs_running_sum, self.count, self.order, self.state.langevin_time, self.last_meas)
        

        ############################
        ########ExperiMental########
        
        # python mode
        if target == "python" or target == "numba":
            # 1. write non-piled up indices into self.cold_indices and their count in self.cold_count. 
            # these will be observed and written into the block
            # Assume, that thermalization time is over already
            # example: [hot, cold, hot, hot, cold, cold] -> cold_indices = [1, 4, 5 ,0, 2, 3], cold_count = 2
            self.cold_count = self.zero_buffer.copy()
            backend.serial_loop(extract_cold_indices_kernel, self.state.alive_count[0], self.state.alive_idx_list, 
                                self.state.langevin_time, self.last_meas,
                                self.state.dt_base/2, self.cold_indices, self.cold_count)
            
            # 2. If there are cold seeds, observe those, the loop only goes up to self.cold_count. 
            # For those measured, the obs_running_sum, the last_meas and the count is updated 
            if self.cold_count > 0:
                self.backend.serial_loop(
                    self.kernel, self.cold_count[0],
                    self.cold_indices,
                    self.state.phi_read,
                    self.obs_running_sum,
                    self.count,
                    self.order,
                    self.state.langevin_time,
                    self.last_meas
                )
                # self.log(f"Observed: {self.cold_indices}, count: {self.cold_count[0]}")

            if self.daq_active:
            # 3. If the DAQ is active, copy the logic from 1., but use the autocorrelation time (self.delta_tau_block). The last flush into the DAQ 
            # has to be have been longer than the autocorrelation time.
                # 
                self.cold_count = self.zero_buffer.copy()
                backend.serial_loop(extract_cold_indices_kernel, self.state.alive_count[0], self.state.alive_idx_list, 
                                    self.state.langevin_time, self.last_flush, 
                                    self.delta_tau_block, self.cold_indices, self.cold_count)
                # self.log(f"DAQ is active, {self.cold_count}")
                
                # # 4. If there are cold seeds, calculate their block means
                if self.cold_count > 0:
                    # only flag those trajs, that have self.daq.block_counts < self.num_blocks
                    cold_count_copy = self.cold_count.copy()
                    self.cold_count = self.zero_buffer.copy()
                    flushy_indices = self.cold_indices.copy()
                    
                    # print("ASDASDASD")
                    backend.serial_loop(extract_cold_indices_kernel, cold_count_copy[0], flushy_indices, 
                                        self.num_blocks, self.daq.block_counts, 0.0, self.cold_indices, self.cold_count)
                    
                    # self.log(f"enough data on {self.cold_indices}, count: {self.cold_count}")
                    # self.log(f"traj is cold value: {self.obs_running_sum/self.count}")
                    # self.flush_mask = (self.state.langevin_time - self.last_flush>self.delta_tau_block) & (self.daq.block_counts < self.num_blocks)
                    
                    if self.cold_count > 0:
                        self.backend.serial_loop(
                            pack_block_means, self.cold_count[0],
                            self.cold_indices,
                            self.obs_running_sum,
                            self.count,
                            self.state.langevin_time,
                            self.last_flush,
                            self.block_means
                        )
                        
                        idxs = self.cold_indices[:self.cold_count[0]].copy()
                        vals = self.block_means[idxs].copy()
                        self.out_q.put((idxs, vals))

                        # self.out_q.put((self.cold_indices.copy(), self.block_means.copy(), self.cold_count[0]))

                    # self.state.alive_count = self.cold_count.copy()
                    # self.state.alive_idx_list = self.cold_indices.copy()

                    # self.log(f"Flushing blocks: {self.cold_indices}, count: {self.cold_count[0]}")
                    # self.log(f"Flush mask: {self.flush_mask}")
                        
                    # if np.any(self.flush_mask):  # only flush if needed
                        # pack_block_means(
                        #     self.obs_running_sum, self.count,
                        #     self.state.langevin_time, self.last_flush,
                        #     self.flush_mask, self.block_means,
                        # )

                        # valid_idx = np.where(~np.isnan(self.block_means))[0]
                        # valid_vals = self.block_means[valid_idx]
                        # if valid_idx.shape[0] > 0:
                            # self.log(f"flusing seed {valid_idx}")
                            # self.log(f"Flushing: {(self.cold_indices, self.block_means, self.cold_count[0])}")
                            # self.out_q.put((idx_list, flushed_means, flush_count))
                            # self.out_q.put((valid_idx, valid_vals))

            # those seeds that have gone through more than autocorrelation time are flushed. 

            # self.log(f"cold_count_host {self.cold_count}")
            # self.log(f"observing seeds {self.cold_indices}")
            
        # cold_count_host = self.cold_count.copy_to_host()[0]

        # if self.cold_count > 0:
        #     self.backend.parallel_loop(
        #         self.kernel, self.cold_count,
        #         self.cold_indices,
        #         self.state.phi_read,
        #         self.obs_running_sum,
        #         self.count,
        #         self.order,
        #         self.state.langevin_time,
        #         self.last_meas
        #     )

        #     self.log(f"cold_count_host {cold_count_host}")
        #     self.log(f"observing seeds {self.cold_indices.copy_to_host()}")
        # if self.daq_active:

        #     backend.parallel_loop(extract_cold_indices_kernel, self.state.n_seeds, self.state.langevin_time, self.last_flush, 
        #                       self.delta_tau_block, self.cold_indices, self.cold_count)
        #     cold_count_host = self.cold_count.copy_to_host()[0]
            
        #     if cold_count_host > 0:
        #             self.backend.parallel_loop(
        #                 pack_block_means, cold_count_host,
        #                 self.cold_indices,
        #                 self.obs_running_sum,
        #                 self.count,
        #                 self.state.langevin_time,
        #                 self.last_flush,
        #                 self.block_means
        #             )

        #             self.log(f"cold_count_host_2 {cold_count_host}")
        #             self.log(f"flushing seeds {self.cold_indices.copy_to_host()}")
                    # self.out_q.put((valid_idx, valid_vals))

        ############################
        ########ExperiMental########

        # if self.daq_active:
        #     # self.flush_mask = (self.state.langevin_time - self.last_flush>self.delta_tau_block) & (self.daq.block_counts < self.num_blocks)

        #     mark_flush_mask(
        #         self.state.langevin_time, self.last_flush,
        #         self.flush_mask, self.daq.block_counts, self.delta_tau_block,
        #         self.num_blocks
        #     )

        #     # self.log(f"New Flush Mask: {self.flush_mask}")

        #     if np.any(self.flush_mask):  # only flush if needed
        #         pack_block_means(
        #             self.obs_running_sum, self.count,
        #             self.state.langevin_time, self.last_flush,
        #             self.flush_mask, self.block_means,
        #         )

        #         valid_idx = np.where(~np.isnan(self.block_means))[0]
        #         valid_vals = self.block_means[valid_idx]
        #         if valid_idx.shape[0] > 0:
        #             # self.log(f"flusing seed {valid_idx}")
        #             self.out_q.put((valid_idx, valid_vals))

    def reset(self):
        self.obs_running_sum[:] = 0.0
        self.step_counter = 0
        # if self.daq_active:
        #     self.daq.block_means.clear()finish

    def finish(self):
        self.stop_event.set()
        self.daq.join()

    def log(self, message): log(self, self.name, message)