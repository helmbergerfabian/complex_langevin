from threading import Event, Thread
from queue import Queue, Empty

from complex_langevin.config import log, CL_INT, CL_REAL

import numpy as np
def jackknife_1d(data):
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    full_mean = np.mean(data)
    jk_samples = np.array([
        np.mean(np.delete(data, i)) for i in range(n)
    ])
    mean = np.mean(jk_samples)
    sem = np.sqrt((n - 1) * np.mean((jk_samples - mean) ** 2))
    return mean, sem

class DAQThread(Thread):
    def __init__(self, n_seeds, in_q: Queue, stop_event: Event, max_blocks = None):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.stop_event = stop_event

        self.max_blocks = max_blocks or 50  # configurable
        self.block_means = np.full((n_seeds, self.max_blocks), np.nan+1j*np.nan)
        self.block_counts = np.zeros(n_seeds, dtype=CL_REAL)
        self.log_file = None

    def run(self):
        self.log("Thread started")
        while not self.stop_event.is_set():
            try:
                block = self.in_q.get(timeout=0.5)
                # self.log(f"Received new data, seed: {block}")
                # self.log(block)
            except Empty:
                continue
            self.process(block)
            self.in_q.task_done()

    
    def compute_jackknife_stats(self):
        n_seeds, n_blocks_max = self.block_means.shape
        means = np.full(n_seeds, np.nan)
        sems = np.full(n_seeds, np.nan)

        for seed in range(n_seeds):
            count = self.block_counts[seed]
            if count >= 2:
                data = self.block_means[int(seed), :int(count)]
                mean, sem = jackknife_1d(data)
                means[seed] = mean
                sems[seed] = sem

        return means, sems
    
    def process(self, data):
        indices, values = data
        for seed, val in zip(indices, values):
            k = int(self.block_counts[seed])
            if k < self.block_means.shape[1]:
                self.block_means[seed, k] = val
                self.block_counts[seed] += 1

    
    # def process(self, data):
    #     cold_indices, new_mean, cold_count = data
    #     for idx in range(cold_count):
    #         _idx = cold_indices[idx]
    #         k = CL_INT(self.block_counts[_idx])
    #         if k < self.block_means.shape[1]:
    #             self.block_means[_idx, k] = new_mean[_idx]
    #             self.block_counts[_idx] += 1
            # else:
            #     print(f"[DAQ] Warning: block buffer full for seed {_idx}")
        # indices, values = data  # 1D arrays
        # for seed, new_mean in zip(indices, values):
        #     k = self.block_counts[seed]
        #     if k < self.block_means.shape[1]:
        #         self.block_means[seed, k] = new_mean
        #         self.block_counts[seed] += 1
        #     else:
        #         print(f"[DAQ] Warning: block buffer full for seed {seed}")
    
    def log(self, message): log(self, "DAQ", message)