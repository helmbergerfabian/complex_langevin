from threading import Event, Thread
from queue import Queue, Empty

class DAQThread(Thread):
    def __init__(self, in_q: Queue, stop_event: Event):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.stop_event = stop_event
        self.block_means = []

    def run(self):
        print("[DAQ] Thread started.")
        while not self.stop_event.is_set():
            try:
                block = self.in_q.get(timeout=0.5)
            except Empty:
                continue
            self.process(block)
            self.in_q.task_done()

    def process(self, data_block):
        block_mean = data_block[0] / data_block[1]
        self.block_means.append(block_mean)