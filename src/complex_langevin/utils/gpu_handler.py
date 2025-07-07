import numpy as np
from numba import cuda # type: ignore 
from complex_langevin.simulation.state import SimState

from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numpy import ndarray

class GPU_handler:
    def __init__(self, simstate: SimState, exclude=None):
        if exclude is None:
            self.exclude = []
        else:
            self.exclude = exclude   

        self.simstate = simstate
        self.tensor_names = [
            attr_name for attr_name, attr in simstate.__dict__.items() 
            if isinstance(attr, np.ndarray)
            ]

    def to_device(self):
        """
        Transfer all tensors corresponding to tensor_data to the GPU and update the corresponding attributes.
        """
        for attr_name in self.tensor_names:
            if attr_name not in self.exclude:
                tensor: ndarray = getattr(self.simstate, attr_name) 
                device_tensor: DeviceNDArray = cuda.to_device(tensor)  
                setattr(self.simstate, attr_name, device_tensor)  

    def to_host(self):
        """
        Transfer all tensors corresponding to tensor_data from the GPU to the host and update the corresponding attributes.
        """
        for attr_name in self.tensor_names:
            if attr_name not in self.exclude:
                tensor = getattr(self.simstate, attr_name)
                if isinstance(tensor, DeviceNDArray):
                    host_tensor: ndarray = tensor.copy_to_host()
                    setattr(self.simstate, attr_name, host_tensor)
                else:
                    setattr(self.simstate, attr_name, tensor)