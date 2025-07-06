### complex_langevin/models/base.py

from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def drift(self, x):
        pass
    
    @abstractmethod
    def action(self, x):
        pass