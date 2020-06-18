# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
import copy
from abc import abstractmethod

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Sampler(torch.optim.Optimizer, abc.ABC):
    """ The base class for samplers.

    """
   
    @abstractmethod
    def sample_params(self,  *args, **kwargs):
        pass

    @abstractmethod
    def expectation_params(self, *args, **kwargs):
        pass


        
