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
class BaseSampler(torch.optim.Optimizer, abc.ABC):
    """ The base class for samplers.

    """

    def __init__(self):
        super(BaseSampler, self).__init__()

    @abstractmethod
    def sample_params(self, *args, **kwargs):
        pass

    @abstractmethod
    def expectation_params(self, *args, **kwargs):
        pass
