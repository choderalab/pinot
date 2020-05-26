#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Kernel(torch.nn.Module, abc.ABC):
    r""" A Gaussian Process Kernel that hosts parameters.


    """

    @abc.abstractmethod
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

