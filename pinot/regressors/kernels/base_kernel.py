# =============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math

# =============================================================================
# MODULE CLASSES
# =============================================================================
class BaseKernel(torch.nn.Module, abc.ABC):
    r""" A Gaussian Process Kernel that hosts parameters.

    Parameters
    ----------
    epsilon : noise
        noise added to kernel.

    """

    def __init__(self, epsilon=1e-6):
        super(BaseKernel, self).__init__()
        self.epsilon = epsilon

    @abc.abstractmethod
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError
