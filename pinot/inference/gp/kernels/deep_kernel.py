#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
from pinot.inference.gp.kernels.kernel import Kernel

# =============================================================================
# MODULE CLASSES
# =============================================================================
class DeepKernel(Kernel):
    r""" A Gaussian Process Kernel with neural network representation.
    """
    def __init__(self, representation, base_kernel):
        super(DeepKernel, self).__init__()
        self.base_kernel = base_kernel

    def forward(self, h, h_=None):
        r""" Forward function that simply calls the base kernel
        from the representation.
        """
        if h_ is None:
            h_ = h

        k = self.base_kernel.forward(h, h_)

        return k