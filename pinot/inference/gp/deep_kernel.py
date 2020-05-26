#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc

# =============================================================================
# MODULE CLASSES
# =============================================================================
class DeepKernel(esp.gp.Kernel):
    r""" A Gaussian Process Kernel with neural network representation.


    """
    def __init__(self, representation, base_kernel):
        super(DeepKernel, self).__init__()
        self.representation = representation
        self.base_kernel = base_kernel

    def forward(self, x, x_=None):
        r""" Forward function that simply calls the base kernel
        after representation.

        """

        if x_ is None:
            x_ = x

        k = self.base_kernel.forward(
                self.representation(x),
                self.representation(x_))

        return k
        
        
