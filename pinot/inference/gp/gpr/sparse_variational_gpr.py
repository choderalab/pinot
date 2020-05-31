#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math
from pinot.inference.gp.gpr.base_gpr import GPR

# =============================================================================
# MODULE CLASSES
# =============================================================================
class SVGPR(GPR):
    """ Sparse Variational Gaussian Process regression.

    """
    def __init__(self, kernel, n_inducing_points):
        super(VGPR, self).__init__()
        self.kernel = kernel

        # NOTE:
        # in this version we don't support multitask yet
        initializer_std=0.1,
