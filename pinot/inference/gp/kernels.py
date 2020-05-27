#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
from pinot.inference.gp.kernel import Kernel

# =============================================================================
# MODULE CLASSES
# =============================================================================
class RBF(Kernel):
    r""" A Gaussian Process Kernel that hosts parameters.


    """
    def __init__(self, l=1.0):
        super(RBF, self).__init__()
        self.l = l

    def distance(self, x, x_):
        return torch.norm(
                x[:, None, :] - x_[None, :, :],
                p=2,
                dim=2)

    def forward(self, x, x_=None, l=None):
        # replicate x if there's no x_
        if x_ is None:
            x_ = x

        # set l to default if there's no l
        if l is None:
            l = self.l

        # for now, only allow two dimension
        assert x.dim() == 2
        assert x_.dim() == 2

        # (batch_size, batch_size)
        distance = self.distance(x, x_)

        # convariant matrix
        # (batch_size, batch_size)
        k = torch.exp(-0.5 * distance / (l ** 2))

        return k
