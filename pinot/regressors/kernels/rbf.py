# =============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
from pinot.regressors.kernels.base_kernel import BaseKernel

# =============================================================================
# MODULE CLASSES
# =============================================================================
class RBF(BaseKernel):
    r""" A Gaussian Process Kernel that hosts parameters.

    Note
    ----
    l could be either of shape 1 or hidden dim
    """

    def __init__(self, in_features, scale=0.0, variance=0.0, ard=True):

        super(RBF, self).__init__()

        if ard is True:
            self.scale = torch.nn.Parameter(scale * torch.ones(in_features))

        self.variance = torch.nn.Parameter(torch.tensor(variance))

    def distance(self, x, x_):
        return torch.norm(x[:, None, :] - x_[None, :, :], p=2, dim=2)

    def forward(self, x, x_=None):
        # replicate x if there's no x_
        if x_ is None:
            x_ = x

        # for now, only allow two dimension
        assert x.dim() == 2
        assert x_.dim() == 2

        x = x * torch.exp(self.scale)
        x_ = x_ * torch.exp(self.scale)

        # (batch_size, batch_size)
        distance = self.distance(x, x_)

        # convariant matrix
        # (batch_size, batch_size)
        k = torch.exp(self.variance) * torch.exp(-0.5 * distance)

        return k
