# =============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
from pinot.inference.gp.kernels.kernel import Kernel

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Matern52(Kernel):
    r"""Matern52 kernel.

    $$

    k_\text{m52}(x, x') =
    (1 + \sqrt{5r ^ 2} + 5/3 r^2)
    \operatorname{exp}(-\sqrt{-5r ^ 2})

    $$

    """
    def __init__(self, scale=0.0, variance=0.0):
        super(Matern52, self).__init__()
        self.scale = torch.nn.Parameter(
                torch.tensor(scale))
        self.variance = torch.nn.Parameter(
                torch.tensor(variance))

    def distance(self, x, x_):
        return torch.norm(
                x[:, None, :] - x_[None, :, :],
                p=2,
                dim=2)

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
        distance_sq = distance ** 2

        # convariant matrix
        # (batch_size, batch_size)
        k = torch.exp(self.variance) * (
                1.0\
              + torch.sqrt(5.0 * distance_sq)\
              + (5.0 / 3.0) * distance_sq)\
              * torch.exp(-torch.sqrt(5.0 * distance_sq))

        return k
