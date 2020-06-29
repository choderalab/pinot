# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import abc

# =============================================================================
# BASE CLASSES
# =============================================================================
class BaseRegressor(torch.nn.Module):
    """Base class for `Head` object that translates latent representation
    `h` to a distribution.

    Methods
    -------
    condition : Forward pass to come up with predictive distribution.

    """

    def __init__(self):
        super(BaseRegressor, self).__init__()

    @abc.abstractmethod
    def condition(self, h, *args, **kwargs):
        """ Forward pass to come up with predictive distribution. """
        raise NotImplementedError
