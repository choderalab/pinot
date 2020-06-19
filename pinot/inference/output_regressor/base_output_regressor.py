# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import abc

# =============================================================================
# BASE CLASSES
# =============================================================================
class BaseOutputRegressor(torch.nn.Module):
    """ Base class for `Head` object that translates latent representation
    `h` to a distribution.

    Methods
    -------

    """
    def __init__(self):
        super(BaseHead, self).__init__()

    @abc.abstractmethod
    def condition(self, h, *args, **kwargs):
        raise NotImplementedError
