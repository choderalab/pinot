# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc

# =============================================================================
# MODULE CLASSES
# =============================================================================
class BaseKernel(torch.nn.Module, abc.ABC):
    """ A Gaussian Process Kernel that hosts parameters."""

    def __init__(self, epsilon=1e-6):
        super(BaseKernel, self).__init__()
        self.epsilon = epsilon

    @abc.abstractmethod
    def forward(self, x, *args, **kwargs):
        """ Forward pass.

        Parameters
        ----------
        x : `torch.Tensor`
            Kernel input.

        Returns
        -------
        k : `torch.Tensor`
            Kernel output.

        """
        raise NotImplementedError
