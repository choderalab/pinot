# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
from abc import abstractmethod

# =============================================================================
# MODULE CLASSES
# =============================================================================
class BaseSampler(torch.optim.Optimizer, abc.ABC):
    """The base class for samplers."""

    def __init__(self, *args, **kwargs):
        super(BaseSampler, self).__init__(*args, **kwargs)

    @abstractmethod
    def sample_params(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args :

        **kwargs :


        Returns
        -------

        """
        pass

    @abstractmethod
    def expectation_params(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args :

        **kwargs :


        Returns
        -------

        """
        pass
