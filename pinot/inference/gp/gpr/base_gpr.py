#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math

# =============================================================================
# MODULE CLASSES
# =============================================================================
class GPR(torch.nn.Module, abc.ABC):
    """ Gaussian Process Regression.
    """
    def __init__(self, epsilon=1e-5):
        super(GPR, self).__init__()
        self.epsilon = epsilon

    @abc.abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def condition(self, *args, **kwargs):
        raise NotImplementedError

    def _perturb(self, k):
        """ Add small noise `epsilon` to the diagnol of kernels.
        Parameters
        ----------
        k : torch.tensor
            kernel to be perturbed.
        """

        # introduce noise along the diagnol
        noise = self.epsilon * torch.eye(
                *k.shape,
                device=k.device)

        return k + noise

    def _get_kernel_and_auxiliary_variables(
            self, x_tr, y_tr, x_te=None, sigma=1.0, epsilon=1e-5,
        ):

        # compute the kernels
        k_tr_tr = self._perturb(self.forward(x_tr, x_tr))

        if x_te is not None: # during test
            k_te_te = self._perturb(self.forward(x_te, x_te))
            k_te_tr = self._perturb(self.forward(x_te, x_tr))
            # k_tr_te = self.forward(x_tr, x_te)
            k_tr_te = k_te_tr.t() # save time

        else: # during train
            k_te_te = k_te_tr = k_tr_te = k_tr_tr

        # (batch_size_tr, batch_size_tr)
        k_plus_sigma = k_tr_tr + (sigma ** 2) * torch.eye(k_tr_tr.shape[0])

        # (batch_size_tr, batch_size_tr)
        l_low = torch.cholesky(k_plus_sigma)
        l_up = l_low.t()

        # (batch_size_tr. 1)
        l_low_over_y, _ = torch.triangular_solve(
            input=y_tr,
            A=l_low,
            upper=False)

        # (batch_size_tr, 1)
        alpha, _ = torch.triangular_solve(
            input=l_low_over_y,
            A=l_up,
            upper=True)

        return k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha