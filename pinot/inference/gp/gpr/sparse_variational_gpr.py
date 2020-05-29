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
class VGPR(GPR):
    """ Variational Gaussian Process regression.

    """
    def __init__(self, kernel, n_training_samples,
            mu_init_std=1e-3, sigma_init_std=1e-3):
        super(VGPR, self).__init__()
        self.kernel = kernel

        # introduce variational variables mu and sigma
        self.mu = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(n_training_samples, ),
                scale=mu_init_std*torch.ones(n_training_samples, )))

        self.sigma = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(n_training_samples, n_training_samples),
                scale=sigma_init_std*torch.ones(
                    n_training_samples, n_training_samples)))

    def _get_kernel_and_auxiliary_variables(
            self, x_tr, y_tr, x_te=None, sigma=1.0,
        ):

        # compute the kernels
        k_tr_tr = self._perturb(self.kernel.forward(x_tr, x_tr))

        if x_te is not None: # during test
            k_te_te = self._perturb(self.kernel.forward(x_te, x_te))
            k_te_tr = self._perturb(self.kernel.forward(x_te, x_tr))
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

    def loss(self, x_tr, y_tr):
        """ ELBO.

        """
        # get the kernels and so on
        k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha = self.\
            _get_kernel_and_auxiliary_variables(
                x_tr, y_tr)

        raise NotImplementedError
