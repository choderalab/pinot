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
class ExactGPR(GPR):
    """ Exact Gaussian process regression.
    """
    def __init__(self, kernel, log_sigma=-3.0):
        super(ExactGPR, self).__init__()
        self.kernel = kernel
        
        self.log_sigma = torch.nn.Parameter(
                torch.tensor(
                    log_sigma))


    def _get_kernel_and_auxiliary_variables(
            self, x_tr, y_tr, x_te=None,
        ):
        
        # grab sigma
        sigma = torch.exp(self.log_sigma)

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
        k_plus_sigma = k_tr_tr + torch.exp(self.log_sigma) * torch.eye(
                k_tr_tr.shape[0],
                device=k_tr_tr.device)

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
        r""" Compute the loss.
        Note
        ----
        Defined to be negative Gaussian likelihood.
        Parameters
        ----------
        x_tr : torch.tensor, shape=(batch_size, ...)
            training data.
        y_tr : torch.tensor, shape=(batch_size, 1)
            training data measurement.
        """
        # point data to object
        self._x_tr = x_tr
        self._y_tr = y_tr

        # get the parameters
        k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha\
            = self._get_kernel_and_auxiliary_variables(x_tr, y_tr)

        # we return the exact nll with constant
        nll = 0.5 * (y_tr.t() @ alpha) + torch.trace(l_low)\
            + 0.5 * y_tr.shape[0] * math.log(2.0 * math.pi)

        print(self.log_sigma)

        return nll

    def condition(self, x_te, x_tr=None, y_tr=None, sampler=None):
        r""" Calculate the predictive distribution given `x_te`.
        Note
        ----
        Here we allow the speicifaction of sampler but won't actually
        use it here.
        Parameters
        ----------
        x_tr : torch.tensor, shape=(batch_size, ...)
            training data.
        y_tr : torch.tensor, shape=(batch_size, 1)
            training data measurement.
        x_te : torch.tensor, shape=(batch_size, ...)
            test data.
        sigma : float or torch.tensor, shape=(), default=1.0
            noise parameter.
        """
        if x_tr is None:
            x_tr = self._x_tr
        if y_tr is None:
            y_tr = self._y_tr

        # get parameters
        k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha\
            = self._get_kernel_and_auxiliary_variables(
                x_tr, y_tr, x_te)

        # compute mean
        # (batch_size_te, 1)
        mean = k_te_tr @ alpha

        # (batch_size_tr, batch_size_te)
        v, _ = torch.triangular_solve(
            input=k_tr_te,
            A=l_low,
            upper=False)

        # (batch_size_te, batch_size_te)
        variance = k_te_te - v.t() @ v

        # ensure symetric
        variance = 0.5 * (variance + variance.t())

        # $ p(y|X) = \int p(y|f)p(f|x) df $
        # variance += torch.exp(self.log_sigma) * torch.eye(
        #         *variance.shape,
        #         device=variance.device)

        # construct noise predictive distribution
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            mean.flatten(),
            variance)

        return distribution