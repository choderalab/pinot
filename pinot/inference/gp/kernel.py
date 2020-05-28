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
class Kernel(torch.nn.Module, abc.ABC):
    r""" A Gaussian Process Kernel that hosts parameters.


    """

    @abc.abstractmethod
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def _get_kernel_and_auxiliary_variables(
            self, x_tr, y_tr, x_te=None, sigma=1.0, epsilon=1e-5,
        ):

        # compute the kernels
        k_tr_tr = self.forward(x_tr, x_tr)

        if x_te is not None: # during test
            k_te_te = self.forward(x_te, x_te)
            k_te_tr = self.forward(x_te, x_tr)
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

    def loss(self, x_tr, y_tr, sigma=1.0):
        """ Compute the loss.

        Parameters
        ----------
        x_tr : torch.tensor, shape=(batch_size, ...)
            training data.
        y_tr : torch.tensor, shape=(batch_size, 1)
            training data measurement.

        """

        # get the parameters
        k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha\
            = self._get_kernel_and_auxiliary_variables(x_tr, y_tr, sigma=sigma)

        # we return the exact nll with constant
        nll = 0.5 * (y_tr.t() @ alpha) + torch.trace(l_low)\
            + 0.5 * y_tr.shape[0] * math.log(2.0 * math.pi)

        return nll

    def mean_and_variance(self, x_tr, y_tr, x_te=None, sigma=1.0):
        # get parameters
        k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha\
            = self._get_kernel_and_auxiliary_variables(
                x_tr, y_tr, x_te, sigma=sigma)

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

        return mean, variance

    def condition(self, x_tr, y_tr, x_te=None, sigma=1.0):
        r""" Calculate the predictive distribution given `x_te`.

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

        # get parameters
        k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha\
            = self._get_kernel_and_auxiliary_variables(
                x_tr, y_tr, x_te, sigma=sigma)

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

        # construct noise predictive distribution
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            mean.flatten(),
            variance)

        return distribution
