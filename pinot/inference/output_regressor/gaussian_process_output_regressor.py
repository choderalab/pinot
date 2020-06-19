#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math
from pinot.inference.output_regressors.base_output_regressor\
    import BaseOutputRegressor

# =============================================================================
# BASE CLASSES
# =============================================================================
class GaussianProcessOutputRegressor(BaseOutputRegressor):
    """ Gaussian Process Regression.

    """
    def __init__(self, epsilon=1e-5):
        super(GaussianProcessOutputRegressor, self).__init__()
        self.epsilon = epsilon

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

# =============================================================================
# MODULE CLASSES
# =============================================================================
class ExactGaussianProcessOuputRegressor(GaussianProcessOutputRegressor):
    """ Exact Gaussian Process.

    """
    def __init__(
            self,
            in_features,
            kernel=None,
            ):
        super(ExactGaussianProcessHead, self).__init__()

        if kernel is None:
            kernel = pinot.inference.kernels.rbf.RBF

        # put representation hidden units
        self.kernel = kernel(in_features)

    def _get_kernel_and_auxiliary_variables(
            self, x_tr, y_tr, x_te=None, sigma=1.0, epsilon=1e-5,
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
        k_plus_sigma = k_tr_tr + (sigma ** 2) * torch.eye(
            k_tr_tr.shape[0],
            device=k.device)

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

        return nll


class VariationalGaussianProcessOuputRegressor(GaussianProcessHead):
    """ Variational Gaussian Process.

    """
    def __init__(self, inducing_points, kernel=None):
        super(VariationalGaussianProcessOuputRegressor, self).__init__()

        class _VariationalGP(gpytorch.models.ApproximateGP):

            def __init__(inducing_points, kernel):

                if kernel is None:
                    kernel = gpytorch.kernels.RBFKernel()

                variational_distribution = gpytorch.variational\
                    .CholeskyVariationalDistribution(
                        inducing_points.size(-1))

                variational_strategy = gpytorch.variational\
                    .VariationalStrategy(
                        self,
                        inducing_points,
                        variational_distribution,
                        learn_inducing_points=True
                    )

                super().__init__(variaitonal_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    kernel)
                self.likelihood = likelihood

        self.gp = _VariationalGP(inducing_points, kernel=kernel)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.objective_function = gpytorch.mlls.VariationalELBO

    def condition(self, x):
        mean_x = self.gp.mean_module(x)
        covar_x = self.gp.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(
            mean_x,
            covar_x)

    def loss(self, x, y):
        distribution = self.condition(x)
        return -self.objective_function(distribution, y)
