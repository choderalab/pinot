# =============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math
from pinot.regressors.base_regressor import BaseRegressor
import gpytorch

# =============================================================================
# BASE CLASSES
# =============================================================================
class GaussianProcessRegressor(BaseRegressor):
    """ Gaussian Process Regression.

    """

    def __init__(self, epsilon=1e-5):
        super(GaussianProcessRegressor, self).__init__()
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
        noise = self.epsilon * torch.eye(*k.shape, device=k.device)

        return k + noise


# =============================================================================
# MODULE CLASSES
# =============================================================================
class ExactGaussianProcessRegressor(GaussianProcessRegressor):
    """ Exact Gaussian Process.

    """

    def __init__(
            self,
            in_features,
            kernel=None,
            ):
        super(ExactGaussianProcessOutputRegressor, self).__init__()

        if kernel is None:
            kernel = pinot.regressors.kernels.rbf.RBF

        # point unintialized class to self
        self.kernel_cls = kernel

        # put representation hidden units
        self.kernel = kernel(in_features)

        self.in_features = in_features

    def _get_kernel_and_auxiliary_variables(
        self, x_tr, y_tr, x_te=None, sigma=1.0, epsilon=1e-5,
    ):

        # compute the kernels
        k_tr_tr = self._perturb(self.kernel.forward(x_tr, x_tr))

        if x_te is not None:  # during test
            k_te_te = self._perturb(self.kernel.forward(x_te, x_te))
            k_te_tr = self._perturb(self.kernel.forward(x_te, x_tr))
            # k_tr_te = self.forward(x_tr, x_te)
            k_tr_te = k_te_tr.t()  # save time

        else:  # during train
            k_te_te = k_te_tr = k_tr_te = k_tr_tr

        # (batch_size_tr, batch_size_tr)
        k_plus_sigma = k_tr_tr + (sigma ** 2) * torch.eye(
            k_tr_tr.shape[0], device=k_tr_tr.device
        )

        # (batch_size_tr, batch_size_tr)
        l_low = torch.cholesky(k_plus_sigma)
        l_up = l_low.t()

        # (batch_size_tr. 1)
        l_low_over_y, _ = torch.triangular_solve(
            input=y_tr, A=l_low, upper=False
        )

        # (batch_size_tr, 1)
        alpha, _ = torch.triangular_solve(
            input=l_low_over_y, A=l_up, upper=True
        )

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
        (
            k_tr_tr,
            k_te_te,
            k_te_tr,
            k_tr_te,
            l_low,
            alpha,
        ) = self._get_kernel_and_auxiliary_variables(x_tr, y_tr, x_te)

        # compute mean
        # (batch_size_te, 1)
        mean = k_te_tr @ alpha

        # (batch_size_tr, batch_size_te)
        v, _ = torch.triangular_solve(input=k_tr_te, A=l_low, upper=False)

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
            mean.flatten(), variance
        )

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
        (
            k_tr_tr,
            k_te_te,
            k_te_tr,
            k_tr_te,
            l_low,
            alpha,
        ) = self._get_kernel_and_auxiliary_variables(x_tr, y_tr)

        # we return the exact nll with constant
        nll = (
            0.5 * (y_tr.t() @ alpha)
            + torch.trace(l_low)
            + 0.5 * y_tr.shape[0] * math.log(2.0 * math.pi)
        )

        return nll



class VariationalGaussianProcessRegressor(object):
    """
    """

    def __init__(self,
                 in_features,
                 n_inducing_points=100,
                 inducing_points_boundary=1.0,
                 num_data=1,
                 kernel=None):
        super(VariationalGaussianProcessRegressor, self).__init__()

        # construct inducing points
        inducing_points = torch.distributions.uniform.Uniform(
            -1
            * inducing_points_boundary
            * torch.ones(n_inducing_points, in_features),
            1
            * inducing_points_boundary
            * torch.ones(n_inducing_points, in_features),
        ).sample()

        class _GaussianProcessLayer(gpytorch.models.ApproximateGP):
            def __init__(self,
                         inducing_points,
                         kernel=None
                        ):
                if kernel is None:
                    kernel = gpytorch.kernels.RBFKernel()

                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                    num_inducing_points=inducing_points.size(0))

                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                )
                super().__init__(variational_strategy)

                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)


        self.output_regressor = _GaussianProcessLayer(inducing_points,
                                                     kernel=kernel)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood,
                                            self.output_regressor,
                                            num_data=num_data)

    def forward(self, x):
        distribution = self.output_regressor(x)
        return distribution

    def parameters(self):
        return list(self.output_regressor.hyperparameters())\
              +list(self.output_regressor.variational_parameters())\
              +list(self.likelihood.parameters())

    def eval(self):
        self.output_regressor.eval()
        self.likelihood.eval()

    def train(self):
        self.output_regressor.train()
        self.likelihood.train()

    def condition(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def loss(self, x, y):
        distribution = self.forward(x)
        return -self.mll(distribution, y)

    def to(self, device):
        self.output_regressor = self.output_regressor.to(device)
        self.likelihood = self.likelihood.to(device)
        return self
