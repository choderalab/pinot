# =============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math
from pinot.regressors.base_regressor import BaseRegressor

# import gpytorch
import numpy as np


# =============================================================================
# BASE CLASSES
# =============================================================================
class GaussianProcessRegressor(BaseRegressor):
    """Gaussian Process Regression."""

    def __init__(self, epsilon=1e-5):
        super(GaussianProcessRegressor, self).__init__()
        self.epsilon = epsilon

    @abc.abstractmethod
    def condition(self, *args, **kwargs):
        """ Forward pass to come up with predictive distribution. """
        raise NotImplementedError

    def _perturb(self, k):
        """Add small noise `epsilon` to the diagonal of covariant matrix.

        Parameters
        ----------
        k : `torch.Tensor`, `shape=(n_data_points, n_data_points)`
            Covariant matrix.

        Returns
        -------
        k : `torch.Tensor`, `shape=(n_data_points, n_data_points)`
            Preturbed covariant matrix.

        """
        # introduce noise along the diagonal
        noise = self.epsilon * torch.eye(*k.shape, device=k.device)

        return k + noise


# =============================================================================
# MODULE CLASSES
# =============================================================================
class ExactGaussianProcessRegressor(GaussianProcessRegressor):
    r""" Exact Gaussian Process.

    Parameters
    ----------
    in_features : `int`
        Input features on the latent space.

    kernel : `pinot.regressors.kernels.Kernel`
        Kernel used for Gaussian process.

    log_sigma : `float`
        ..math:: \log\sigma


    """

    def __init__(self, in_features, kernel=None, log_sigma=-3.0):
        super(ExactGaussianProcessRegressor, self).__init__()

        if kernel is None:
            kernel = pinot.regressors.kernels.rbf.RBF

        # point unintialized class to self
        self.kernel_cls = kernel

        # put representation hidden units
        self.kernel = kernel(in_features)

        self.in_features = in_features

        self.log_sigma = torch.nn.Parameter(torch.tensor(log_sigma))

    def _get_kernel_and_auxiliary_variables(
        self, x_tr, y_tr, x_te=None,
    ):
        """ Get kernel and auxiliary variables for forward pass. """

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
        k_plus_sigma = k_tr_tr + torch.exp(self.log_sigma) * torch.eye(
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

    def loss(self, x_tr, y_tr, *args, **kwargs):
        r""" Compute the loss.

        Note
        ----
        Defined to be negative Gaussian likelihood.

        Parameters
        ----------
        x_tr : `torch.Tensor`, `shape=(n_training_data, hidden_dimension)`
            Input of training data.

        y_tr : `torch.Tensor`, `shape=(n_training_data, 1)`
            Target of training data.


        Returns
        -------
        nll : `torch.Tensor`, `shape=(,)`
            Negative log likelihood.

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

    def condition(self, x_te, *args, x_tr=None, y_tr=None, **kwargs):
        r""" Calculate the predictive distribution given `x_te`.

        Note
        ----
        Here we allow the speicifaction of sampler but won't actually
        use it here in this version.

        Parameters
        ----------
        x_te : `torch.Tensor`, `shape=(n_te, hidden_dimension)`
            Test input.

        x_tr : `torch.Tensor`, `shape=(n_tr, hidden_dimension)`
             (Default value = None)
             Training input.

        y_tr : `torch.Tensor`, `shape=(n_tr, 1)`
             (Default value = None)
             Test input.

        sampler : `torch.optim.Optimizer` or `pinot.Sampler`
             (Default value = None)
             Sampler.

        Returns
        -------
        distribution : `torch.distributions.Distribution`
            Predictive distribution.

        """

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


class VariationalGaussianProcessRegressor(GaussianProcessRegressor):
    """Sparse variational Gaussian proces.

    Parameters
    ----------
    in_features : `int`
        Input features on the latent space.

    kernel : `pinot.regressors.kernels.Kernel`
        Kernel used for Gaussian process.

    log_sigma : `float`
        ..math:: \log\sigma


    """

    def __init__(
        self,
        in_features,
        kernel=None,
        log_sigma=-3.0,
        n_inducing_points=100,
        mu_initializer_std=0.1,
        sigma_initializer_value=-2.0,
        kl_loss_scaling=1e-2,
        grid_boundary=1.0,
    ):
        super(VariationalGaussianProcessRegressor, self).__init__()
        if kernel is None:
            kernel = pinot.regressors.kernels.rbf.RBF

        # point unintialized class to self
        self.kernel_cls = kernel

        # put representation hidden units
        self.kernel = kernel(in_features)

        self.in_features = in_features

        self.log_sigma = torch.nn.Parameter(torch.tensor(log_sigma))

        # uniform distribution within boundary
        # (n_inducing_points, hidden_dimension)
        self.x_tr = torch.nn.Parameter(
            torch.distributions.uniform.Uniform(
                -1.0
                * grid_boundary
                * torch.ones(n_inducing_points, in_features),
                1.0
                * grid_boundary
                * torch.ones(n_inducing_points, in_features),
            ).sample()
        )

        # variational mean for inducing points value
        # (n_inducing_points, 1)
        self.y_tr_mu = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(n_inducing_points, 1),
                scale=mu_initializer_std * torch.ones(n_inducing_points, 1),
            ).sample()
        )

        # to ensure lower cholesky
        self.y_tr_sigma_diag = torch.nn.Parameter(
            sigma_initializer_value * torch.ones(n_inducing_points)
        )

        # (0.5 * n_inducing_points * (n_inducing_points - 1), )
        self.y_tr_sigma_tril = torch.nn.Parameter(
            torch.zeros(
                int(n_inducing_points * (n_inducing_points - 1) * 0.5)
            )
        )

        self.n_inducing_points = n_inducing_points

        self.kl_loss_scaling = kl_loss_scaling

    def _y_tr_sigma(self):
        """ Getting the covariance matrix for variational training input."""
        # embed diagnoal of sigma
        y_tr_diag = torch.diag_embed(torch.exp(self.y_tr_sigma_diag))

        # (n_inducing_points, n_inducing_points)
        mask = torch.gt(
            torch.range(0, self.y_tr_sigma_diag.shape[0] - 1)[:, None],
            torch.range(0, self.y_tr_sigma_diag.shape[0] - 1)[None, :],
        )

        mask = mask.to(device=y_tr_diag.device)

        # (n_inducing_points, n_inducing_points)
        y_tr_sigma = y_tr_diag.masked_scatter(mask, self.y_tr_sigma_tril)

        return y_tr_sigma

    def _k_tr_tr(self):
        return self._perturb(self.kernel(self.x_tr))

    def _k_tr_te(self, x_te):
        return self._perturb(self.kernel(self.x_tr, x_te))

    def _k_te_tr(self, x_te):
        return self._k_tr_te(x_te).t()

    def _k_te_te(self, x_te):
        return self._perturb(self.kernel(x_te))

    def forward(self, x_te):
        """Forward pass.

        Parameters
        ----------
        x_te : `torch.Tensor`, `(x_te, hidden_dimension)`
            Test input.

        Returns
        -------
        predictive_mean : `torch.Tensor`, `(x_te, )`
            Predictive mean.

        predictive_cov : `torch.Tensor`, `(x_te, x_te)`
            Preditive covariance.

        """

        # get the kernels
        k_tr_tr, k_tr_te, k_te_tr, k_te_te = (
            self._k_tr_tr(),
            self._k_tr_te(x_te),
            self._k_te_tr(x_te),
            self._k_te_te(x_te),
        )

        # (n_tr, n_tr)
        l_k_tr_tr = torch.cholesky(k_tr_tr, upper=False)

        # (n_tr, 1)
        l_k_tr_tr_inv_mu, _ = torch.triangular_solve(
            input=self.y_tr_mu, A=l_k_tr_tr, upper=False
        )

        # (n_tr, n_tr)
        l_k_tr_tr_inv_sigma, _ = torch.triangular_solve(
            input=self._y_tr_sigma(), A=l_k_tr_tr, upper=False
        )

        # (n_tr, te)
        l_k_tr_tr_inv_k_tr_te, _ = torch.triangular_solve(
            input=k_tr_te, A=l_k_tr_tr, upper=False
        )

        # (n_te, 1)
        predictive_mean = l_k_tr_tr_inv_k_tr_te.t() @ l_k_tr_tr_inv_mu

        # (n_tr, n_te)
        k_tr_tr_inv_k_tr_te = (
            l_k_tr_tr_inv_k_tr_te.t() @ l_k_tr_tr_inv_k_tr_te
        )

        # (n_te, n_te)
        l_k_tr_tr_inv_sigma_t_at_l_k_tr_tr_inv_k_tr_te = (
            l_k_tr_tr_inv_sigma.t() @ l_k_tr_tr_inv_k_tr_te
        )

        # (n_te, n_te)
        compose_l_k_tr_tr_inv_sigma_t_at_l_k_tr_tr_inv_k_tr_te = (
            l_k_tr_tr_inv_sigma_t_at_l_k_tr_tr_inv_k_tr_te.t()
            @ l_k_tr_tr_inv_sigma_t_at_l_k_tr_tr_inv_k_tr_te
        )

        # (n_te, n_te)
        predictive_cov = (
            k_te_te
            - k_tr_tr_inv_k_tr_te
            + compose_l_k_tr_tr_inv_sigma_t_at_l_k_tr_tr_inv_k_tr_te
        )

        predictive_cov += torch.exp(self.log_sigma) * torch.eye(
            k_te_te.shape[0], device=k_te_te.device
        )

        return predictive_mean, predictive_cov

    def condition(self, x_te, *args, **kwargs):
        """

        Parameters
        ----------
        x_te :

        sampler :
             (Default value = None)

        Returns
        -------

        """
        

        predictive_mean, predictive_cov = self.forward(x_te)

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=predictive_mean.flatten(), covariance_matrix=predictive_cov
        )

        return distribution

    def loss(self, x_te, y_te, *args, **kwargs):
        """ Loss function.

        Parameters
        ----------
        x_te : `torch.Tensor`, `shape=(n_te, hidden_dimension)`
            Training input.

        y_te : `torch.Tensor`, `shape=(n_te, )`
            Training target.


        Returns
        -------
        loss : `torch.Tensor`, `shape=(,)`
            Loss function.

        """
        # define prior
        prior_mean = torch.zeros(self.n_inducing_points, 1)
        prior_tril = self._k_tr_tr().cholesky()

        prior_tril = prior_tril.to(device=x_te.device)
        prior_mean = prior_mean.to(device=x_te.device)

        distribution = self.condition(x_te)

        nll = -distribution.log_prob(y_te.flatten()).mean()

        log_p_u = self.exp_log_full_gaussian(
            self.y_tr_mu, self._y_tr_sigma(), prior_mean, prior_tril
        )

        log_q_u = -self.entropy_full_gaussian(
            self.y_tr_mu, self._y_tr_sigma()
        )

        loss = nll + self.kl_loss_scaling * (log_q_u - log_p_u)

        # import pdb; pdb.set_trace()
        return loss

    @staticmethod
    def exp_log_full_gaussian(mean1, tril1, mean2, tril2):
        const_term = -0.5 * mean1.shape[0] * np.log(2 * np.pi * np.exp(1))
        log_det_prior = -torch.sum(torch.log(torch.diag(tril2)))
        LpiLq = torch.triangular_solve(
            tril1.double(), tril2.double(), upper=False
        )[0]
        trace_term = -0.5 * torch.sum(LpiLq ** 2)
        mu_diff = mean1 - mean2
        quad_solve = torch.triangular_solve(
            mu_diff.double(), tril2.double(), upper=False
        )[0]
        quad_term = -0.5 * torch.sum(quad_solve ** 2)
        res = const_term + log_det_prior + trace_term + quad_term
        return res

    @staticmethod
    def entropy_full_gaussian(mean1, tril1):
        const_term = 0.5 * mean1.shape[0] * np.log(2 * np.pi * np.exp(1))
        log_det_prior = torch.sum(torch.log(torch.diag(tril1)))
        return log_det_prior + const_term


class BiophysicalVariationalGaussianProcessRegressor(
    VariationalGaussianProcessRegressor
):
    def __init__(
        self,
        in_features,
        kernel=None,
        log_sigma=-3.0,
        n_inducing_points=100,
        mu_initializer_std=0.1,
        sigma_initializer_value=-2.0,
        kl_loss_scaling=1e-2,
        grid_boundary=1.0,
    ):
        super(VariationalGaussianProcessRegressor, self).__init__()
        if kernel is None:
            kernel = pinot.regressors.kernels.rbf.RBF

        # point unintialized class to self
        self.kernel_cls = kernel

        # put representation hidden units
        self.kernel = kernel(in_features)

        self.in_features = in_features

        self.log_sigma = torch.nn.Parameter(torch.tensor(log_sigma))

        # uniform distribution within boundary
        # (n_inducing_points, hidden_dimension)
        self.x_tr = torch.nn.Parameter(
            torch.distributions.uniform.Uniform(
                -1.0
                * grid_boundary
                * torch.ones(n_inducing_points, in_features),
                1.0
                * grid_boundary
                * torch.ones(n_inducing_points, in_features),
            ).sample()
        )

        # variational mean for inducing points value
        # (n_inducing_points, 1)
        self.y_tr_mu = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(n_inducing_points, 1),
                scale=mu_initializer_std * torch.ones(n_inducing_points, 1),
            ).sample()
        )

        # to ensure lower cholesky
        self.y_tr_sigma_diag = torch.nn.Parameter(
            sigma_initializer_value * torch.ones(n_inducing_points)
        )

        # (0.5 * n_inducing_points * (n_inducing_points - 1), )
        self.y_tr_sigma_tril = torch.nn.Parameter(
            torch.zeros(
                int(n_inducing_points * (n_inducing_points - 1) * 0.5)
            )
        )

        self.n_inducing_points = n_inducing_points

        self.kl_loss_scaling = kl_loss_scaling

        self.log_sigma_measurement = torch.nn.Parameter(torch.zeros(1))

    def fractional_saturation(self, func_value=None, test_ligand_concentration=None):
        """
        we pass units in molar
        test_ligand_concentration : tensor(1) float32 in 1M units
        returns: fractional saturation between 0 and 1
        """
        return 1.0 / (1.0 + torch.exp(func_value) / test_ligand_concentration)

    def condition(
        self, x_te, test_ligand_concentration=None, output='measurement',
        *args, **kwargs
    ):
        """

        Parameters
        ----------
        x_te :

        sampler :
             (Default value = None)

        output : str
            Either 'measurement' or 'delta_g'
            The type of quantities to construct distribution on.

        Returns
        -------

        """
        assert isinstance(output, str)

        distribution_delta_g = self._condition_delta_g(x_te)

        if output == 'delta_g':
            return distribution_delta_g

        elif output == 'measurement':
            f_sample = self._sample_f(distribution_delta_g)

            distribution_measurement = self._condition_measurement(
                f_sample,
                test_ligand_concentration=test_ligand_concentration
            )
            return distribution_measurement

        else:
            raise RuntimeError(
                'Only measurement or delta g for now.'
            )

    def condition_deltaG(self, x_te):
        distribution_delta_g = self._condition_delta_g(x_te)
        f_sample = self._sample_f(distribution_delta_g)
        return f_sample, distribution_delta_g


    def _condition_measurement(self, f_sample, test_ligand_concentration, noise_model='robust'):
        """
        noise_model: robust -> StudentT, normal->Normal
        """
        mu_m = self.fractional_saturation(
            func_value=f_sample[:, None],
            test_ligand_concentration=test_ligand_concentration,
        )
        sigma_m = torch.exp(self.log_sigma_measurement)
        
        if noise_model=='normal':
            distribution_measurement = torch.distributions.normal.Normal(
                loc=mu_m, scale=sigma_m
            )
        elif noise_model=='robust':
            distribution_measurement = torch.distributions.studentT.StudentT(
                df=1,loc=mu_m, scale=sigma_m
            )
        return distribution_measurement

    def _condition_delta_g(self, x_te):
        predictive_mean, predictive_cov = self.forward(x_te)

        distribution_delta_g = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=predictive_mean.flatten(), covariance_matrix=predictive_cov
        )
        return distribution_delta_g


    def _sample_f(self, distribution_delta_g):
        f_sample = distribution_delta_g.rsample()
        return f_sample


    def _sample_measurement(self, distribution_measurement):
        return distribution_measurement.sample()


    def loss(
        self, x_te, y_te, test_ligand_concentration=None, nr_samples=1, *args, **kwargs
    ):
        """ Loss function.

        Parameters
        ----------
        x_te : `torch.Tensor`, `shape=(n_te, hidden_dimension)`
            Training input.

        y_te : `torch.Tensor`, `shape=(n_te, )`
            Training target.


        Returns
        -------
        loss : `torch.Tensor`, `shape=(,)`
            Loss function.

        """
        # define prior
        prior_mean = torch.zeros(self.n_inducing_points, 1)
        prior_tril = self._k_tr_tr().cholesky()

        prior_tril = prior_tril.to(device=x_te.device)
        prior_mean = prior_mean.to(device=x_te.device)

        

        log_p_u = self.exp_log_full_gaussian(
            self.y_tr_mu, self._y_tr_sigma(), prior_mean, prior_tril
        )

        log_q_u = -self.entropy_full_gaussian(
            self.y_tr_mu, self._y_tr_sigma()
        )

        nll = 0
        for iis in range(nr_samples):
            distribution_measurement = self.condition(
                x_te=x_te, test_ligand_concentration=test_ligand_concentration
            )

            nll += -distribution_measurement.log_prob(y_te.flatten()).mean()

        nll = nll/nr_samples

        loss = nll + self.kl_loss_scaling * (log_q_u - log_p_u)

        return loss


#
# GPyTorch implementation
#
# class VariationalGaussianProcessRegressor(GaussianProcessRegressor):
#     """
#     """
#
#     def __init__(self,
#                  in_features,
#                  n_inducing_points=100,
#                  inducing_points_boundary=1.0,
#                  num_data=1,
#                  kernel=None):
#         super(VariationalGaussianProcessRegressor, self).__init__()
#
#         # construct inducing points
#         inducing_points = torch.distributions.uniform.Uniform(
#             -1
#             * inducing_points_boundary
#             * torch.ones(n_inducing_points, in_features),
#             1
#             * inducing_points_boundary
#             * torch.ones(n_inducing_points, in_features),
#         ).sample()
#
#         class _GaussianProcessLayer(gpytorch.models.ApproximateGP):
#             def __init__(self,
#                          inducing_points,
#                          kernel=None
#                         ):
#                 if kernel is None:
#                     kernel = gpytorch.kernels.RBFKernel()
#
#                 variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
#                     num_inducing_points=inducing_points.size(0))
#
#                 variational_strategy = gpytorch.variational.VariationalStrategy(
#                     self,
#                     inducing_points,
#                     variational_distribution,
#                     learn_inducing_locations=True,
#                 )
#                 super().__init__(variational_strategy)
#
#                 self.mean_module = gpytorch.means.ConstantMean()
#                 self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
#
#             def forward(self, x):
#                 mean = self.mean_module(x)
#                 covar = self.covar_module(x)
#                 return gpytorch.distributions.MultivariateNormal(mean, covar)
#
#
#         self.gp = _GaussianProcessLayer(inducing_points, kernel=kernel)
#         self.variational_parameters = self.gp.variational_parameters
#         self.hyperparameters = self.gp.hyperparameters
#
#         self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
#
#         self.mll = gpytorch.mlls.VariationalELBO(
#             self.likelihood,
#             self.gp,
#             num_data=num_data)
#
#     def forward(self, x):
#         distribution = self.gp(x)
#         return distribution
#
#     def condition(self, *args, **kwargs):
#         return self.gp(*args, **kwargs)
#
#     def loss(self, x, y):
#         distribution = self.gp(x)
#         return -self.mll(distribution, y.flatten())
