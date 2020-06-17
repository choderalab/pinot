#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math
from pinot.inference.gp.gpr.base_gpr import GPR
import numpy as np

# =============================================================================
# MODULE CLASSES
# =============================================================================
class SVGPR(GPR):
    """ Sparse variational GPR.

    """
    def __init__(
            self,
            kernel,
            in_features,
            log_sigma=-3.0,
            n_inducing_points=100,
            initializer_std=0.1,
            kl_loss_scaling=1.0,
            grid_boundary=1):
        super(SVGPR, self).__init__()
        self.kernel = kernel

        # trainable noise
        self.log_sigma = torch.nn.Parameter(
                torch.tensor(
                    log_sigma))

        # (n_inducing_points, hidden_dimension)
        self.x_tr = torch.nn.Parameter(
            torch.distributions.uniform.Uniform(
                -1 * grid_boundary * torch.ones(
                    n_inducing_points,
                    in_features),
                1 * grid_boundary * torch.ones(
                    n_inducing_points,
                    in_features)).sample())

        self.y_tr_mu = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(n_inducing_points, 1),
                scale=initializer_std*torch.ones(n_inducing_points, 1)
                ).sample())

        # to ensure lower cholesky
        self.y_tr_diag = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(n_inducing_points),
                scale=initializer_std*torch.ones(
                    n_inducing_points)).sample())

        self.y_tr_tril = torch.nn.Parameter(
            torch.zeros(int(n_inducing_points*(n_inducing_points-1)*0.5)))

        self.n_inducing_points = n_inducing_points

        self.kl_loss_scaling = kl_loss_scaling

    def _y_tr_s(self):

        y_tr_diag = torch.diag_embed(torch.exp(self.y_tr_diag))

        mask = torch.gt(
            torch.range(0, self.y_tr_diag.shape[0]-1)[:, None],
            torch.range(0, self.y_tr_diag.shape[0]-1)[None, :])

        y_tr_s = y_tr_diag.masked_scatter(
            mask,
            self.y_tr_tril)

        return y_tr_s

    def _k_tr_tr(self):
        return self._perturb(
            self.kernel.base_kernel(
                self.x_tr,
                self.x_tr))

    def _k_tr_te(self, x_te):
        return self._perturb(
            self.kernel.base_kernel(
                self.x_tr,
                self.kernel.representation(x_te)))

    def _k_te_tr(self, x_te):
        return self._k_tr_te(x_te).t()

    def _k_te_te(self, x_te):
        return self._perturb(self.kernel.forward(x_te, x_te))

    def forward(self, x_te):
        """ Forward pass.

        """

        # get the kernels
        (
            k_tr_tr,
            k_tr_te,
            k_te_tr,
            k_te_te
        ) = (
            self._k_tr_tr(),
            self._k_tr_te(x_te),
            self._k_te_tr(x_te),
            self._k_te_te(x_te)
        )


        # (n_tr, 1)
        k_tr_tr_inv_mu, _ = torch.triangular_solve(
            input=self.y_tr_mu,
            A=k_tr_tr)

        # (n_te, 1)
        predictive_mean = k_te_tr @ k_tr_tr_inv_mu

        # (n_te, n_te)
        k_tr_tr_inv_at_k_tr_te, _ = torch.triangular_solve(
            input=k_tr_te,
            A=k_tr_tr)

        # (n_te, n_te)
        k_te_tr_at_k_tr_tr_inv_at_k_tr_te\
            = k_te_tr @ k_tr_tr_inv_at_k_tr_te

        # (n_tr, n_tr) # lower
        s_tr = self._y_tr_s()

        # (n_tr, n_tr)
        a_tr = s_tr @ s_tr.t()

        # (n_tr, n_te)
        a_tr_at_k_tr_tr_inv_at_k_tr_te\
            = a_tr @ k_tr_tr_inv_at_k_tr_te

        # (n_tr, n_te)
        k_tr_tr_inv_at_a_tr_at_k_tr_tr_inv_at_k_tr_te, _\
            = torch.triangular_solve(
                input=a_tr_at_k_tr_tr_inv_at_k_tr_te,
                A=k_tr_tr)

        # (n_te, n_te)
        k_te_tr_at_k_tr_tr_inv_at_a_tr_at_k_tr_tr_inv_at_k_tr_te\
            = k_te_tr @ k_tr_tr_inv_at_a_tr_at_k_tr_tr_inv_at_k_tr_te

        # (n_te, n_te)
        predictive_cov\
            = k_te_te\
            - k_te_tr_at_k_tr_tr_inv_at_k_tr_te\
            + k_te_tr_at_k_tr_tr_inv_at_a_tr_at_k_tr_tr_inv_at_k_tr_te

        # add noise
        # (n_te, n_te)
        predictive_cov += torch.exp(
            self.log_sigma
            ) * torch.eye(
                predictive_cov.shape[-1],
                device=predictive_cov.device)

        return predictive_mean, predictive_cov

    def condition(self, x_te):
        predictive_mean, predictive_cov = self.forward(x_te)


        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=predictive_mean.flatten(),
            covariance_matrix=predictive_cov)

        return distribution

    def loss(self, x_te, y_te):
        # define prior
        prior_mean = torch.zeros(self.n_inducing_points, 1)
        prior_tril = self._k_tr_tr().cholesky()

        distribution = self.condition(x_te)

        nll = -distribution.log_prob(y_te).sum()

        log_p_u = self.exp_log_full_gaussian(
            x_te,
            self._y_tr_s(),
            prior_mean,
            prior_tril)

        log_q_u = -self.entropy_full_gaussian(
            self.y_tr_mu,
            self._y_tr_s())

        loss = nll + self.kl_loss_scaling * (log_q_u - log_p_u)

        return loss


    @staticmethod
    def exp_log_full_gaussian(mean1, tril1, mean2, tril2):
        const_term = -0.5 * mean1.shape[0] * np.log(2 * np.pi * np.exp(1))
        log_det_prior = -torch.sum(torch.log(torch.diag(tril2)))
        LpiLq = torch.triangular_solve(tril1, tril2, upper=False)[0]
        trace_term = -0.5 * torch.sum(LpiLq ** 2)
        mu_diff = mean1 - mean2
        quad_solve = torch.triangular_solve(mu_diff, tril2, upper=False)[0]
        quad_term = -0.5 * torch.sum(quad_solve ** 2)
        res = const_term + log_det_prior + trace_term + quad_term
        return res


    @staticmethod
    def entropy_full_gaussian(mean1, tril1):
        const_term = 0.5 * mean1.shape[0] * np.log(2 * np.pi * np.exp(1))
        log_det_prior = torch.sum(torch.log(torch.diag(tril1)))
        return log_det_prior + const_term
