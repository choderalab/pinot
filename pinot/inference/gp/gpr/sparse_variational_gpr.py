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

        self.log_sigma = torch.nn.Parameter(
                torch.tensor(
                    log_sigma))

        # (n_inducing_points, hidden_dimension)
        self.x_u = torch.nn.Parameter(
            torch.distributions.uniform.Uniform(
                -1 * grid_boundary * torch.ones(n_inducing_points, in_features),
                1 * grid_boundary * torch.ones(n_inducing_points, in_features)).sample())

        # to ensure lower cholesky
        self.f_u_s_diag = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(n_inducing_points),
                scale=initializer_std*torch.ones(n_inducing_points)).sample())

        self.f_u_s_tril = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(int(n_inducing_points*(n_inducing_points-1)*0.5)),
                scale=initializer_std*torch.ones(int(n_inducing_points*(n_inducing_points-1)*0.5))
            ).sample())

        self.f_u_mu = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(n_inducing_points, 1),
                scale=initializer_std*torch.ones(n_inducing_points, 1)).sample())

        self.n_inducing_points = n_inducing_points

        self.kl_loss_scaling = kl_loss_scaling

    def _f_u_s(self):

        f_u_s_diag = torch.diag_embed(torch.exp(self.f_u_s_diag))

        mask = torch.gt(
            torch.range(0, self.f_u_s_diag.shape[0]-1)[:, None],
            torch.range(0, self.f_u_s_diag.shape[0]-1)[None, :])

        f_u_s = f_u_s_diag.masked_scatter(
            mask,
            self.f_u_s_tril)

        return f_u_s

    def _kuu(self):
        return self._perturb(self.kernel.base_kernel(self.x_u, self.x_u))

    def _kuf(self, x_te):
        return self._perturb(self.kernel.base_kernel(
            self.x_u, self.kernel.representation(x_te)))

    def _kfu(self, x_te):
        return self._kuf(x_te).t()

    def _kff(self, x_te):
        return self._perturb(self.kernel.forward(x_te, x_te))

    def _forward_prep(self):

        kuu = self._kuu()

        prior_tril = torch.cholesky(kuu)
        prior_mean = torch.zeros(self.n_inducing_points, 1)

        k_inv_mu, _ = torch.triangular_solve(
            self.f_u_mu,
            prior_tril,
            upper=False)

        k_inv_s, _ = torch.triangular_solve(
            self._f_u_s(),
            prior_tril,
            upper=False)

        return prior_tril, prior_mean, k_inv_mu, k_inv_s

    def forward(self, x_te):
        """ Forward pass.

        """


        (
            kuu,
            kfu,
            kuf,
            kff
        ) = (
            self._kuu(),
            self._kfu(x_te),
            self._kuf(x_te),
            self._kff(x_te)
        )


        # (n_inducing_points, n_inducing_points)
        prior_tril, prior_mean, k_inv_mu, k_inv_s = self._forward_prep()

        LKinvKuf, _ = torch.triangular_solve(
            kuf,
            prior_tril,
            upper=False)


        kfu_kuu_inv_kuf = LKinvKuf.t() @ LKinvKuf


        LKinvUmean, _ = torch.triangular_solve(
                self.f_u_mu, prior_tril, upper=False)


        m_all = torch.einsum('ab,ac->cb', LKinvUmean, LKinvKuf)

        LLSu, _ = torch.triangular_solve(self._f_u_s(), prior_tril, upper=False)
        LSuKuuf = torch.matmul(LLSu.t(), LKinvKuf)
        kfuuSukuuf = LSuKuuf.t() @ LSuKuuf

        v1 = kff
        v2 = kfu_kuu_inv_kuf
        v3 = kfuuSukuuf
        v = v1 - v2 + v3
        v = (v + torch.exp(self.log_sigma) * torch.eye(v.shape[-1]))
        L = torch.cholesky(v)


        self.prior_tril = prior_tril
        self.prior_mean = prior_mean

        return m_all, L

    def condition(self, x_te):
        m_all, L = self.forward(x_te)
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=m_all.flatten(),
            scale_tril=L)

        return distribution

    def loss(self, x_te, y_te):

        distribution = self.condition(x_te)

        nll = -distribution.log_prob(y_te).sum()

        log_p_u = self.exp_log_full_gaussian(
            self.f_u_mu,
            self._f_u_s(),
            self.prior_mean,
            self.prior_tril)

        log_q_u = -self.entropy_full_gaussian(
            self.f_u_mu,
            self._f_u_s())

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
