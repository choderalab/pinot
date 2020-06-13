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
    def __init__(self, kernel, n_tr,
            initializer_std=0.,
            noise_model='normal-homoschedastic-fixed'):

        super(VGPR, self).__init__()
        self.kernel = kernel
        self.noise_model = noise_model

        # initialize variational paramters
        self.mu = torch.nn.Parameter(
            torch.zeros(n_tr))

        # to ensure lower cholesky
        self.sigma_diag = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(n_tr),
                scale=initializer_std*torch.ones(n_tr)).sample())

        self.sigma_tril = torch.nn.Parameter(
            torch.distributions.normal.Normal(
                loc=torch.zeros(int(n_tr*(n_tr-1)*0.5)),
                scale=initializer_std*torch.ones(int(n_tr*(n_tr-1)*0.5))
            ).sample())


    def _make_sigma(self):
        return torch.diag_embed(
            torch.exp(self.sigma_diag)).masked_scatter(
                torch.gt(
                    torch.range(0, self.sigma_diag.shape[0]-1)[:, None],
                    torch.range(0, self.sigma_diag.shape[0]-1)[None, :]),
                self.sigma_tril)

    def loss(self, x_tr, y_tr):
        """ ELBO.
        """
        # point data to object
        self._x_tr = x_tr
        self._y_tr = y_tr

        # get the variational parameters
        mu = self.mu
        sigma = self._make_sigma()

        # latent covariance
        k_zz = self._perturb(self.kernel(x_tr))
        k_zz_inv = torch.inverse(k_zz)

        # construct distribution for u
        u_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=mu,
            scale_tril=sigma)

        # sample u
        u = u_distribution.rsample()[:, None]

        # compute KL divergence
        kl = 0.5 * (
            torch.logdet(k_zz) - \
            torch.logdet(sigma.t() @ sigma) +\
            # NOTE:
            # here we omitted the dimension term since it doesn't matter
            torch.trace(
                torch.matmul(
                    k_zz_inv,
                    sigma.t() @ sigma)) +\
            (mu[None, :] @ k_zz_inv @ mu[:, None])).sum()

        if self.noise_model == 'normal-homoschedastic-fixed':
            nll = -torch.distributions.normal.Normal(
                loc=u,
                scale=1.0).log_prob(y_tr).sum()
        else:
            raise NotImplementedError

        return nll + kl

    def condition(self, x_te, x_tr=None, y_tr=None, n_samples=100):
        """ Predictive distribution.
        """
        # restore training data
        if x_tr is None:
            x_tr = self._x_tr
        if y_tr is None:
            y_tr = self._y_tr

        # get variational parameters
        mu = self.mu
        sigma = self._make_sigma()

        # construct distribution for u
        u_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=mu,
            scale_tril=sigma)

        # compute the kernels
        k_tr_tr = self._perturb(self.kernel.forward(x_tr, x_tr))
        k_te_te = self._perturb(self.kernel.forward(x_te, x_te))
        k_te_tr = self._perturb(self.kernel.forward(x_te, x_tr))
        k_tr_te = k_te_tr.t() # save time

        # (batch_size_tr, batch_size_tr)
        k_plus_sigma = k_tr_tr + (sigma ** 2) * torch.eye(k_tr_tr.shape[0])

        # (batch_size_tr, batch_size_tr)
        l_low = torch.cholesky(k_plus_sigma)
        l_up = l_low.t()

        means = []
        variances = []

        for _ in range(n_samples):
            # sample u
            u = u_distribution.rsample()[:, None]

            # (batch_size_tr, 1)
            l_low_over_y, _ = torch.triangular_solve(
                input=u,
                A=l_low,
                upper=False)

            # (batch_size_tr, 1)
            alpha, _ = torch.triangular_solve(
                input=l_low_over_y,
                A=l_up,
                upper=True)

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

            means.append(mean)
            variances.append(variance)

        distributions = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.stack(means).squeeze(),
                covariance_matrix=torch.stack(variances))

        distribution = torch.distributions.mixture_same_family\
                .MixtureSameFamily(
                        torch.distributions.Categorical(
                            torch.ones(len(means))),
                        distributions)

        return distribution