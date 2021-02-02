import torch
import math
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.models import ApproximateGP, ExactGP


class ExactGaussianProcesses(ExactGP):
    def __init__(self, in_features, num_data, mean=None, covar=None):
        # Can replace this with MultiTaskGaussianLikelihood when one wishes to use MultiTaskGP
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        temp_x = torch.zeros(num_data, in_features)
        temp_y = torch.zeros(num_data, 1)
        super(ExactGaussianProcesses, self).__init__(temp_x, temp_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(in_features) if mean is None else mean
        self.covar_module = gpytorch.kernels.RBFKernel() if covar is None else covar
        self.num_data = num_data

    def forward(self, h):
        mean_x = self.mean_module(h)
        covar_x = self.covar_module(h)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def loss(self, h, y, *args, **kwargs):
        """
        Compute the negative log likelihood
        :param h:
        :param y:
        :return:
        """
        self.set_train_data(h, y) # Only set new data during training
        dist_f = self(h, *args, **kwargs)  # The predictive distribution
        llh = self.likelihood(dist_f)  # Compute the likelihood distribution
        return -llh.log_prob(y)  # Compute negative log likelihood of the observed labels

    def condition(self, h, *args, **kwargs):
        """
        Returns the conditional distribution p(f|x)
        :param h:
        :return:
        """
        dist = self(h, *args, **kwargs)
        return dist


class VariationalGP(ApproximateGP):
    def __init__(self, in_features, num_data, inducing_points=None, num_inducing_points=100,
                 mean=None, covar=None, beta=1.0):
        if inducing_points is None:
            # Randomly initialize inducing points
            inducing_points = torch.rand(num_inducing_points, in_features)

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2)
        )

        # LMCVariationalStrategy for introducing correlation among tasks if we want MultiTaskGP
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(VariationalGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.LinearMean(in_features) if mean is None else mean
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        ) if covar is None else covar
        self.num_inducing = inducing_points.size(-2)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.num_data = num_data
        self.beta = beta

    def forward(self, h):
        """

        :param h:
        :return:
        """
        mean_x = self.mean_module(h)
        covar_x = self.covar_module(h)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def loss(self, h, y, *args, **kwargs):
        """
        Computes the negative ELBO
        :param h:
        :param y:
        :return:
        """
        approximate_dist_f = self(h, *args, **kwargs)
        num_batch = approximate_dist_f.event_shape[0]
        # Compute the log-likelihood and the KL divergence, following the same steps as in function
        # forward() of _ApproximateMarginalLogLikelihood
        log_likelihood = self.likelihood.expected_log_prob(y, approximate_dist_f, **kwargs).sum(-1).div(num_batch)
        kl_divergence = self.variational_strategy.kl_divergence().div(self.num_data).mul(self.beta)
        elbo = log_likelihood - kl_divergence
        return -(elbo.sum())

    def condition(self, x, *args, **kwargs):
        """ Computes the predictive distribution over y* given x*
        :param x:
        :return:
        """
        dist_f = self(x, *args, **kwargs)
        dist_y = self.likelihood(dist_f)
        return dist_y
