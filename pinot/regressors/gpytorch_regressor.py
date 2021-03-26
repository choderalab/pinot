import torch
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.models import ApproximateGP


class VariationalGP(ApproximateGP):
    def __init__(self, in_features, inducing_points=None, n_inducing_points=100,
                 mean=None, covar=None, beta=1e-2, variational_dist="cholesky"):
        """
        :param in_features: dimension of the input to GP layer
        :param inducing_points: inducing points, second dimension should be the same as in_features
        :param num_inducing_points: number of inducing points, if inducing_points are not given, will be used
            to randomly initialize inducing points
        :param mean: gpytorch.means.Mean
        :param covar: gpytorch.kernels.Kernel
        :param beta: the relative weight of the KL term in ELBO
        """
        if inducing_points is None:
            # Randomly initialize inducing points
            inducing_points = torch.rand(n_inducing_points, in_features)

        if variational_dist == "cholesky":
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2))
        elif variational_dist == "delta":
            variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-2))
        elif variational_dist == "meanfield":
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(-2))
        else:
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2))

        # LMCVariationalStrategy for introducing correlation among tasks if we want MultiTaskGP
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(VariationalGP, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.LinearMean(in_features) if mean is None else mean

        if covar is None:
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif isinstance(covar, str):
            if covar == "RBFKernel":
                self.covar_module = gpytorch.kernels.RBFKernel()
            elif covar == "LinearKernel":
                self.covar_module = gpytorch.kernels.LinearKernel()
            else:
                self.covar_module = gpytorch.kernels.PolynomialKernel(3)
        else:
            self.covar_module = covar

        self.num_inducing = inducing_points.size(-2)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.beta = beta

    def forward(self, h, **kwargs):
        """
        Computes the GP prior
        :param h:
        :return:
        """
        mean_x = self.mean_module(h)
        covar_x = self.covar_module(h)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def loss(self, h, y, *args, **kwargs):
        """
        Computes the negative ELBO
        :param h: learned features of graphs, shape (n, in_features)
        :param y: labels of graphs, shape (n,)
        :return:
        """
        approximate_dist_f = self(h, *args, **kwargs)
        num_batch = approximate_dist_f.event_shape[0]
        # Compute the log-likelihood and the KL divergence, following the same steps as in function
        # forward() of _ApproximateMarginalLogLikelihood
        log_likelihood = self.likelihood.expected_log_prob(y, approximate_dist_f, **kwargs).sum(-1).div(num_batch)
        # kl_divergence = self.variational_strategy.kl_divergence().div(self.num_data).mul(self.beta)
        kl_divergence = self.variational_strategy.kl_divergence().mul(self.beta)
        elbo = log_likelihood - kl_divergence
        return -(elbo.sum())

    def condition(self, x, *args, **kwargs):
        """ Computes the predictive distribution over y* given x*
        :param x: learned features of graphs, shape (n, in_features)
        :return:
        """
        dist_f = self(x, *args, **kwargs)
        dist_y = self.likelihood(dist_f)
        return dist_y
