""" Combine representation, parameterization, and distribution class
to construct a model.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


class Net(torch.nn.Module):
    """ An object that combines the representation and parameter
    learning, puts into a predicted distribution and calculates the
    corresponding divergence.

    """

    def __init__(self, representation, parameterization,
                 distribution_class=torch.distributions.normal.Normal,
                 param_transform=lambda x, y: (x, torch.exp(y) + 1e-5),
                 expectation_fn=lambda x, y: x):
        super(Net, self).__init__()
        self.representation = representation
        self.parametrization = parameterization
        self.distribution_class = distribution_class
        self.param_transform = param_transform
        self.expectation_fn = expectation_fn

    def forward(self, g):
        """ Forward pass.
        """
        # graph representation $\mathcal{G}$
        # ->
        # latent representation $h$
        h = self.representation(g)

        # latent representation $h$
        # ->
        # parameters $\theta$
        theta = self.parametrization(h)

        return theta

    def condition(self, g):
        """ Compute the output distribution.
        """
        # get the parameters
        theta = self.forward(g)

        # parameterize the distribution according to the parameters
        # and the distribution class

        distribution = self.distribution_class(
            *self.param_transform(
                *torch.unbind(theta, dim=-1)))

        return distribution

    def sample(self, g, n_samples=0):
        """ Parameterize a distribution and sample from it.
        """
        distribution = self.condition(g)

        if n_samples == 0:
            return distribution.sample()

        return distribution.sample_n(n_samples)

    def expectation(self, g):
        theta = self.forward(g)
        return self.expectation_fn(
            *torch.unbind(theta, dim=-1))

    def loss(self, g, y):
        """ Compute the loss with a input graph and a set of parameters.
        """

        distribution = self.condition(g)

        return -distribution.log_prob(y)
