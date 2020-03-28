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
            param_transform=lambda x, y: (x, torch.abs(y))):
        super(Net, self).__init__()
        self.representation = representation
        self.parameterization = parameterization
        self.distribution_class = distribution_class
        self.param_transform = param_transform

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
        theta = self.parameterization(h)

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

    def sample(self, g, n_samples):
        """ Parameterize a distribution and sample from it.
        """
        distribution = self.condition(g)

        return distribution.sample(n_samples)

    def loss(self, g, y):
        """ Compute the loss with a input graph and a set of parameters.
        """
        distribution = self.condition(g)
        
        return -distribution.log_prob(y)

