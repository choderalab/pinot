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


    Attributes
    ----------
    representation: a `pinot.representation` module
        the model that translates graphs to latent representations
    output_regression: a `torch.nn.Moudle` or None,
        if None, this will be set as a simple `Linear` layer that inputs
        the latent dimension and output the number of parameters for
        `self.distribution_class`
    distribution_class: a `torch.distribution` object,
        default=`torch.distributions.normal.Normal`
        the class of distribution the model outputs
    noise_model: either a string (
        one of 
            'homoschedastic',
            'heteroschedastic',
            'homoschedastic-fixed') 
        or a function that transforms a set of parameters.


    """

    def __init__(
        self,
        representation,
        output_regression=None,
        distribution_class=torch.distributions.normal.Normal,
        noise_model='heteroschedastic',
    ):
        
        super(Net, self).__init__()
        self.representation = representation
        self.distribution_class = distribution_class

        # grab the last dimension of `representation`
        regression_in_features = [
                layer for layer in list(self.representation.modules())\
                        if hasattr(layer, 'out_features')][-1].out_features

        # TODO:
        # make this less ugly
        regression_out_features = len(distribution_class.arg_constraints)

        if output_regression is None:
            # make the output regression as simple as a linear one
            # if nothing is specified
            self._output_regression = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(regression_in_features, 1)\
                                for _ in range(regression_out_features)
                    ])

            def output_regression(theta):
                return [f(theta) for f in self._output_regression]

        self.output_regression = output_regression

        # TODO:
        # make this less ugly
        if isinstance(noise_model, str):
            if noise_model == 'heteroschedastic':
                self.noise_model = lambda x : x

            elif noise_model == 'homoschedasstic':
                # NOTE: we assume noise parameter is the last one
                self.LOG_SIGMA = torch.tensor(0.)
                self.noise_model = lambda x: x[:-1] + \
                        [self.LOG_SIGMA * torch.ones_like(x[-1])]

            elif noise_model == 'homoschedastic':
                self.noise_model = lambda x: x[:-1] + torch.ones_like(x[-1])
            

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
        theta = self.output_regression(h)
       
        return theta

    def condition(self, g):
        """ Compute the output distribution.
        """
        # get the parameters
        theta = self.forward(g)

        # put theta through noise model
        theta = self.noise_model(theta)

        # parameterize the distribution according to the parameters
        # and the distribution class
        distribution = self.distribution_class(*theta)

        return distribution

    def expectation(self, g):
        # construct the distribution
        distribution = self.condition(g)
        
        # expectation is the analytical mean of the distribution
        return distribution.mean()

    def loss(self, g, y):
        """ Compute the loss with a input graph and a set of parameters.
        """

        distribution = self.condition(g)

        return -distribution.log_prob(y)
