""" Combine representation, parameterization, and distribution class
to construct a model.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import abc
import pinot
from pinot.regressors import NeuralNetworkRegressor

# =============================================================================
# BASE CLASSES
# =============================================================================
class BaseNet(torch.nn.Module, abc.ABC):
    """ Base class for `Net` object that inputs graphs and outputs
    distributions and is trainable.
    Methods
    -------
    condition :
    """

    def __init__(self, representation, output_regressor, *args, **kwargs):
        super(BaseNet, self).__init__()

        # bookkeeping
        self.representation = representation
        self.output_regressor = output_regressor

    @abc.abstractmethod
    def condition(self, g, sampler=None, *args, **kwargs):
        raise NotImplementedError

    def loss(self, g, y):
        """ Negative log likelihood loss.
        """
        # g -> h
        h = self.representation(g)

        return self._loss(h, y)

    def _loss(self, h, y):
        # if there is a special loss function implemented in the head,
        # use that instead

        if hasattr(self.output_regressor, 'loss'):
            return self.output_regressor.loss(h, y)

        distribution = self._condition(h)
        nll = -distribution.log_prob(y).mean()
        return nll


class Net(BaseNet):
    """ An object that combines the representation and parameter
    learning, puts into a predicted distribution and calculates the
    corresponding divergence.
    Attributes
    ----------
    representation: a `pinot.representation` module
        the model that translates graphs to latent representations
    """

    def __init__(
        self,
        representation,
        output_regressor=NeuralNetworkRegressor,
        **kwargs
    ):

        super(Net, self).__init__(
            representation=representation,
            output_regressor=output_regressor)

        # read the representation hidden units here
        # grab the last dimension of `representation`
        self.representation_out_features = [
            layer
            for layer in list(self.representation.modules())
            if hasattr(layer, "out_features")
        ][-1].out_features

        self.output_regressor_cls = output_regressor

        # if nothing is specified for head,
        # use the MLE with heteroschedastic model
        output_regressor = output_regressor(
            in_features=self.representation_out_features, **kwargs
        )

        self.output_regressor = output_regressor

    def _condition(self, h):
        """ Compute the output distribution.
        """

        # h -> distribution
        distribution = self.output_regressor.condition(h)

        return distribution

    def condition(self, g, sampler=None, n_samples=64):
        """ Compute the output distribution with sampled weights.
        """
        # g -> h
        h = self.representation(g)

        if sampler is None:
            return self._condition(h)

        if not hasattr(sampler, "sample_params"):
            return self._condition(h)

        # initialize a list of distributions
        distributions = []

        for _ in range(n_samples):
            sampler.sample_params()
            distributions.append(self._condition(g))

        # get the parameter of these distributions
        # NOTE: this is not necessarily the most efficienct solution
        # since we don't know the memory footprint of
        # torch.distributions
        mus, sigmas = zip(
            *[
                (distribution.loc, distribution.scale)
                for distribution in distributions
            ]
        )

        # concat parameters together
        # (n_samples, batch_size, measurement_dimension)
        mu = torch.stack(mus).cpu()  # distribution no cuda
        sigma = torch.stack(sigmas).cpu()

        # construct the distribution
        distribution = torch.distributions.normal.Normal(loc=mu, scale=sigma)

        # make it mixture
        distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
            torch.distributions.Categorical(torch.ones(mu.shape[0],)),
            torch.distributions.Independent(distribution, 2),
        )

        return distribution
