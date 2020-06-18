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
    def __init__(self, head, *args, **kwargs):
        super(BaseNet, self).__init__()

        # bookkeeping
        self.head = head

    @abc.abstractmethod
    def condition(self, g, sampler=None, *args, **kwargs):
        raise NotImplementedError

    def loss(self, g, y, *args, **kwargs):
        """ Negative log likelihood loss.
        """
        # if there is a special loss function implemented in the head,
        # use that instead

        if hasattr(self.head, 'loss'):
            # get latent representation
            h = self.representation(g)

            return self.head.loss(h, y)

        # no sampling in training phase
        distribution = self.condition(g, sampler=None, *args, **kwargs)

        # the loss here is always negative log likelihood
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
        head=pinot.inference.heads.mle_head.MaximumLikelihoodEstimationHead,
        **head_kwargs
    ):

        super(Net, self).__init__(head=head)
        self.representation = representation


        # read the representation hidden units here
        # grab the last dimension of `representation`
        representation_hidden_units = [
                layer for layer in list(self.representation.modules())\
                        if hasattr(layer, 'out_features')][-1].out_features

        # if nothing is specified for head,
        # use the MLE with heteroschedastic model
        head = head(
            representation_hidden_units=representation_hidden_units,
            **head_kwargs
        )

        self.head = head


    def _condition(self, g):
        """ Compute the output distribution.
        """
        # g -> h
        h = self.representation(g)

        # h -> distribution
        distribution = self.head.condition(h)

        return distribution

    def condition(self, g, sampler=None, n_samples=64):
        """ Compute the output distribution with sampled weights.
        """
        if sampler is None:
            return self._condition(g)

        if not hasattr(sampler, 'sample_params'):
            return self._condition(g)

        # initialize a list of distributions
        distributions = []

        for _ in range(n_samples):
            sampler.sample_params()
            distributions.append(self._condition(g))

        # get the parameter of these distributions
        # NOTE: this is not necessarily the most efficienct solution
        # since we don't know the memory footprint of
        # torch.distributions
        mus, sigmas = zip(*[
                (distribution.loc, distribution.scale)
                for distribution in distributions])

        # concat parameters together
        # (n_samples, batch_size, measurement_dimension)
        mu = torch.stack(mus).cpu() # distribution no cuda
        sigma = torch.stack(sigmas).cpu()

        # construct the distribution
        distribution = torch.distributions.normal.Normal(
                loc=mu,
                scale=sigma)

        # make it mixture
        distribution = torch.distributions.mixture_same_family\
                .MixtureSameFamily(
                        torch.distributions.Categorical(
                            torch.ones(mu.shape[0],)),
                        torch.distributions.Independent(distribution, 2))

        return distribution
