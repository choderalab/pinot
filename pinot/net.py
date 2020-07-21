""" Combine representation, parameterization, and distribution class
to construct a model.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
from pinot.regressors import NeuralNetworkRegressor
from pinot.regressors import ExactGaussianProcessRegressor

# =============================================================================
# BASE CLASSES
# =============================================================================
class BaseNet(torch.nn.Module, abc.ABC):
    """Base class for `Net` object that inputs graphs and outputs
    distributions and is trainable.

    Parameters
    ----------

    Returns
    -------

    Methods
    -------
    condition :
    """

    def __init__(
        self, representation, output_regressor_class, *args, **kwargs
    ):
        super(BaseNet, self).__init__()

        # bookkeeping
        self.representation = representation

    @abc.abstractmethod
    def condition(self, g, sampler=None, *args, **kwargs):
        """

        Parameters
        ----------
        g :

        sampler :
             (Default value = None)
        *args :

        **kwargs :


        Returns
        -------

        """
        raise NotImplementedError

    def loss(self, g, y, *args, **kwargs):
        """Negative log likelihood loss.

        Parameters
        ----------
        g :

        y :


        Returns
        -------

        """
        # g -> h
        h = self.representation(g)

        return self._loss(h, y, *args, **kwargs)

    def _loss(self, h, y, *args, **kwargs):
        """

        Parameters
        ----------
        h :

        y :


        Returns
        -------

        """
        # use loss function from output_regressor, if already implemented
        if hasattr(self.output_regressor, "loss"):
            return self.output_regressor.loss(h, y, *args, **kwargs)

        distribution = self._condition(h, *args, **kwargs)
        nll = -distribution.log_prob(y).sum()
        return nll


class Net(BaseNet):
    """An object that combines the representation and parameter
    learning, puts into a predicted distribution and calculates the
    corresponding divergence.

    Parameters
    ----------
    representation : `pinot.representation` module
        The model that translates graphs to latent representations.

    output_regressor : `pinot.regressors.BaseRegressor`
        Output regressor that inputs latent encode and outputs distributions.



    Methods
    -------
    loss : Compute loss function.

    condition : Construct predictive distribution.

    """

    def __init__(
        self,
        representation,
        output_regressor_class=NeuralNetworkRegressor,
        output_regressor=None,
        output_likelihood_class=None,
        **kwargs
    ):

        super(Net, self).__init__(
            representation=representation,
            output_regressor_class=output_regressor_class,
            output_likelihood_class=output_likelihood_class,
        )

        # read the representation hidden units here
        # grab the last dimension of `representation`
        self.representation_out_features = [
            layer
            for layer in list(self.representation.modules())
            if hasattr(layer, "out_features")
        ][-1].out_features

        # `output_regressor` overwrites `output_regressor_class`
        # if nothing is specified for head,
        # use the MLE with heteroschedastic model
        if output_regressor is None and output_regressor_class is not None:
            output_regressor = output_regressor_class(
                in_features=self.representation_out_features, **kwargs
            )


        # determine if the output regressor is an `ExactGaussianProcess`
        self.has_exact_gp = False
        if isinstance(output_regressor, ExactGaussianProcessRegressor):
            self.has_exact_gp = True

        self.representation = representation

        if output_likelihood_class is not None:
            self.output_regressor = output_likelihood_class(
                base_regressor=output_regressor
            )
        else:
            self.output_regressor = output_regressor

    def loss(self, g, y, *args, **kwargs):
        """ Negative log likelihood loss.

        Parameters
        ----------
        g : `dgl.DGLGraph`
            Training input graph.

        y : `torch.Tensor`, `shape=(n_tr, 1)`
            Training target.


        Returns
        -------
        loss : `torch.Tensor`, `shape=(, )`
            Loss function value.

        """
        # g -> h
        h = self.representation(g)

        if self.has_exact_gp is True:
            self.g_last = g
            self.y_last = y

        return self._loss(h, y, *args, **kwargs)

    def _condition(self, h, *args, **kwargs):
        """ Compute the output distribution from latent without sampling. """

        # h -> distribution
        distribution = self.output_regressor.condition(h, *args, **kwargs)

        return distribution

    def condition(self, g, *args, **kwargs):
        """ Compute the output distribution with sampled weights.

        Parameters
        ----------
        g : `dgl.DGLGraph`
            Input graph.

        sampler : `torch.optim.Optimizer` or `pinot.Sampler`
             (Default value = None)
             Sampler to sample weights and come up with predictive distribution.

        n_samples : `int`
             (Default value = 64)
             Number of samples to be drown to come up with predictive
             distribution.

        Returns
        -------
        distribution : `torch.distributions.Distribution`
            Predictive distribution.

        """
        # g -> h
        h = self.representation(g)
        # kwargs = {}

        # workout kwargs
        # for sampling
        sampler = None
        n_samples = 64

        if 'sampler' in kwargs:
            sampler = kwargs.pop('sampler')

        if self.has_exact_gp is True:
            h_last = self.representation(self.g_last)
            kwargs = {**{"x_tr": h_last, "y_tr": self.y_last}, **kwargs}


        if sampler is not None and hasattr(sampler, 'sample_params'):
            sampler.sample_params()
        
        return self._condition(h, *args, **kwargs)

