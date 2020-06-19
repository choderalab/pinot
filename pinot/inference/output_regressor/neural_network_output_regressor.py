# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import pinot
from pinot.inference.heads.base_output_regressor import BaseOutputRegressor

# =============================================================================
# MODULE CLASSES
# =============================================================================
class MaximumLikelihoodEstimationHead(BaseHead):
    """ Head for MaximumLikelihoodEstimation (MLE).

    Methods
    -------
    condition :
        input: latent representation, tensor (n, d)
        output: distribution

    Attributes
    ----------
    measurement_dimension : dimension of the measurements to be modeled.
    output_regression : a `torch.nn.Module` or None,
        if None, this will be set as a simple `Linear` layer that inputs
        the latent dimension and output the number of parameters for
        `self.distribution_class`
    noise_model : either a string (
        one of
            'normal-homoschedastic',
            'normal-heteroschedastic',
            'normal-homoschedastic-fixed')
        or a function that transforms a set of parameters.

    """
    def __init__(
            self,
            representation_hidden_units,
            output_regression=None,
            noise_model='normal-heteroschedastic',
            measurement_dimension=1):

        super(MaximumLikelihoodEstimationHead, self).__init__()

        # get output regression if it is not specified
        if output_regression is None:
            # make the output regression as simple as a linear one
            # if nothing is specified
            self._output_regression = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            representation_hidden_units,
                            measurement_dimension)\
                                for _ in range(2) # NOTE: hard-coded
                    ])

            def output_regression(theta):
                return [f(theta) for f in self._output_regression]

        # bookkeeping
        self.noise_model = noise_model
        self.output_regression = output_regression


    def condition(self, h):
        """ Compute the output distribution.

        Parameters
        ----------
        h : tensor, shape=(n, d)

        """

        theta = self.output_regression(h)

        if self.noise_model == 'normal-heteroschedastic':
            mu, log_sigma = theta
            distribution = torch.distributions.normal.Normal(
                    loc=mu,
                    scale=torch.exp(log_sigma))

        elif self.noise_model == 'normal-homoschedastic':
            mu, _ = theta

            # initialize a `LOG_SIMGA` if there isn't one
            if not hasattr(self, 'LOG_SIGMA'):
                self.LOG_SIGMA = torch.zeros((1, self.measurement_dimension))
                self.LOG_SIGMA.requires_grad = True

            distribution = torch.distributions.normal.Normal(
                    loc=mu,
                    scale=torch.exp(self.LOG_SIGMA))

        elif self.noise_model == 'normal-homoschedastic-fixed':
            mu, _ = theta
            distribution = torch.distributions.normal.Normal(
                    loc=mu,
                    scale=torch.ones((1, self.measurement_dimension)))

        else:
            assert isinstance(
                    self.noise_model,
                    dict)

            distribution = self.noise_model[distribution](
                    self.noise_model[kwargs])


        return distribution
