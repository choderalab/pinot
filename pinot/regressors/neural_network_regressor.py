# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import pinot
from pinot.regressors.base_regressor import BaseRegressor

# =============================================================================
# MODULE CLASSES
# =============================================================================
class NeuralNetworkRegressor(BaseRegressor):
    """Regressor for MaximumLikelihoodEstimation (MLE).

    Parameters
    ----------
    in_features : `int`
        Input features on the latent space.

    """

    def __init__(
        self,
        in_features,
        output_regression=None,
        noise_model="normal-heteroschedastic",
        measurement_dimension=1,
        *args, **kwargs
    ):

        super(NeuralNetworkRegressor, self).__init__()

        # get output regression if it is not specified
        if output_regression is None:
            # make the output regression as simple as a linear one
            # if nothing is specified
            self._output_regression = torch.nn.ModuleList(
                [
                    torch.nn.Linear(in_features, measurement_dimension)
                    for _ in range(2)  # NOTE: hard-coded
                ]
            )

            def output_regression(theta):
                """

                Parameters
                ----------
                theta :


                Returns
                -------

                """
                return [f(theta) for f in self._output_regression]

        # bookkeeping
        self.noise_model = noise_model
        self.output_regression = output_regression

    def condition(self, h, **kwargs):
        """Compute the output distribution.

        Parameters
        ----------
        h : `torch.Tensor`, `(n_te, hidden_dimension)`
            Latent input.

        Returns
        -------
        distribution : `torch.distributions.Distribution`
            Predictie distribution.


        """

        theta = self.output_regression(h)

        if self.noise_model == "normal-heteroschedastic":
            mu, log_sigma = theta
            distribution = torch.distributions.normal.Normal(
                loc=mu, scale=torch.exp(log_sigma)
            )

        elif self.noise_model == "normal-homoschedastic":
            mu, _ = theta

            # initialize a `LOG_SIMGA` if there isn't one
            if not hasattr(self, "LOG_SIGMA"):
                self.LOG_SIGMA = torch.zeros((1, self.measurement_dimension))
                self.LOG_SIGMA.requires_grad = True

            distribution = torch.distributions.normal.Normal(
                loc=mu, scale=torch.exp(self.LOG_SIGMA)
            )

        elif self.noise_model == "normal-homoschedastic-fixed":
            mu, _ = theta
            distribution = torch.distributions.normal.Normal(
                loc=mu, scale=torch.ones((1, self.measurement_dimension))
            )

        else:
            assert isinstance(self.noise_model, dict)

            distribution = self.noise_model[distribution](
                self.noise_model[kwargs]
            )

        return distribution
