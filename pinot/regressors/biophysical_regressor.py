# =============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math
import numpy as np
from pinot.regressors.base_regressor import BaseRegressor

# =============================================================================
# MODULE CLASSES
# =============================================================================
class BiophysicalRegressor(BaseRegressor):
    r""" Biophysically inspired model

    Parameters
    ----------

    log_sigma : `float`
        ..math:: \log\sigma observation noise

    base_regressor : a regressor object that generates a latent F

    """

    def __init__(self, base_regressor_class=None, *args, **kwargs):
        super(BiophysicalRegressor, self).__init__()
        # get the base regressor
        self.base_regressor_class = base_regressor_class
        self.base_regressor = base_regressor_class(
                *args, **kwargs
        )

        # initialize measurement parameter
        self.log_sigma_measurement = torch.nn.Parameter(torch.zeros(1))

    def _get_measurement(self, delta_g, concentration=1e-3):
        """ Translate ..math:: \Delta G to percentage inhibition.

        Parameters
        ----------
        delta_g : torch.Tensor, shape=(number_of_graphs, 1)
           Binding free energy. 

        concentration : torch.Tensor, shape=(,) or (1, number_of_concentrations)
            Concentration of ligand.

        Returns
        -------
        measurement : torch.Tensor, shape=(number_of_graphs, 1)
            or (number_of_graphs, number_of_concentrations)
            Percentage of inhibition.

        """
        return 1.0 / (1.0 + torch.exp(-delta_g) / concentration)

    def _condition_delta_g(self, x_te, *args, **kwargs):
        """ Returns predictive distribution of binding free energy. """
        return self.base_regressor.condition(x_te, *args, **kwargs)

    def _condition_measurement(self, x_te=None, concentration=1e-3, delta_g_sample=None):
        """ Returns predictive distribution of percentage of inhibtiion. """
        # sample delta g if not specified
        if delta_g_sample is None and x_te is not None:
            delta_g_sample = self._condition_delta_g(x_te).rsample()

        distribution_measurement = torch.distributions.normal.Normal(
            loc=self._get_measurement(delta_g_sample, concentration=concentration),
            scale=self.log_sigma_measurement.exp()
        )

    def _condition_ic_50(
            self, 
            x_te=None,
            delta_g_sample=None,
            concentration_low=0.0,
            concentration_high=1.0,
            number_of_concentrations=1024,
    ):
        """ Returns predictive distribution of ic50 """
        # sample delta g if not specified
        if delta_g_sample is None and x_te is not None:
            delta_g_sample = self._condition_delta_g(x_te).rsample()

        # get the possible array of concentration
        concentration = torch.linspace(
            start=concentration_low,
            end=concentration_high,
            steps=number_of_concentrations)

        distribution_measurement = torch.distributions.normal.Normal(
            loc=self._get_measurement(delta_g_sample, concentration=concentration),
            scale=self.log_sigma_measurement.exp(),
        )

        # (number_of_concentrations, 1)
        measurement_sample = distribution_measurement.rsample()







    def condition(
        self, 
        *args,
        output="measurement", 
        **kwargs,
    ):
        """ Public method for predictive distribution construction. """
        if output == "measurement":
            return self._condition_measurement(*args, **kwargs)

        elif output == "delta_g":
            return self._condition_delta_g(*args, **kwargs)

        elif output == "ic50":
            return self._condition_ic50(*args, **kwargs)

        else:
            raise RuntimeError('We only support condition measurement and delta g')

        
