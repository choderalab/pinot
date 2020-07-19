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
        return 1.0 / (1.0 + torch.exp(-delta_g) / concentration)

    def _condition_delta_g(self, x_te, *args, **kwargs):
        return self.base_regressor.condition(x_te, *args, **kwargs)

    def _condition_measurement(self, x_te, concentration=1e-3, delta_g_sample=None):
        if delta_g_sample is None:
            delta_g_sample = self._condition_delta_g(x_te).rsample()

        distribution_measurement = torch.distributions.normal.Normal(
            loc=self._get_measurement(delta_g_sample, concentration=concentration),
            scale=self.log_sigma_measurement.exp()
        )

    def condition(
        self, 
        *args,
        output="measurement", 
        **kwargs,
    ):
        if output == "measurement":
            return self._condition_measurement(*args, **kwargs)

        elif output == "delta_g":
            return self._condition_delta_g(*args, **kwargs)

        else:
            raise RuntimeError('We only support condition measurement and delta g')

        
