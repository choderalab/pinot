# =============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math
from pinot.regressors.base_regressor import BaseRegressor
from pinot.regressors.gaussian_process_regressor import ExactGaussianProcessRegressor
from pinot.regressors.neural_network_regressor import NeuralNetworkRegressor

# import gpytorch
import numpy as np

class BiophysicalRegressor(torch.nn.Module):
    r""" Biophysically inspired model

    Parameters
    ----------

    log_sigma : `float`
        ..math:: \log\sigma observation noise

    base_regressor : a regressor object that generates a latent F


    """

    def __init__(self, base_regressor=None):
        super(BiophysicalRegressor, self).__init__()
        self.base_regressor = base_regressor
        self.log_sigma_measurement = torch.zeros(1)


	def g(self, func_value=None, test_ligand_concentration=1e3-):
	    return 1 / (1 + torch.exp(-func_value) / test_ligand_concentration)


	def condition(self, h=None, test_ligand_concentration=1e-3, **kwargs):
        distribution_base_regressor = self.base_regressor.condition(h, kwargs)

        #we sample from the latent f to paush things through the likelihood
        #Note: if we do this, in order to get good estimates of LLK we may need to draw multiple samples
        f_sample = distribution_base_regressor.rsample()

        mu_m = self.g(func_value=f_sample, test_ligand_concentration=test_ligand_concentration)
        sigma_m = torch.exp(self.log_sigma_measurement)

        distribution_measurement = torch.distributions.normal.Normal(loc=mu_m, scale=sigma_m)

        return distribution_measurement


    def loss(self, h=None, y=None, test_ligand_concentration=1e-3, **kwargs):
    	distribution_measurement = self.condition(h=h, test_ligand_concentration=test_ligand_concentration, kwargs)
    	loss_measurement = -distribution_measurement.log_prob(y)
    	return loss_measurement


   	def marginal_sample(self, h=None, n_samples=100, test_ligand_concentration=1e-3, **kwargs):
   		"""
   		Note:
   		This is currently inefficient bebcause we would normally only need 
   		to redo the sampling bit and the final nonlinearity
   		We can make this efficient later
   		"""
   		samples_measurement = []
   		for ns in range(n_samples):
    		distribution_measurement += self.condition(h=h, test_ligand_concentration=test_ligand_concentration, kwargs)
    		samples_measurement.append(distribution_measurement.sample())
    	return samples_measurement


    def marginal_loss(self, h=None, y=None, test_ligand_concentration=1e-3, n_samples=10, **kwargs):
    	"""
    	sample n_samples often from loss in order to get a better approximation
    	"""
    	marginal_loss_measurement = 0
    	for ns in range(n_samples):
    		marginal_loss_measurement += self.loss(h=h, y=y, test_ligand_concentration=test_ligand_concentration, kwargs)
    	marginal_loss_measurement/=n_samples
    	return marginal_loss_measurement

