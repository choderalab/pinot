import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                 mean_module=gpytorch.means.ConstantMean, prior=None,
                 kernel=gpytorch.kernels.RBFKernel,
                 ard_num_dims=None):
        """
        Takes as input configuration args

        Parameters
        ----------
        likelihood : gpytorch.likelihoods object
        mean_module : gpytorch.means object
        prior : float, 1d pytorch Tensor
        """
        super(ExactGPModel, self).__init__(None, None, likelihood)
        self.num_outputs = 1
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel(ard_num_dims=ard_num_dims)
        )
        self.mean_module = mean_module()
        
        if prior:
            self.mean_module.initialize(constant=prior)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # make compatible with botorch... might go back and name forward, posterior 
    posterior = forward