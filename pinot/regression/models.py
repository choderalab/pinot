import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, batch_size, out_shape, likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                 kernel=gpytorch.kernels.RBFKernel):
        """
        Takes as input configuration args
        """
        train_x = torch.zeros((batch_size, out_shape))
        train_y = torch.zeros((batch_size,))
        
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)