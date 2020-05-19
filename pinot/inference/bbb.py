""" Make a Bayesian-by-backprop model from any torch.nn.Module.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl

# =============================================================================
# MODULE CLASSES
# =============================================================================
class BBB(pinot.Sampler):
    """ Gaussian Variational Posterior Bayesian-by-Backprop.

    """
    def __init__(self, optimizer, initializer_std=1e-3):
        super(BBB, self).__init__()
        self.optimizer = optimizer

        # sigma here is initialized from a Gaussian distribution
        # with dimensions matching the parameters
        self.sigma = [[torch.nn.Parameter(
            torch.distributions.normal.Normal(
                torch.zeros_like(p),
                initializer_std * torch.ones_like(p)
            ).sample() for p in group['params']]
            for group in self.mu]


    def step(self, closure):
        """ Performs a single optimization step.
        
        Parameters
        ----------
        closure : callable
            a closure function that returns the loss
        """
        # just in case
        loss = None

        # perturb the parameters
        epsilon = [[
            torch.distributions.normal.Normal(
                torch.zeros_like(p),
                torch.ones_like(p)
            ).sample() for p in group['params']]
            for group in self.optimizer.param_group]

        
        
        




    def foward(self, sigma=1.0, *args, **kwargs):
        # compose the weights
        epsilon = [
            torch.distributions.normal.Normal(
                torch.zeros_like(self.mu[idx]), sigma * torch.ones_like(self.mu[idx])
            ).sample()
            for idx in range(self.n_param)
        ]

        theta = [
            self.mu[idx] + self.sigma[idx] * epsilon[idx] for idx in range(self.n_param)
        ]

        self.base_module.load_state_dict(zip(self.base_module.state_dict.keys(), theta))

        return self.base_module.forward(*args, **kwargs)

    def sample(self, sigma=1.0, n_samples=1, *args, **kwargs):

        return torch.stack(
            [self.foward(sigma, *args, **kwargs) for _ in range(n_samples)], dim=0
        )
