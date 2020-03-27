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
class GaussianVariationalPosteriorBayesianByBackprop(torch.nn.Module):
    def __init__(self, base_module, initializer_std=1e-3):
        super(GaussianVariationalPosteriorBayesianByBackprop, self).__init__()
        self.base_module = base_module

        # $\mu$ and $\sigma$ here are trainable parameters
        self.mu = list(self.base_module.parameters())
        self.n_param = len(self.mu)
        self.sigma = [torch.nn.Parameter(torch.distributions.normal.Normal(
            torch.zeros_like(weight),
            initializer_std * torch.ones_like(weight)
        ).sample()) for weight in self.mu]

    def foward(self, sigma=1.0, *args, **kwargs):
        # compose the weights
        epsilon = [
            torch.distributions.normal.Normal(
                torch.zeros_like(self.mu[idx]),
                sigma * torch.ones_like(self.mu[idx])
            ).sample() for idx in range(self.n_param)]

        theta = [
            self.mu[idx] + self.sigma[idx]\
                * epsilon[idx] for idx in range(self.n_param)]

        self.base_module.load_state_dict(
            zip(
                self.base_module.state_dict.keys(),
                theta))

        return self.base_module.forward(*args, **kwargs)

    def sample(self, sigma=1.0, n_samples=1, *args, **kwargs):

        return torch.stack(
            [
                self.foward(sigma, *args, **kwargs) for _ in range(n_samples)
            ], dim=0)
