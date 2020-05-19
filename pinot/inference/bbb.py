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
    def __init__(self, optimizer, initializer_std=1e-3,
            theta_prior=torch.distributions.normal.Normal(0, 1)
        ):
        super(BBB, self).__init__()
        self.optimizer = optimizer

        # sigma here is initialized from a Gaussian distribution
        # with dimensions matching the parameters
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.optimizer.state[p]
                state['log_sigma'] = torch.distributions.normal.Normal(
                    torch.zeros_like(p),
                    initializer_std * torch.ones_like(p)
                ).sample()


    @torch.no_grad()
    def step(self, closure):
        """ Performs a single optimization step.
        
        Parameters
        ----------
        closure : callable
            a closure function that returns the loss
        """

        # just in case
        loss = None

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.optimizer.state[p]
                
                epsilon = torch.distributions.normal.Normal(
                        torch.zeros_like(p),
                        torch.ones_like(p)).sample()
                
                # clone mu and log_sigma
                mu = p.clone()
                log_sigma = state['log_sigma'].clone()

                with torch.enable_grad():
                    mu.requires_grad = True
                    log_sigma.requires_grad=True
                    
                    # compute the kl loss term here
                    kl_loss = torch.distributions.normal.Normal(
                            loc=mu,
                            scale=torch.exp(state['sigma'])).log_prob(p) -\
                              self.theta_prior.log_prob(theta)

                    
                    d_kl_d_mu = torch.autograd(kl_loss, mu)
                    d_kl_d_log_sigma = torch.autograd(kl_loss, log_sigma)

        # do one step with perturbed weights
        with torch.enable_grad():
            loss = closure()



        
        




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
