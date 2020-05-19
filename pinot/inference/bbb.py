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
class BBB(torch.optim.Optimizer):
    """ Gaussian Variational Posterior Bayesian-by-Backprop.

    """
    def __init__(self, optimizer, initializer_std=1e-3,
            theta_prior=torch.distributions.normal.Normal(0, 1)
        ):
        super(BBB, self).__init__()
        self.optimizer = optimizer

        # TODO:
        # make this compilable with more than one param group
        assert len(self.optimizer.param_groups
                ) == 1, 'Now we only support one param group.'

        # copy the original param group
        # this makes the hyperparameters stable
        log_sigma_param_group = self.optimizer.param_groups[0].copy()

        # initialize log_sigma
        log_sigma_param_group['param'] = [torch.distributions.normal.Normal(
            loc=torch.zeros_like(p),
            scale=initializer_std * torch.ones_like(p))
            for p in log_sigma_param_group['param']]

        # append this to param_group
        self.optimizer.add_param_group(
                log_sigma_param_group)
           

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

        for p, log_sigma in zip(
                *[
                    self.optimizer.param_groups[0]['params'],
                    self.optimizer.param_groups[1]['params']
                ]):


            state = self.optimizer.state[p]
            
            # sample a noise
            epsilon = torch.distributions.normal.Normal(
                    torch.zeros_like(p),
                    torch.ones_like(p)).sample()
            
            # clone mu and log_sigma
            mu = p.detach().clone()

            # perturb p to get theta
            theta = p + epsilon * torch.exp(log_sigma.detach())
            p.copy_(theta)

            # calculate kl loss and the gradients thereof
            with torch.enable_grad():
                mu.requires_grad = True
                theta.requires_grad = True
                 
                # compute the kl loss term here
                kl_loss = torch.distributions.normal.Normal(
                        loc=mu,
                        scale=torch.exp(log_sigma)).log_prob(theta) -\
                          self.theta_prior.log_prob(theta)
            
            d_kl_d_mu = torch.autograd.grad(kl_loss, mu)
            d_kl_d_log_sigma = torch.autograd.grad(kl_loss, log_sigma)
            d_kl_d_theta = torch.autograd.grad(kl_loss, theta)

            # put the results in state dicts
            state['d_kl_d_mu'] = d_kl_d_mu
            state['d_kl_d_log_sigma'] = d_kl_d_log_sigma
            state['d_kl_d_theta'] = d_kl_d_theta

            # keep track of perturbation noise for cancellation later
            state['mu'] = mu
            state['epsilon'] = epsilon
            
        # do one step with perturbed weights
        with torch.enable_grad():
            loss = closure()

        for p, log_sigma in zip(
                *[
                    self.optimizer.param_groups[0]['params'],
                    self.optimizer.param_groups[1]['params']
                ]):
                
            state = self.optimizer.state[p]
            
            # cancel the perturbation
            p.copy_(state['mu'])

            # modify grad
            p.grad.add_(state['d_kl_d_mu'] + state['d_kl_d_theta'])
            log_sigma.grad.copy_(
                    state['epsilon'] * (p.grad + state['d_kl_d_theta']) +\
                            state['d_kl_d_log_sigma'])

        
        # update parameters based on whatever schedule
        # `self.optimizer` proposes
        self.optimizer.step()


            

                

        
        




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
