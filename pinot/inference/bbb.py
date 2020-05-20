""" Make a Bayesian-by-backprop model from any torch.nn.Module.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import pinot

# =============================================================================
# MODULE CLASSES
# =============================================================================
class BBB(pinot.inference.Sampler):
    """ Gaussian Variational Posterior Bayesian-by-Backprop.
    """
    def __init__(self, optimizer, initializer_std=1e-3,
            theta_prior=torch.distributions.normal.Normal(0, 1)
        ):
        
        self.optimizer = optimizer
        self.theta_prior = theta_prior

        # TODO:
        # make this compilable with more than one param group
        assert len(self.optimizer.param_groups
                ) == 1, 'Now we only support one param group.'

        # copy the original param group
        # this makes the hyperparameters stable
        log_sigma_param_group = self.optimizer.param_groups[0].copy()

        # initialize log_sigma
        log_sigma_param_group['params'] = [torch.distributions.lor_normal.LogNormal(
            loc=torch.zeros_like(p),
            scale=initializer_std * torch.ones_like(p)).sample()
            for p in log_sigma_param_group['params']]

        # append this to param_group
        self.optimizer.add_param_group(
                log_sigma_param_group)
        
        # initialize
        for p, log_sigma in zip(
                *[
                    self.optimizer.param_groups[0]['params'],
                    self.optimizer.param_groups[1]['params']
                ]):
            
            p.grad = torch.zeros_like(p)
            log_sigma.grad = torch.zeros_like(log_sigma)

        self.optimizer.step()
        self.optimizer.zero_grad()

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
            # $ \epsilon ~ \mathcal{N}(0, 1) $
            epsilon = torch.distributions.normal.Normal(
                    torch.zeros_like(p),
                    torch.ones_like(p)).sample()
            
            # clone mu and log_sigma
            mu = p.detach().clone()

            # make a copy of sigma to make sure we don't mess up the rest of
            # the gradients
            _log_sigma = log_sigma.detach().clone()

            # perturb p to get theta
            # $ \theta = \mu + \sigma \epsilon $
            theta = p + epsilon * torch.exp(_log_sigma)
            p.copy_(theta)

            # calculate kl loss and the gradients thereof
            with torch.enable_grad():
                mu.requires_grad = True
                theta.requires_grad = True
                _log_sigma.requires_grad = True
                 
                # compute the kl loss term here
                kl_loss = torch.distributions.normal.Normal(
                        loc=mu,
                        scale=torch.exp(_log_sigma)).log_prob(theta).sum() -\
                          self.theta_prior.log_prob(theta).sum()
            
            d_kl_d_mu = torch.autograd.grad(
                kl_loss, mu, retain_graph=True)[0]
            d_kl_d_log_sigma = torch.autograd.grad(
                kl_loss, _log_sigma, retain_graph=True)[0]
            d_kl_d_theta = torch.autograd.grad(
                kl_loss, theta, retain_graph=False)[0]

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

            log_sigma.requires_grad = True

            log_sigma.backward(
                state['epsilon'] * (p.grad + state['d_kl_d_theta']) +\
                            state['d_kl_d_log_sigma'])
            
            # modify grad
            p.grad.add(state['d_kl_d_mu'] + state['d_kl_d_theta'])
        
        # update parameters based on whatever schedule
        # `self.optimizer` proposes
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def sample_params(self):
        with torch.no_grad():
            for p, log_sigma in zip(
                    *[
                        self.optimizer.param_groups[0]['params'],
                        self.optimizer.param_groups[1]['params']
                    ]):
                
                p.copy_(
                    torch.distributions.normal.Normal(
                        loc=self.optimizer.state[p]['mu'],
                        scale=torch.exp(log_sigma)
                    ).sample())
                
    def expectation_params(self):
        with torch.no_grad():
            for p, log_sigma in zip(
                    *[
                        self.optimizer.param_groups[0]['params'],
                        self.optimizer.param_groups[1]['params']
                    ]):
                
                p.copy_(self.optimizer.state[p]['mu'])
