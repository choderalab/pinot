# =============================================================================
# IMPORTS
# =============================================================================
import torch


# =============================================================================
# MODULE CLASSES
# =============================================================================
class AdLaLa(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            h=0.25,
            gamma=0.1,
            sigma=0.01,
            epsilon=0.05,
            tau=1e-4,
            xi_init=1e-3):
    
        defaults = dict(
            h=torch.tensor(h),
            tau=torch.tensor(tau),
            gamma=torch.tensor(gamma),
            sigma=torch.tensor(sigma),
            epsilon=torch.tensor(epsilon),
            xi_init=torch.tensor(xi_init),)

        super(AdLaLa, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.

        """
        # call closure if closure is specified
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for w in group['params']:
                state = self.state[w]

                # state initialization
                if len(state) == 0:

                    # initialize momentum
                    state['p'] = torch.zeros_like(w)
                    state['xi'] = group['xi_init']

                    group['c'] = torch.exp(
                            -group['h'] * group['gamma'])
                    group['d'] = torch.sqrt(
                            1. - torch.exp(-2. * group['h'] * group['gamma']
                                )) * torch.sqrt(group['tau'])

                    # B_(h/2) step: p := p - 0.5 * h * d_l
                    state['p'].add_(-0.5 * group['h'] * w.grad)
                    
                # A_(h/2) step: theta := theta + 0.5 * h * p
                w.add_(0.5 * group['h'] * state['p'])
        
        for group in self.param_groups:
            for w in group['params']:
                state = self.state[w]
                if group['partition'].lower() == 'adla':

                    # C_(h/2) step: p := p * exp(-0.5 * h * \xi)
                    state['p'].mul_(torch.exp(-0.5 * group['h'] * state['xi']))

                    # D_(h/2) step: p := p + sqrt(0.5 * h) * \sigma * R
                    state['p'].add_(
                        torch.sqrt(0.5 * group['h']) *\
                        group['sigma'] *\
                        torch.randn(w.shape))
                    
                    # E_(h/2) step: \xi := \xi + 0.5 * h * \epsilon * (p^T p - N * \tao)
                    state['xi'].add_(
                        0.5 * group['h'] * group['epsilon'] *\
                        (torch.sum(torch.pow(state['p'].flatten(), 2)) - state['p'].shape[0] * group['tau']))
                
                elif group['partition'].lower() == 'la':
                    
                    # O step: p := c*p + d*R
                    state['p'].mul_(group['c'])
                    state['p'].add_(group['d'] * torch.randn(w.shape))

        for group in self.param_groups[::-1]:
            for w in group['params']:
                state = self.state[w]
                if group['partition'].lower() == 'adla':
                    # E_(h/2) step: \xi := \xi + 0.5 * h * \epsilon * (p^T p - N * \tao)
                    state['xi'].add_(
                        0.5 * group['h'] * group['epsilon'] *\
                        (torch.sum(torch.pow(state['p'].flatten(), 2)) - state['p'].shape[0] * group['tau']))

                    # D_(h/2) step
                    state['p'].add_(
                        torch.sqrt(0.5 * group['h']) *\
                        group['sigma'] *\
                        torch.randn(w.shape))

                    # C_(h/2) step
                    state['p'].mul_(torch.exp(-0.5 * group['h'] * state['xi']))

                # A_(h/2) step
                w.add_(0.5 * group['h'] * state['p'])

        with torch.enable_grad():
            # re-evaluate
            loss = closure()

        for group in self.param_groups[::-1]:
            for w in group['params']:
                state = self.state[w]
                # B step
                state['p'].add_(-h * w.grad)
                    
