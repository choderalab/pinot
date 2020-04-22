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
            h=0.25,
            tau1=1e-4,
            tau2=1e-4,
            gamma=0.1,
            sigma=0.01,
            xi_init=1e-3,
            eps=0.05):
    
        defaults = dict(
            h=h,
            tau=tau,
            gamma=gamma,
            sigma=sigma,
            xi_init=xi_init,
            epsilon=epsilon)

        super(AdLaLa, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.

        """
        # call closure if closure is specified
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss =closure()

        for group in self.param_groups:
            for w in group['params']:
                
                state = self.state[w]

                # state initialization
                if len(state) == 0:
                    state['step'] = 0

                    # initialize momentum
                    state['p'] = torch.zeros_like(w)
                    state['xi'] = group['xi_init']
                    state['c'] = torch.exp(
                            -group['h'] * group['gamma'])
                    state['d'] = torch.sqrt(
                            1 - torch.exp(-2. * group['h'] * group['gamma']
                                ) * torch.sqrt(group['tau'])

                    # B_(h/2) step: p := p - 0.5 * h * d_l
                    state['p'] = state['p'] - 0.5 * group['h'] * w.grad 

                # A_(h/2) step: theta := theta + 0.5 * h * p
                w.add_(0.5 * group['h'] * state['p'])

                # C_(h/2) step: p := p * exp(-0.5 * h * \xi)
                state['p'].mul_(torch.exp(-0.5 * group['h'] * state['xi']))

                # D_(h/2) step: p := p + sqrt(0.5 * h) * \sigma * R
                state['p'].add_(
                    torch.sqrt(0.5 * h) *\
                    group['sigma'] *\
                    torch.randn(w.shape))

                # E_(h/2) step: \xi := \xi + 0.5 * h * \epsilon * (p^T p - N * \tao)
                state['xi'].add_(
                    0.5 * group['h'] * group['epsilon'] *\
                    (torch.sum(torch.square(state['p'])) - group['p'].shape[0] * state['tau']))

                # D_(h/2) step
                state['p'].add_(
                    torch.sqrt(0.5 * state['h']) *\
                    group['sigma'] *\
                    torch.randn(w.shape))

                # C_(h/2) step
                state['p'].mul_(torch.exp(-0.5 * group['h'] * state['xi']))

                # A_(h/2) step
                w.add_(0.5 * state['h'] * state['p'])

                # re-evaluate
                loss = closre()

                # B step
                state['p'].add_(-h * w.grad)
                    
                     
  
        

