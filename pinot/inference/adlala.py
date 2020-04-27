# =============================================================================
# IMPORTS
# =============================================================================
import torch


# =============================================================================
# MODULE CLASSES
# =============================================================================
class AdLaLa(torch.optim.Optimizer):
    """ Adaptive Langevin-Langevin Integrator.

    ref: `https://arxiv.org/abs/1908.11843`
    
    original problem-specific implementation:
        `https://github.com/`
        `TiffanyVlaar/`
        `ThermodynamicParameterizationOfNNs/`
        `blob/master/AdLaLa.ipynb`

    Attributes
    ----------
    params : iterator
        an iterator over a series of `torch.nn.Parameter` to be optimized.
    h : float, default=0.25
        learning rate.
    gamma : float, dafault=0.1
        friction parameter (for Langevin only).
    sigma : float, default=0.01
        driving noise amplitude (for Adaptive Langevin only).
    epsilon: float, default=0.01
        coupling coefficient (for Adaptive Langevin only).
    tau: float, default=1e-4
        temperature.
    xi_init: float, default=1e-3
        initial value for $\xi$.

    Methods
    -------
    step(closure): apply gradient.
        
    """
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

        Perform the steps in the following order:
        
        ```
        A^1_(h/2) 
        A^2_(h/2) 
        C^1_(h/2) 
        D^1_(h/2)
        E^1_(h/2)
        O^2_h
        E^1_(h/2)
        D^1_(h/2)
        C^1_(h/2)
        A^2_(h/2)
        A^1_(h/2)
        B^2_h
        B^1_h
        ```
        where 1 denotes adaptive Langevin part
        and 2 denotes Langevin part

        the steps are:
        ```
        A: w := w + hp
        B: p := p-h\\nabla L(w)
        C: p := e^{-h\\xi} p
        D: p := p + \\sigma \\sqrt{h}R_n
        E: \xi := \\xi + h\\epsilon [p^T p - N\\tau]
        O: p := c p + d R

        ```

        """
        # call closure if closure is specified
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # loop through param groups with possible different parameter settings.
        for group in self.param_groups:
            for w in group['params']:
                # NOTE: `w.grad == None` is different from `w.grad == 0.`
                # as the later could be used to sample parameters without taking grads.
                if w.grad == None:
                    continue

                state = self.state[w]

                # state initialization
                if len(state) == 0:

                    # initialize momentum
                    state['p'] = torch.zeros_like(w)
                    
                    # initialize $\xi$
                    state['xi'] = group['xi_init']

                    # specify the $c$ and $d$ parameters for Langevin
                    group['c'] = torch.exp(
                            -group['h'] * group['gamma'])
                    group['d'] = torch.sqrt(
                            1. - torch.exp(-2. * group['h'] * group['gamma']
                                )) * torch.sqrt(group['tau'])

                    # do a B_(h/2) when during initialization
                    # B_(h/2) step: `p := p - 0.5 * h * d_l`
                    state['p'].add_(-0.5 * group['h'] * w.grad)
                    
                # A_(h/2) step: `w := theta + 0.5 * h * p`
                w.add_(0.5 * group['h'] * state['p'])
        
        # C1 D1 E1 and O2 steps
        for group in self.param_groups:
            for w in group['params']:

                if w.grad == None:
                    continue

                state = self.state[w]
                if group['partition'].lower() == 'adla': # adaptive Langevin

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
                
                elif group['partition'].lower() == 'la': # Langevin
                    
                    # O step: p := c*p + d*R
                    state['p'].mul_(group['c'])
                    state['p'].add_(group['d'] * torch.randn(w.shape))

        for group in self.param_groups[::-1]:
            for w in group['params']:

                if w.grad == None:
                    continue

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

                if w.grad == None:
                    continue

                state = self.state[w]
                # B step
                state['p'].add_(-group['h'] * w.grad)
                    
