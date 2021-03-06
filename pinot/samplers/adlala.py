# =============================================================================
# IMPORTS
# =============================================================================
import torch
from pinot.samplers.base_sampler import BaseSampler

# =============================================================================
# MODULE CLASSES
# =============================================================================
class AdLaLa(BaseSampler):
    """Adaptive Langevin-Langevin Integrator.

    Apply the following kinds of update steps to different "partitions" (i.e. param_groups)

    A: w := w + hp
    B: p := p-h nabla L(w)
    C: p := e^{-h * xi} p
    D: p := p + sigma sqrt{h}R_n
    E: xi := xi + h * epsilon [p^T p - N * tau]
    O: p := c p + d R

    By default, apply "adaptive Langevin" to self.param_groups[0],
        and apply "Langevin" to self.param_groups[1].

    Specifically, apply the following sequence of substeps:
    A^0_(h/2) A^1_(h/2) C^0_(h/2) D^0_(h/2) E^0_(h/2) O^0_h E^0_(h/2) D^0_(h/2) C^0_(h/2) A^1_(h/2) A^0_(h/2) B^1_h B^0_h

    Parameters
    ----------

    Returns
    -------

    Methods
    -------
    step(closure): apply gradient

    TODO
    ----
    * h should be an optimizer-wide constant, not a per-group parameter
        --> want to just say "h" instead of "h * fraction" in substeps...
    * allow user-specified splittings
    * reduce code-duplication for iterating over parameters w in a group
        where w.grad is not None
    * resolve ambiguity about definition of N
    References
    ----------
    * [Leimkuhler, Matthews, Vlaar, 2019] Partitioned integrators for thermodynamic
        parameterization of neural networks
        https://arxiv.org/abs/1908.11843
    * [Vlaar, 2020] Implementation of AdLaLa in PyTorch
        https://github.com/TiffanyVlaar/ThermodynamicParameterizationOfNNs/blob/master/AdLaLa.ipynb
    """

    def __init__(
        self,
        params,
        h=0.25,
        gamma=0.1,
        sigma=0.01,
        epsilon=0.05,
        tau=1e-4,
        xi_init=1e-3,
    ):
        """
        Arguments
        ---------
        params : iterator
            an iterator over a series of `torch.nn.Parameter` to be optimized.
        h : float, default=0.25
            learning rate.
        gamma : float, default=0.1
            friction parameter (for Langevin only).
        sigma : float, default=0.01
            driving noise amplitude (for Adaptive Langevin only).
        epsilon: float, default=0.01
            coupling coefficient (for Adaptive Langevin only).
        tau: float, default=1e-4
            temperature.
        xi_init: float, default=1e-3
            initial value for $xi$.
        """

        defaults = dict(
            h=torch.tensor(h),
            tau=torch.tensor(tau),
            gamma=torch.tensor(gamma),
            sigma=torch.tensor(sigma),
            epsilon=torch.tensor(epsilon),
            xi_init=torch.tensor(xi_init),
        )

        self.initialized = False

        super(AdLaLa, self).__init__(params, defaults)

    def A_step(self, group, fraction=0.5):
        """A_fraction: w := w + (fraction h) p

        Parameters
        ----------
        group :

        fraction :
             (Default value = 0.5)

        Returns
        -------

        """
        h = self.param_groups[group]["h"]

        for w in self.param_groups[group]["params"]:
            if w.grad is not None:
                w.add_(fraction * h * self.state[w]["p"])

    def B_step(self, group, fraction=0.5):
        """B_fraction: p := p - (fraction h) nabla L(w)

        Parameters
        ----------
        group :

        fraction :
             (Default value = 0.5)

        Returns
        -------

        """
        h = self.param_groups[group]["h"]

        for w in self.param_groups[group]["params"]:
            if w.grad is not None:
                self.state[w]["p"].add_(-fraction * h * w.grad)

    def C_step(self, group, fraction=0.5):
        """C_fraction: p := e^{-(fraction h) * xi} p

        Parameters
        ----------
        group :

        fraction :
             (Default value = 0.5)

        Returns
        -------

        """
        h = self.param_groups[group]["h"]
        for w in self.param_groups[group]["params"]:
            if w.grad is not None:
                state = self.state[w]
                state["p"].mul_(torch.exp(-fraction * h * state["xi"]))

    def D_step(self, group, fraction=0.5):
        """D_fraction: p := p + sigma sqrt{fraction h} R_n

        Parameters
        ----------
        group :

        fraction :
             (Default value = 0.5)

        Returns
        -------

        """
        g = self.param_groups[group]
        h, sigma = g["h"], g["sigma"]

        for w in self.param_groups[group]["params"]:
            if w.grad is not None:
                state = self.state[w]
                state["p"].add_(
                    torch.sqrt(fraction * h) * sigma * torch.randn(w.shape)
                )

    def E_step(self, group, fraction=0.5):
        """E_fraction: xi := xi + (fraction h) * epsilon [p^T p - N * tau]
        where N is the number of number of parameters
        TODO: resolve ambiguity about whether N is the number of parameters in the group
            vs. the number of parameters in the current tensor

        Parameters
        ----------
        group :

        fraction :
             (Default value = 0.5)

        Returns
        -------

        """
        g = self.param_groups[group]
        h, tau, epsilon = g["h"], g["tau"], g["epsilon"]

        for w in self.param_groups[group]["params"]:
            if w.grad is not None:
                state = self.state[w]
                state["xi"].add_(
                    fraction
                    * h
                    * epsilon
                    * (
                        torch.sum(torch.pow(state["p"].flatten(), 2))
                        - torch.numel(state["p"]) * tau
                    )
                )

    def O_step(self, group, fraction=0.5):
        """O_fraction: p := c p + d R

        where
            c = exp(- (fraction h) gamma)
            d = sqrt(1 - exp(-2 (fraction h) gamma)) sqrt(tau)

        Parameters
        ----------
        group :

        fraction :
             (Default value = 0.5)

        Returns
        -------

        """
        g = self.param_groups[group]
        h, tau, gamma = g["h"], g["tau"], g["gamma"]

        for w in self.param_groups[group]["params"]:
            if w.grad is not None:
                # TODO: these are just scalars, may not have to be torched?
                dt = fraction * h
                c = torch.exp(-dt * gamma)
                d = torch.sqrt(1 - torch.exp(-2.0 * dt * gamma)) * torch.sqrt(
                    tau
                )

                state = self.state[w]
                state["p"] = (c * state["p"]) + (d * torch.randn(w.shape))

    def initialize(self, closure):
        """Initialize state and group parameters, and take a half kick step.

        Initialize state:
            * p : initialized to 0
                # TODO: should this be initialized using layer temperature?
            * xi : initialized to xi_init for any layers treated by Adaptive Langevin

        Initialized group variables:
            * c, d : parameters for Langevin OU step

        For all layers i, take a B^i_(h/2) step.

        Parameters
        ----------
        closure :


        Returns
        -------

        """

        # call closure
        with torch.enable_grad():
            closure()

        for group in self.param_groups:
            for w in group["params"]:
                # NOTE: `w.grad == None` is different from `w.grad == 0.`
                # as the later could be used to sample parameters without taking grads.
                if (
                    w.grad is None
                ):  # skip network params not contributing to loss.
                    continue

                state = self.state[w]

                # state initialization
                if len(state) == 0:
                    # initialize momentum
                    state["p"] = torch.zeros_like(w)

                    # initialize $\xi$
                    state["xi"] = group["xi_init"]

                    # TODO: can remove these...
                    # specify the $c$ and $d$ parameters for Langevin
                    group["c"] = torch.exp(-group["h"] * group["gamma"])
                    group["d"] = torch.sqrt(
                        1.0 - torch.exp(-2.0 * group["h"] * group["gamma"])
                    ) * torch.sqrt(group["tau"])

        # TODO: a B_(h/2) step is performed during initialization stage, but
        #   should this instead be a B_h step?
        for i in range(len(self.param_groups)):
            self.B_step(i, 0.5)  # requires gradient

    @torch.no_grad()
    def step(self, closure=None):
        """Composition of Langevin and Adaptive Langevin substeps.

        Parameters
        ----------
        closure :
             (Default value = None)

        Returns
        -------

        """

        # make sure closure is specified
        assert closure is not None, "Closure is needed in the training loop."

        if not self.initialized:
            self.initialize(closure)
            self.initialized = True

        # execute substeps that don't require gradients
        self.A_step(0, 0.5)
        self.A_step(1, 0.5)
        self.C_step(0, 0.5)
        self.D_step(0, 0.5)
        self.E_step(0, 0.5)
        self.O_step(1, 1.0)
        self.E_step(0, 0.5)
        self.D_step(0, 0.5)
        self.C_step(0, 0.5)
        self.A_step(1, 0.5)
        self.A_step(0, 0.5)

        # call closure again, so that gradients are up-to-date
        with torch.enable_grad():
            closure()

        self.B_step(1, 1.0)  # requires gradient
        self.B_step(0, 1.0)  # requires gradient

    @torch.no_grad()
    def sample_params(self):
        """ """
        self.zero_grad()

        #
        # def closure():
        #     """ """
        #     for group in self.param_groups:
        #         for w in group["params"]:
        #             w.backward(torch.zeros_like(w))

        # self.step(closure)

        self.D_step(0, 0.5)
        self.E_step(0, 0.5)
        self.O_step(1, 1.0)
        self.E_step(0, 0.5)
        self.D_step(0, 0.5)

    @torch.no_grad()
    def expectation_params(self):
        """ """
        # TODO:
        # is there anything we can do here?
        pass
