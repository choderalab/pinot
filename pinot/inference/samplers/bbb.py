""" Make a Bayesian-by-backprop model from any torch.nn.Module.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import pinot
from pinot.inference.samplers.base_sampler import BaseSampler


# =============================================================================
# MODULE CLASSES
# =============================================================================
class BBB(BaseSampler):
    """ Gaussian Variational Posterior Bayesian-by-Backprop.
    """

    def __init__(
        self,
        optimizer,
        initializer_std=0.1,
        theta_prior=torch.distributions.normal.Normal(0, 1.0),
        sigma_lr=1e-5,
        kl_loss_scaling=1.0,
    ):

        self.optimizer = optimizer
        self.theta_prior = theta_prior
        self.kl_loss_scaling = kl_loss_scaling

        # TODO:
        # make this compilable with more than one param group
        assert (
            len(self.optimizer.param_groups) == 1
        ), "Now we only support one param group."

        # copy the original param group
        # this makes the hyperparameters stable
        sigma_param_group = self.optimizer.param_groups[0].copy()

        # initialize log_sigma
        sigma_param_group["params"] = [
            torch.distributions.normal.Normal(
                loc=torch.zeros_like(p), scale=initializer_std * torch.ones_like(p)
            )
            .sample()
            .abs()
            .log()
            for p in sigma_param_group["params"]
        ]

        sigma_param_group["lr"] = sigma_lr

        # append this to param_group
        self.optimizer.add_param_group(sigma_param_group)

        # initialize
        for p, sigma in zip(
            *[
                self.optimizer.param_groups[0]["params"],
                self.optimizer.param_groups[1]["params"],
            ]
        ):

            p.grad = torch.zeros_like(p)
            sigma.grad = torch.zeros_like(sigma)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def __str__(self):
        return "BBB_" + str(self.optimizer)

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

        for p, sigma in zip(
            *[
                self.optimizer.param_groups[0]["params"],
                self.optimizer.param_groups[1]["params"],
            ]
        ):

            state = self.optimizer.state[p]

            # sample a noise
            # $ \epsilon ~ \mathcal{N}(0, 1) $
            epsilon = torch.distributions.normal.Normal(
                torch.zeros_like(p), torch.ones_like(p)
            ).sample()

            # clone mu before perturbation
            mu = p.detach().clone()

            # perturb p to get theta
            # $ \theta = \mu + \sigma \epsilon $
            theta = p + epsilon * torch.exp(sigma)
            p.copy_(theta)

            # calculate kl loss and the gradients thereof
            with torch.enable_grad():
                mu.requires_grad = True
                theta.requires_grad = True
                sigma.requires_grad = True

                # compute the kl loss term here
                kl_loss = (
                    torch.distributions.normal.Normal(loc=mu, scale=torch.exp(sigma))
                    .log_prob(theta)
                    .sum()
                    - self.kl_loss_scaling * self.theta_prior.log_prob(theta).sum()
                )

            d_kl_d_mu = torch.autograd.grad(kl_loss, mu, retain_graph=True)
            d_kl_d_sigma = torch.autograd.grad(kl_loss, sigma, retain_graph=True)
            d_kl_d_theta = torch.autograd.grad(kl_loss, theta, retain_graph=False)

            # put the results in state dicts
            state["d_kl_d_mu"] = d_kl_d_mu[0]
            state["d_kl_d_sigma"] = d_kl_d_sigma[0]
            state["d_kl_d_theta"] = d_kl_d_theta[0]

            # keep track of perturbation noise for cancellation later
            state["mu"] = mu
            state["epsilon"] = epsilon

        # do one step with perturbed weights
        with torch.enable_grad():
            loss = closure()

        for p, sigma in zip(
            *[
                self.optimizer.param_groups[0]["params"],
                self.optimizer.param_groups[1]["params"],
            ]
        ):

            state = self.optimizer.state[p]

            # cancel the perturbation
            p.copy_(state["mu"])

            sigma.requires_grad = True

            sigma.backward(
                state["epsilon"] * torch.exp(sigma) * (p.grad + state["d_kl_d_theta"])
                + state["d_kl_d_sigma"]
            )

            # modify grad
            p.grad.add_(state["d_kl_d_mu"] + state["d_kl_d_theta"])

        # update parameters based on whatever schedule
        # `self.optimizer` proposes
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def sample_params(self):
        with torch.no_grad():
            for p, sigma in zip(
                *[
                    self.optimizer.param_groups[0]["params"],
                    self.optimizer.param_groups[1]["params"],
                ]
            ):

                p.copy_(
                    torch.distributions.normal.Normal(
                        loc=self.optimizer.state[p]["mu"], scale=torch.exp(sigma)
                    ).sample()
                )

    def expectation_params(self):
        with torch.no_grad():
            for p, sigma in zip(
                *[
                    self.optimizer.param_groups[0]["params"],
                    self.optimizer.param_groups[1]["params"],
                ]
            ):

                p.copy_(self.optimizer.state[p]["mu"])
