# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def probability_of_improvement(distribution, y_best=0.0):
    r""" Probability of Improvement (PI).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution` object
        the predictive distribution.
    y_best : float or float tensor, default=0.0
        the best value so far.

    """

    return 1.0 - distribution.cdf(y_best)

def expected_improvement(distribution, y_best=0.0):
    r""" Expected Improvement (EI).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution` object
        the predictive distribution.
    y_best : float or float tensor, default=0.0
        the best value so far.

    """
    return distribution.mean - y_best

def upper_confidence_bound(distribution, y_best=0.0, kappa=0.5):
    r""" Upper Confidence Bound (UCB).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution` object
        the predictive distribution.
    y_best : float or float tensor, default=0.0
        the best value so far.

    """

    from pinot.inference.utils import confidence_interval
    _, high = confidence_interval(distribution, kappa)
    return high

def random(distribution, y_best=0.0, seed=2666):
    # torch.manual_seed(seed)
    return torch.rand(distribution.batch_shape)

def q_upper_confidence_bound(model, gs, q=10, kappa=0.5,
                             num_samples=1000,
                             sampler=SobolQMCNormalSampler,
                             objective=IdentityMCObjective):
    """Evaluate MC-based batch Upper Confidence Bound on the candidate set `gs`.

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].)

    `qUCB = E(max(mu + |Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.

    Args:
        X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
            points each.

    Returns:
        A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
        design points `X`, where `batch_shape'` is the broadcasted batch shape of
        model and input `X`.
    """
    def UCB(model, gs):
        posterior = model.posterior(gs)
        samples = model.sampler(posterior)
        obj = model.objective(samples)
        mean = obj.mean(dim=0)
        return mean + model.kappa_prime * (obj - mean).abs()
    
    model.sampler = sampler(num_samples)
    model.objective = objective()
    model.kappa_prime = math.sqrt(kappa * math.pi / 2)
    
    ucb_samples = torch.zeros((num_samples, gs.batch_size, q))
    for q_idx in range(q):
        ucb_samples[:,:,q_idx] = UCB(model, gs)

    # get batch samples
    indices = torch.randint(gs.batch_size, ucb_samples.shape[-2:])
    qucb_samples = torch.gather(ucb_samples, 1, indices.expand(ucb_samples.shape))
    qucb_samples = qucb_samples.max(dim=-1)[0].mean(dim=0)

    return indices, qucb_samples