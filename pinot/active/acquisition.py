# =============================================================================
# IMPORTS
# =============================================================================
import torch
from pinot.metrics import _independent

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def temporal(distribution, y_best=0.0):
    r"""Picks the first in sequence.
    Designed to be used with temporal datasets to compare with baselines.

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    utility = torch.range(
        start=0,
        end=len(distribution.mean.flatten()) - 1
        ).flip(0)
    if torch.cuda.is_available():
        utility = utility.cuda()
    return utility


def probability_of_improvement(distribution, y_best=0.0):
    r""" Probability of Improvement (PI).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    return 1.0 - distribution.cdf(y_best)


def uncertainty(distribution, y_best=0.0):
    r""" Uncertainty.

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    return distribution.variance


def expected_improvement_analytical(distribution, y_best=0.0):
    r""" Analytical Expected Improvement (EI).

    Closed-form derivation assumes predictive posterior is a multivariate normal distribution.

    From https://arxiv.org/abs/1206.2944:

        EI(x) = (\mu(x) - f(x_best)) * cdf(Z)] + [\sigma(x) * pdf(Z)] if \sigma(x) > 0
                0                                                     if \sigma(x) = 0

        where

        Z = \frac{\mu(x) - f(x_best)}{\sigma(x)}

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    assert isinstance(distribution, torch.distributions.Normal)
    mu = distribution.mean
    sigma = distribution.stddev
    Z = (mu - y_best)/sigma

    normal = torch.distributions.Normal(0, 1)
    cdf = lambda x: normal.cdf(x)
    pdf = lambda x: torch.exp(normal.log_prob(x))
    return (mu - y_best) * cdf(Z) + sigma * pdf(Z)


def expected_improvement_monte_carlo(distribution, y_best, n_samples=1000):
    r""" Monte Carlo Expected Improvement (EI).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    n_samples : int
        The number of samples to use for the monte carlo estimation.

    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    improvement = torch.nn.functional.relu(distribution.sample((n_samples, )) - y_best)
    return improvement.mean(axis=0)


def upper_confidence_bound(distribution, y_best=0.0, kappa=0.95):
    r""" Upper Confidence Bound (UCB).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """

    from pinot.samplers.utils import confidence_interval

    _, high = confidence_interval(distribution, kappa)
    return high


def random(distribution, y_best=0.0, seed=2666):
    """ Random assignment of scores under normal distribution.

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.

    Note
    ----
    Random seed set to `2666`, the title of the single greatest novel in
    human literary history by Roberto Bolano.
    This needs to be set to `None` if parallel experiments were to be performed.
    """
    # torch.manual_seed(seed)
    return torch.rand(distribution.batch_shape)