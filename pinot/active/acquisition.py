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
