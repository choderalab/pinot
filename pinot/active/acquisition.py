# =============================================================================
# IMPORTS
# =============================================================================
import torch
from pinot.metrics import _independent

# =============================================================================
# UTILITIES
# =============================================================================
def _get_utility(net, unseen_data, acq_func, y_best=0.0):
    """ Obtain distribution and utility from acquisition func.
    """
    # obtain predictive posterior
    gs, ys = unseen_data
    distribution = _independent(net.condition(gs))

    # obtain utility from vanilla acquisition func
    utility = acq_func(distribution, y_best=y_best)
    return utility

def _greedy(utility, q=1):
    """ Greedy batch acquisition
    """
    # fill batch greedily
    pending_pts = torch.topk(utility, min(q, len(utility))).indices
    
    return pending_pts

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def thompson_sampling(net, unseen_data, y_best=0.0, q=1, unique=True):
    """ Generates m Thompson samples and maximizes them.
    
    Parameters
    ----------
    net : pinot Net object
        Trained net.

    unseen_data : tuple
        Dataset from which pending points are selected.

    y_best : float
        The best target value seen so far.

    q : int
        Number of Thompson samples to obtain.

    unique : bool
        Enforce no duplicates in batch if True.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    # obtain predictive posterior
    gs, ys = unseen_data
    distribution = _independent(net.condition(gs))
    
    # obtain samples from posterior
    thetas = distribution.sample((q,))
    pending_pts = torch.argmax(thetas, axis=1)
    
    if unique:

        # enforce no duplicates in batch
        pending_pts = torch.unique(pending_pts).tolist()
        
        while len(pending_pts) < q:
            theta = distribution.sample()
            pending_pts.append(torch.argmax(theta).item())
    
    # convert to tensor
    pending_pts = torch.LongTensor(pending_pts)
    
    return pending_pts


def temporal(net, unseen_data, y_best=0.0, q=1):
    r"""Picks the first in sequence.
    Designed to be used with temporal datasets to compare with baselines.

    Parameters
    ----------
    Parameters
    ----------
    net : pinot Net object
        Trained net.

    unseen_data : tuple
        Dataset from which pending points are selected.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)
    
    q : int
        Number of points to add to the batch.


    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    def _temporal(distribution, y_best=0.0):
        utility = torch.range(
            start=0,
            end=len(distribution.mean.flatten()) - 1
            ).flip(0)
        if torch.cuda.is_available():
            utility = utility.cuda()
        return utility

    utility = _get_utility(
        net,
        unseen_data,
        _temporal,
        y_best=y_best
    )
    

    pending_pts = _greedy(
        utility,
        q=q,
    )
    
    return pending_pts


def probability_of_improvement(net, unseen_data, y_best=0.0, q=1):
    r""" Probability of Improvement (PI).

    Parameters
    ----------
    net : pinot Net object
        Trained net.

    unseen_data : tuple
        Dataset from which pending points are selected.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)
    
    q : int
        Number of points to add to the batch.


    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    def _pi(distribution, y_best=0.0):
        return 1.0 - distribution.cdf(y_best)

    utility = _get_utility(
        net,
        unseen_data,
        _pi,
        y_best=y_best
    )
    

    pending_pts = _greedy(
        utility,
        q=q,
    )
    
    return pending_pts


def uncertainty(net, unseen_data, y_best=0.0, q=1):
    r""" Uncertainty.

    Parameters
    ----------
    net : pinot Net object
        Trained net.

    unseen_data : tuple
        Dataset from which pending points are selected.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)
    
    q : int
        Number of points to add to the batch.


    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    def _uncertainty(distribution, y_best=0.0):
        return distribution.variance

    utility = _get_utility(
        net,
        unseen_data,
        _uncertainty,
        y_best=y_best
    )
    

    pending_pts = _greedy(
        utility,
        q=q,
    )
    
    return pending_pts


def expected_improvement_analytical(net, unseen_data, y_best=0.0, q=1):
    r""" Analytical Expected Improvement (EI).

    Closed-form derivation assumes predictive posterior is a multivariate normal distribution.

    From https://arxiv.org/abs/1206.2944:

        EI(x) = (\mu(x) - f(x_best)) * cdf(Z)] + [\sigma(x) * pdf(Z)] if \sigma(x) > 0
                0                                                     if \sigma(x) = 0

        where

        Z = \frac{\mu(x) - f(x_best)}{\sigma(x)}

    Parameters
    ----------
    net : pinot Net object
        Trained net.

    unseen_data : tuple
        Dataset from which pending points are selected.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)
    
    q : int
        Number of points to add to the batch.


    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    def _ei_analytical(distribution, y_best=0.0):
        assert isinstance(distribution, torch.distributions.Normal)
        mu = distribution.mean
        sigma = distribution.stddev
        Z = (mu - y_best)/sigma

        normal = torch.distributions.Normal(0, 1)
        cdf = lambda x: normal.cdf(x)
        pdf = lambda x: torch.exp(normal.log_prob(x))
        return (mu - y_best) * cdf(Z) + sigma * pdf(Z)

    utility = _get_utility(
        net,
        unseen_data,
        _ei_analytical,
        y_best=y_best
    )
    

    pending_pts = _greedy(
        utility,
        q=q,
    )
    
    return pending_pts


def expected_improvement_monte_carlo(net, unseen_data, y_best=0.0, q=1, n_samples=1000):
    r""" Monte Carlo Expected Improvement (EI).

    Parameters
    ----------
    net : pinot Net object
        Trained net.

    unseen_data : tuple
        Dataset from which pending points are selected.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    q : int
        Number of points to add to the batch.

    n_samples : int
        The number of samples to use for the monte carlo estimation.        

    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    def _ei_monte_carlo(distribution, y_best, n_samples=1000):
        improvement = torch.nn.functional.relu(
            distribution.sample((n_samples, )) - y_best
        )
        return improvement.mean(axis=0)

    utility = _get_utility(
        net,
        unseen_data,
        _ei_monte_carlo,
        y_best=y_best
    )
    

    pending_pts = _greedy(
        utility,
        q=q,
    )
    
    return pending_pts


def upper_confidence_bound(net, unseen_data, y_best=0.0, q=1, kappa=0.95):
    r""" Upper Confidence Bound (UCB).

    Parameters
    ----------
    net : pinot Net object
        Trained net.

    unseen_data : tuple
        Dataset from which pending points are selected.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)
    
    q : int
        Number of points to add to the batch.

    Returns
    -------
    utility : `torch.Tensor`, `shape=(n_candidates, )`
        Utility for candidates under predictive distribution.
    """
    def _ucb(distribution, y_best=0.0, kappa=0.95):
        from pinot.samplers.utils import confidence_interval
        _, high = confidence_interval(distribution, kappa)
        return high

    utility = _get_utility(
        net,
        unseen_data,
        _ucb,
        y_best=y_best
    )
    

    pending_pts = _greedy(
        utility,
        q=q,
    )
    
    return pending_pts


def random(net, unseen_data, y_best=0.0, q=1, seed=2666):
    """ Random assignment of scores under normal distribution.

    Parameters
    ----------
    net : pinot Net object
        Trained net.

    unseen_data : tuple
        Dataset from which pending points are selected.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)
    
    q : int
        Number of points to add to the batch.

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
    def _random(distribution, y_best=0.0, seed=2666):
        return torch.rand(distribution.batch_shape)

    # torch.manual_seed(seed)
    utility = _get_utility(
        net,
        unseen_data,
        _random,
        y_best=y_best
    )
    

    pending_pts = _greedy(
        utility,
        q=q,
    )
    
    return pending_pts