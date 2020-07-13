# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# MODULE FUNCTIONS [SEQUENTIAL]
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
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.


    """
    score = torch.range(
        start=0,
        end=len(distribution.mean.flatten()) - 1
        ).flip(0)
    if torch.cuda.is_available():
        score = score.cuda()
    return score


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
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.
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
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.
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
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.
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
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.
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
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.
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
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.

    Note
    ----
    Random seed set to `2666`, the title of the single greatest novel in
    human literary history by Roberto Bolano.
    This needs to be set to `None` if parallel experiments were to be performed.

    """
    # torch.manual_seed(seed)
    return torch.rand(distribution.batch_shape)


# =============================================================================
# BATCH Utilities
# =============================================================================

def _get_utility(net, unseen_data, acq_func, y_best=0.0):
    """ Obtain distribution and utility from acquisition func.
    """
    # obtain predictive posterior
    gs, ys = unseen_data
    distribution = net.condition(gs)

    # obtain utility from vanilla acquisition func
    utility = acq_func(distribution, y_best=y_best)
    return utility

def _exponential_weighted_sampling(net, unseen_data, acq_func, q=5, y_best=0.0):
    """ Exponential weighted expected improvement
    """
    utility = _get_utility(
        net,
        unseen_data,
        acq_func,
        y_best=y_best
    )

    # generate probability distribution
    weights = torch.exp(-score)

    # normalize weights to make distribution
    weights_norm = weights/weights.sum()

    # perform multinomial sampling from exp-transformed acq score
    pending_pts = _multinomial_sampling(weights_norm)
    
    return pending_pts

def _multinomial_sampling(score, q=5):
    """
    """    
    # perform multinomial sampling
    pending_pts = WeightedRandomSampler(
        weights=weights_norm,
        num_samples=q,
        replacement=False)
    
    # convert to tensor
    pending_pts = torch.LongTensor(list(pending_pts))

    return pending_pts

def _greedy(net, unseen_data, acq_func, q=5, y_best=0.0):
    """ Greedy batch acquisition
    """
    utility = _get_utility(
        net,
        unseen_data,
        acq_func,
        y_best=y_best
    )
    
    # fill batch greedily
    pending_pts = torch.topk(utility, q).indices
    
    return pending_pts

# =============================================================================
# MODULE FUNCTIONS [BATCH]
# =============================================================================

def thompson_sampling(net, unseen_data, q=5, y_best=0.0):
    """ Generates m Thompson samples and maximizes them.
    
    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    # obtain predictive posterior
    gs, ys = unseen_data
    distribution = net.condition(gs)
    
    # obtain samples from posterior
    thetas = distribution.sample((q,))
    
    # enforce no duplicates in batch
    pending_pts = torch.unique(torch.argmax(thetas, axis=1)).tolist()
    
    while len(pending_pts) < q:
        theta = distribution.sample()
        pending_pts.append(torch.argmax(theta).item())
    
    # convert to tensor
    pending_pts = torch.LongTensor(pending_pts)
    
    return pending_pts


def exponential_weighted_ei_analytical(net, unseen_data, q=5, y_best=0.0):
    """ Exponential weighted expected improvement with analytical evaluation.

    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    pending_pts = _exponential_weighted_sampling(
        net,
        unseen_data,
        expected_improvement_analytical,
        q=q,
        y_best=y_best,
    )
    
    return pending_pts


def exponential_weighted_ei_monte_carlo(net, unseen_data, q=5, y_best=0.0):
    """ Exponential weighted expected improvement with monte carlo sampling.
    
    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    pending_pts = _exponential_weighted_sampling(
        net,
        unseen_data,
        expected_improvement_monte_carlo,
        q=q,
        y_best=y_best,
    )
    
    return pending_pts


def exponential_weighted_pi(net, unseen_data, q=5, y_best=0.0):
    """ Exponential weighted probability of improvement.

    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    pending_pts = _exponential_weighted_sampling(
        net,
        unseen_data,
        probability_of_improvement,
        q=q,
        y_best=y_best,
    )
    
    return pending_pts


def exponential_weighted_ucb(net, unseen_data, q=5, y_best=0.0):
    """ Exponential weighted expected improvement

    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    pending_pts = _exponential_weighted_sampling(
        net,
        unseen_data,
        upper_confidence_bound,
        q=q,
        y_best=y_best,
    )
    
    return pending_pts


def greedy_ucb(net, unseen_data, q=5, y_best=0.0):
    """ Greedy upper confidence bound

    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.    
    """
    pending_pts = _greedy(
        net,
        unseen_data,
        upper_confidence_bound,
        q=q,
        y_best=y_best,
    )
    
    return pending_pts


def greedy_ei_analytical(net, unseen_data, q=5, y_best=0.0):
    """ Greedy expected improvement, evaluated analytically.

    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    pending_pts = _greedy(
        net,
        unseen_data,
        expected_improvement_analytical,
        q=q,
        y_best=y_best,
    )
    
    return pending_pts


def greedy_ei_monte_carlo(net, unseen_data, q=5, y_best=0.0):
    """ Greedy expected improvement, evaluated using MC sampling.
    
    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    pending_pts = greedy(
        net,
        unseen_data,
        expected_improvement_monte_carlo,
        q=q,
        y_best=y_best,
    )
    
    return pending_pts


def greedy_pi(net, unseen_data, q=5, y_best=0.0):
    """ Greedy probability of improvement.

    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    pending_pts = _greedy(
        net,
        unseen_data,
        probability_of_improvement,
        q=q,
        y_best=y_best,
    )
    
    return pending_pts


def batch_random(net, unseen_data, q=5, y_best=0.0):
    """ Fill batch randomly.

    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    pending_pts = greedy(
        net,
        unseen_data,
        random,
        q=q,
        y_best=y_best,
    )

    return pending_pts


def batch_temporal(net, unseen_data, q=5, y_best=0.0):
    r"""Picks the first q points in sequence.
    Designed to be used with temporal datasets to compare with baselines.

    Parameters
    ----------
    net : pinot Net object
        Trained net.
    unseen_data : tuple
        Dataset from which pending points are selected.
    q : int
        Number of Thompson samples to obtain.
    y_best : float
        The best target value seen so far.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    pending_pts = greedy(
        net,
        unseen_data,
        temporal,
        q=q,
        y_best=y_best,
    )

    return pending_pts