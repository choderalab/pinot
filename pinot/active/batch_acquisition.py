from pinot.active.acquisition import (
    temporal,
    probability_of_improvement,
    expected_improvement_analytical,
    expected_improvement_monte_carlo,
    upper_confidence_bound,
    random
)

from pinot.metrics import _independent

# =============================================================================
# UTILITIES
# =============================================================================

def _get_utility(net, unseen_data, acq_func, y_best=0.0):
    """ Obtain distribution and utility from acquisition func.
    """
    # obtain predictive posterior
    gs, ys = unseen_data
    distribution = net.condition(gs)

    # workup distribution
    distribution = _independent(distribution)

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
    weights = torch.exp(-utility)

    # normalize weights to make distribution
    weights_norm = weights/weights.sum()

    # perform multinomial sampling from exp-transformed acq score
    pending_pts = _multinomial_sampling(weights_norm)
    
    return pending_pts

def _multinomial_sampling(weights, q=5):
    """
    """    
    # perform multinomial sampling
    pending_pts = torch.utils.data.WeightedRandomSampler(
        weights=weights,
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
# MODULE FUNCTIONS
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
    pending_pts = _greedy(
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
    pending_pts = _greedy(
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
    pending_pts = _greedy(
        net,
        unseen_data,
        temporal,
        q=q,
        y_best=y_best,
    )

    return pending_pts