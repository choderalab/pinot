from pinot.metrics import _independent
from pinot.active.acquisition import _greedy

# =============================================================================
# UTILITIES
# =============================================================================

def _get_utility(
    net,
    unseen_data,
    acq_func,
    y_best=0.0,
    concentration=20,
    dG_samples=10,
    n_samples=100):
    """ Obtain distribution and utility from acquisition function.
    """
    # marginalize over distribution of possible delta Gs
    assay_preds = _sample_and_marginalize_delta_G(
        net,
        unseen_data,
        concentration=concentration,
        dG_samples=dG_samples,
        n_samples=m
    )

    # obtain utility from vanilla acquisition func
    utility = acq_func(assay_preds, y_best=y_best)
    return utility


def _sample_and_marginalize_delta_G(
    net,
    unseen_data,
    dG_samples=20,
    n_samples=100,
    concentration=20):
    """
    For multiple rounds of possible delta G values,
    marginalize compound measurement beliefs.

    Returns
    -------
    assay_preds : torch Tensor
        Has shape (dG_samples * n_samples, n_compounds)
    """
    gs, _ = unseen_data

    # for each round of sampling dG
    assay_preds = []
    for d in range(dG_samples):
        
        # obtain predictive posterior
        samples = _independent(
            net.condition(gs, concentration)
        ).sample((n_samples,)).detach()

        assay_preds.append(samples)

    print('Check shape of assay_preds following cat')
    import pdb; pdb.set_trace()
    assay_preds = torch.cat(assay_preds)
    return assay_preds


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def biophysical_thompson_sampling(
    net,
    unseen_data,
    acq_func,
    q=1,
    y_best=0.0,
    concentration=20,
    dG_samples=10,
    unique=True):
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

    unique : bool
        Enforce no duplicates in batch if True.

    Returns
    -------
    pending_pts : torch.LongTensor
        The indices corresponding to pending points.
    """
    def _get_thompson_sample(
        net,
        unseen_data,
        concentration=20,
        dG_samples=10,
        ):
        # marginalize over distribution of possible delta Gs
        thetas = _sample_and_marginalize_delta_G(
            net,
            unseen_data,
            concentration=concentration,
            dG_samples=dG_samples,
            n_samples=1
        )
        
        # get argmax, marginalizing across all distributions
        # (and unraveling the index [find the column of argmax])
        pending_pt = thetas.argmax() % thetas.shape[1]
        return pending_pt.item()

    # fill batch
    pending_pts = []
    while len(pending_pts) < q:

        # do thompson sampling
        pending_pt = _get_thompson_sample(
            net,
            unseen_data,
            concentration=concentration,
            dG_samples=dG_samples
        )

        if unique:
            # enforce no duplicates in batch
            pending_pts = set(pending_pts)
            pending_pts.add(pending_pt)
        
        else:
            pending_pts.append(pending_pt)

    # convert to tensor
    pending_pts = torch.LongTensor(list(pending_pts))
    
    return pending_pts



def biophysical_expected_improvement(
    net,
    unseen_data,
    y_best=0.0,
    q=1,
    concentration=20,
    dG_samples=10,
    n_samples=100):
    """ Expected Improvement
    """
    def _biophysical_ei(assay_preds, y_best=0.0):
        improvement = torch.nn.functional.relu(assay_preds - y_best)
        return improvement.mean(axis=0)

    utility = _get_utility(
        net,
        unseen_data,
        _biophysical_ei,
        y_best=y_best,
        concentration=concentration,
        dG_samples=dG_samples,
        n_samples=n_samples,
    )

    pending_pts = _greedy(
        utility,
        q=q,
    )    

    return pending_pts



def biophysical_probability_of_improvement(
    net,
    unseen_data,
    y_best=0.0,
    q=1,
    concentration=20,
    dG_samples=10,
    n_samples=100):
    """ Probability of Improvement
    """
    def _biophysical_pi(assay_preds, y_best=0.0):
        return (assay_preds > y_best).mean(axis=0)

    utility = _get_utility(
        net,
        unseen_data,
        _biophysical_pi,
        y_best=y_best,
        concentration=concentration,
        dG_samples=dG_samples,
        n_samples=n_samples,
    )

    pending_pts = _greedy(
        utility,
        q=q,
    )    

    return pending_pts


def biophysical_upper_confidence_bound(
    net,
    unseen_data,
    y_best=0.0,
    q=1,
    concentration=20,
    dG_samples=10,
    n_samples=100):
    """ Upper Confidence Bound Improvement
    """
    def _biophysical_ucb(assay_preds, y_best=0.0, kappa=0.95):
        kappa = 0.95
        n_samples = len(assay_preds)
        k = n_samples - int(kappa*n_samples)
        high = torch.topk(assay_preds, k, axis=0).values[-1]
        return high

    utility = _get_utility(
        net,
        unseen_data,
        _biophysical_ucb,
        y_best=y_best,
        concentration=concentration,
        dG_samples=dG_samples,
        n_samples=n_samples,
    )
    
    pending_pts = _greedy(
        utility,
        q=q,
    )    

    return pending_pts