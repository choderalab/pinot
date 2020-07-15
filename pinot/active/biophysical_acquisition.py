from pinot.metrics import _independent

# =============================================================================
# UTILITIES
# =============================================================================
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
        samples = _independent(net.condition(gs, concentration)).sample((n_samples,))
        assay_preds.append(samples)

    assay_preds = torch.cat(assay_preds)
    return assay_preds


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

def _biophysical_ei(assay_preds, y_best=0.0):
    r"""Expected Improvement (EI).
    """
    improvement = torch.nn.functional.relu(assay_preds - y_best)
    return improvement.mean(axis=0)

def _biophysical_pi(assay_preds, y_best=0.0):
    r""" Probability of Improvement (PI).
    """
    return (assay_preds > y_best).mean(axis=0)

def _biophysical_ucb(assay_preds, y_best=0.0, kappa=0.95):
    r""" Upper Confidence Bound (UCB).
    """
    kappa = 0.95
    n_samples = len(assay_preds)
    k = n_samples - int(kappa*n_samples)
    high = torch.topk(assay_preds, k, axis=0).values[-1]
    return high


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def biophysical_thompson_sampling(
    net,
    unseen_data,
    acq_func,
    q=5,
    y_best=0.0,
    concentration=20,
    dG_samples=10):
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

    # enforce no duplicates in batch
    pending_pts = set()
    while len(pending_pts) < q:
        pending_pt = _get_thompson_sample(
            net,
            unseen_data,
            concentration=concentration,
            dG_samples=dG_samples
        )
        pending_pts.add(pending_pt)

    # convert to tensor
    pending_pts = torch.LongTensor(list(pending_pts))
    
    return pending_pts



def biophysical_expected_improvement(
    net,
    unseen_data,
    y_best=0.0,
    concentration=20,
    dG_samples=10,
    n_samples=100):
    """ Expected Improvement
    """
    utility = _get_utility(
        net,
        unseen_data,
        _biophysical_ei,
        y_best=y_best,
        concentration=concentration,
        dG_samples=dG_samples,
        n_samples=n_samples,
    )
    return utility



def biophysical_probability_of_improvement(
    net,
    unseen_data,
    y_best=0.0,
    concentration=20,
    dG_samples=10,
    n_samples=100):
    """ Expected Improvement
    """
    utility = _get_utility(
        net,
        unseen_data,
        _biophysical_pi,
        y_best=y_best,
        concentration=concentration,
        dG_samples=dG_samples,
        n_samples=n_samples,
    )
    return utility



def biophysical_probability_of_improvement(
    net,
    unseen_data,
    y_best=0.0,
    concentration=20,
    dG_samples=10,
    n_samples=100):
    """ Expected Improvement
    """
    utility = _get_utility(
        net,
        unseen_data,
        _biophysical_ucb,
        y_best=y_best,
        concentration=concentration,
        dG_samples=dG_samples,
        n_samples=n_samples,
    )
    return utility