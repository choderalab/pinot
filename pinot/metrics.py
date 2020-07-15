""" Metrics to evaluate and train models.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import pinot

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _independent(distribution):
    """ Make predictive distribution for test set independent.g

    Parameters
    ----------
    distribution : `torch.distribution.Distribution`
        Input distribution.

    Returns
    -------
    distribution : `torch.distribution.Distribution`
        Output distribution.

    """
    return torch.distributions.normal.Normal(
        loc=distribution.mean.flatten(),
        scale=distribution.variance.pow(0.5).flatten(),
    )


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def rmse(net, g, y, *args, n_samples=64, **kwargs):
    """ Rooted mean squarred error. """

    y_hat = net.condition(g, *args, **kwargs).sample([n_samples]).cpu()
    y = y.cpu()
    
    return ((y_hat - y) ** 2).mean(dim=1).pow(0.5).mean()

def r2(net, g, y, *args, n_samples=64, **kwargs):
    """ R2 """
    y_hat = net.condition(g, *args, **kwargs).sample([n_samples]).cpu()
    y = y.cpu()

    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum(dim=1)

    return (1 - torch.div(ss_res, ss_tot)).mean()


def pearsonr(net, g, y, *args, **kwargs):
    """ Pearson R """
    # NOTE: not adapted to sampling methods yet
    from scipy.stats import pearsonr as pr

    y_hat = net.condition(g, *args, **kwargs
        ).mean.detach().cpu()
    y = y.detach().cpu()

    result = pr(y.flatten().numpy(), y_hat.flatten().numpy())
    correlation, _ = result
    return torch.Tensor([correlation])[0]



def avg_nll(net, g, y, *args, **kwargs):
    """ Average negative log likelihood. """

    # TODO:
    # generalize

    # ensure `y` is one-dimensional
    assert (y.dim() == 2 and y.shape[-1] == 1) or (y.dim() == 1)

    # make the predictive distribution
    distribution = net.condition(g, *args, **kwargs)

    # make independent
    distribution = _independent(distribution)

    # calculate the log_prob
    log_prob = distribution.log_prob(y.flatten()).mean()

    return -log_prob
