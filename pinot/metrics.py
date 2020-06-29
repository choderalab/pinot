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
        scale=distribution.variance.pow(0.5).flatten()
    )

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def _mse(y, y_hat):
    """ Mean squarred error. """
    return torch.nn.functional.mse_loss(y, y_hat)


def mse(net, g, y, sampler=None):
    """ Mean squarred error. """

    y_hat = net.condition(g, sampler=sampler).mean.cpu()
    y = y.cpu()

    # gp
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    return _mse(y, y_hat)


def _rmse(y, y_hat):
    """ Rooted mean squarred error. """
    assert y.numel() == y_hat.numel()
    return torch.sqrt(
        torch.nn.functional.mse_loss(y.flatten(), y_hat.flatten())
    )


def rmse(net, g, y, sampler=None):
    """ Rooted mean squarred error. """
    y_hat = net.condition(g, sampler=sampler).mean.cpu()
    y = y.cpu()

    # gp
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    return _rmse(y, y_hat)


def _r2(y, y_hat):
    """ R2 """
    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum()
    return 1 - torch.div(ss_res, ss_tot)


def r2(net, g, y, sampler=None):
    """ R2 """
    y_hat = net.condition(g, sampler=sampler).mean.cpu()
    y = y.cpu()

    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    return _r2(y, y_hat)


def pearsonr(net, g, y, sampler=None):
    """ Pearson R """
    from scipy.stats import pearsonr as pr
    y_hat = net.condition(g, sampler=sampler).mean.detach().cpu()
    y = y.detach().cpu()

    result = pr(y.flatten().numpy(), y_hat.flatten().numpy())
    correlation, _ = result
    return torch.Tensor([correlation])[0]


def log_sigma(net, g, y, sampler=None):
    """ Log sigma. """
    return net.log_sigma

def avg_nll(net, g, y, sampler=None):
    """ Average negative log likelihood. """

    # TODO:
    # generalize

    # ensure `y` is one-dimensional
    assert ((y.dim() == 2 and y.shape[-1] == 1) or
        (y.dim() == 1)
    )

    # make the predictive distribution
    distribution = net.condition(g, sampler=sampler)

    # make independent
    distribution = _independent(distribution)

    # calculate the log_prob
    log_prob = distribution.log_prob(y.flatten()).mean()

    return -log_prob
