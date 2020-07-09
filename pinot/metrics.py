""" Metrics to evaluate and train models.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
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

def _bootstrap(metric, confidence_interval=0.95, n_samples=100):

    def _metric(y, y_hat):
        # sampled indices with replacement
        idxs = np.random.choice(
            list(range(y.shape[0])),
            size=y.shape,
            replace=True,
        ).tolist()

        return metric(y[idxs], y_hat[idxs])

    def bootstrapped_metric(y, y_hat):
        # repeat metrics multiple times
        _metrics = torch.tensor(
            [_metric(y, y_hat) for _ in range(n_samples)]
        )

        # sort
        _metrics, _ = torch.sort(_metrics)

        # indices
        low_idx = int((1 - confidence_interval) * 0.5 * n_samples)
        high_idx = n_samples - low_idx

        # get values
        low = _metrics[low_idx]
        high = _metrics[high_idx]

        return low, high

    def reported_metric(y, y_hat):
        # get confidence interval
        low, high = bootstrapped_metric(y, y_hat)

        # get the original value
        original = metric(y, y_hat)

        return original, low, high

    return reported_metric


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def _mse(y, y_hat):
    """ Mean squarred error. """
    return torch.nn.functional.mse_loss(y, y_hat)


def mse(net, g, y, sampler=None, bootstrap=False, **kwargs):
    """ Mean squarred error. """

    y_hat = net.condition(g, sampler=sampler).mean.cpu()
    y = y.cpu()

    # gp
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    if bootstrap is True:
        return _bootstrap(_mse, **kwargs)(y, y_hat)

    return _mse(y, y_hat)


def _rmse(y, y_hat):
    """ Rooted mean squarred error. """
    assert y.numel() == y_hat.numel()
    return torch.sqrt(
        torch.nn.functional.mse_loss(y.flatten(), y_hat.flatten())
    )


def rmse(net, g, y, sampler=None, bootstrap=False, **kwargs):
    """ Rooted mean squarred error. """
    y_hat = net.condition(g, sampler=sampler).mean.cpu()
    y = y.cpu()

    # gp
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)


    if bootstrap is True:
        return _bootstrap(_rmse, **kwargs)(y, y_hat)


    return _rmse(y, y_hat)


def _r2(y, y_hat):
    """ R2 """
    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum()
    return 1 - torch.div(ss_res, ss_tot)


def r2(net, g, y, sampler=None, bootstrap=False, **kwargs):
    """ R2 """
    y_hat = net.condition(g, sampler=sampler).mean.cpu()
    y = y.cpu()

    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    if bootstrap is True:
        return _bootstrap(_r2, **kwargs)(y, y_hat)

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


def avg_nll(
        net, g, y, sampler=None, bootstrap=False,
        confidence_interval=0.95, n_samples=100,
    ):

    """ Average negative log likelihood. """

    # TODO:
    # generalize

    # ensure `y` is one-dimensional
    assert (y.dim() == 2 and y.shape[-1] == 1) or (y.dim() == 1)

    # make the predictive distribution
    distribution = net.condition(g, sampler=sampler)

    # make independent
    distribution = _independent(distribution)

    # calculate the log_prob
    log_prob = distribution.log_prob(y.flatten()).mean()

    # if bootstrap
    if bootstrap is True:
        _log_prob = distribution.log_prob(y.flatten())

        # sampled indices with replacement
        idxs = np.random.choice(
            list(range(_log_prob.shape[0])),
            size=_log_prob.shape,
            replace=True,
        ).tolist()

        _log_prob = _log_prob[idxs]

        # sort
        _metrics, _ = torch.sort(_log_prob)

        # indices
        low_idx = int((1 - confidence_interval) * 0.5 * n_samples)
        high_idx = n_samples - low_idx

        # get values
        low = _metrics[low_idx]
        high = _metrics[high_idx]

        return -log_prob, -low, -high


    return -log_prob
