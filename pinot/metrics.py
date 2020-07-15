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
def _rmse(y, y_hat):
    """RMSE"""
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)
    assert y.shape == y_hat.shape
    assert y.dim() == 2
    assert y.shape[-1] == 1
    return torch.nn.functional.mse_loss(y, y_hat).pow(0.5)

def rmse(net, g, y, *args, n_samples=64, **kwargs):
    """ Rooted mean squarred error. """

    y = y.cpu()

    results = []
    for _ in range(n_samples):
        y_hat = _independent(
                net.condition(g, *args, **kwargs)
            ).sample().cpu()

        results.append(
            _rmse(y, y_hat))

    return torch.tensor(results).mean()



def _r2(y, y_hat):
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)
    assert y.shape == y_hat.shape
    assert y.dim() == 2
    assert y.shape[-1] == 1

    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum()

    return 1 - torch.div(ss_res, ss_tot)

def r2(net, g, y, *args, n_samples=64, **kwargs):
    """ R2 """

    y = y.cpu()

    results = []
    for _ in range(n_samples):
        y_hat = _independent(
                net.condition(g, *args, **kwargs)
            ).sample().cpu()

        results.append(
            _r2(y, y_hat))

    return torch.tensor(results).mean()


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
