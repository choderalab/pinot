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
    """ Make predictive distribution for test set independent.

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
def absolute_error(net, distribution, y, *args, **kwargs):
    """ Squared error. """
    y_hat = distribution.sample().detach().cpu().reshape(-1, 1)
    return torch.abs(y - y_hat)


def _rmse(y, y_hat):
    """RMSE"""
    y_hat = y_hat.unsqueeze(1) if y_hat.dim() == 1 else y_hat
    y = y.unsqueeze(1) if y.dim() == 1 else y
    assert y.shape == y_hat.shape
    assert y.dim() == 2
    assert y.shape[-1] == 1
    return torch.nn.functional.mse_loss(y, y_hat).pow(0.5)

def rmse(net, distribution, y, *args, n_samples=16, batch_size=32, **kwargs):
    """ Root mean squared error. """

    results = []
    for _ in range(n_samples):
        
        y_hat = distribution.sample().detach().cpu().reshape(-1, 1)
        results.append(_rmse(y, y_hat))

    return torch.tensor(results).mean()



def _r2(y, y_hat):
    y_hat = y_hat.unsqueeze(1) if y_hat.dim() == 1 else y_hat
    y = y.unsqueeze(1) if y.dim() == 1 else y
    assert y.shape == y_hat.shape
    assert y.dim() == 2
    assert y.shape[-1] == 1

    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum()

    return 1 - torch.div(ss_res, ss_tot)

def r2(net, distribution, y, *args, n_samples=16, batch_size=32, **kwargs):
    """ R2 """
    y_hat = distribution.mean.detach().cpu().reshape(-1, 1)
    return _r2(y, y_hat)


def pearsonr(net, distribution, y, *args, batch_size=32, **kwargs):
    """ Pearson R """
    # NOTE: not adapted to sampling methods yet
    from scipy.stats import pearsonr as pr

    y_hat = distribution.mean.detach().cpu().reshape(-1, 1)
    result = pr(y.flatten().numpy(), y_hat.flatten().numpy())
    correlation, _ = result

    return torch.Tensor([correlation])[0]



def avg_nll(net, distribution, y, *args, batch_size=32, **kwargs):
    """ Average negative log likelihood. """

    # TODO:
    # generalize

    # ensure `y` is one-dimensional
    assert (y.dim() == 2 and y.shape[-1] == 1) or (y.dim() == 1)

    # calculate the log_prob
    log_prob = distribution.log_prob(y).detach().cpu().mean()
    return -log_prob