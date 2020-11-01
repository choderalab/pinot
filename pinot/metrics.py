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
    y_hat = y_hat.unsqueeze(1) if y_hat.dim() == 1 else y_hat
    y = y.unsqueeze(1) if y.dim() == 1 else y
    assert y.shape == y_hat.shape
    assert y.dim() == 2
    assert y.shape[-1] == 1
    return torch.nn.functional.mse_loss(y, y_hat).pow(0.5)

def rmse(net, ds, *args, n_samples=16, batch_size=32, **kwargs):
    """ Root mean squared error. """

    # get every y in one tensor
    _, y = ds.batch(len(ds))[0]
    y = y.detach().cpu().reshape(-1, 1)

    results = []
    for _ in range(n_samples):
        
        y_hat = torch.zeros(y.shape)
        for idx, d in enumerate(ds.batch(batch_size, partial_batch=True)):
            
            g_batch, _ = d
            distribution_batch = _independent(net.condition(g_batch, *args, **kwargs))
            y_hat_batch = distribution_batch.sample().detach().cpu().reshape(-1, 1)
            y_hat[idx * batch_size : (idx + 1) * batch_size] = y_hat_batch

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

def r2(net, ds, *args, n_samples=16, batch_size=32, **kwargs):
    """ R2 """
    _, y = ds.batch(len(ds))[0]
    y = y.detach().cpu().reshape(-1, 1)

    y_hat = torch.zeros(y.shape)
    for idx, d in enumerate(ds.batch(batch_size, partial_batch=True)):
    
        g_batch, _ = d
        distribution = _independent(net.condition(g_batch, *args, **kwargs))
        y_hat_batch = distribution.mean.detach().cpu().reshape(-1, 1)
        y_hat[idx * batch_size : (idx + 1) * batch_size] = y_hat_batch
    
    return _r2(y, y_hat)


def pearsonr(net, ds, *args, batch_size=32, **kwargs):
    """ Pearson R """
    # NOTE: not adapted to sampling methods yet
    from scipy.stats import pearsonr as pr

    _, y = ds.batch(len(ds))[0]
    y = y.detach().cpu().reshape(-1, 1)

    y_hat = torch.zeros(y.shape)
    for idx, d in enumerate(ds.batch(batch_size, partial_batch=True)):
    
        g_batch, _ = d
        distribution = _independent(net.condition(g_batch, *args, **kwargs))
        y_hat_batch = distribution.mean.detach().cpu().reshape(-1, 1)
        y_hat[idx * batch_size : (idx + 1) * batch_size] = y_hat_batch

    result = pr(y.flatten().numpy(), y_hat.flatten().numpy())
    correlation, _ = result

    return torch.Tensor([correlation])[0]



def avg_nll(net, ds, *args, batch_size=32, **kwargs):
    """ Average negative log likelihood. """

    # TODO:
    # generalize

    _, y = ds.batch(len(ds))[0]
    y = y.detach().reshape(-1, 1)

    log_probs = torch.zeros(y.shape)
    for idx, d in enumerate(ds.batch(batch_size, partial_batch=True)):
        
        g_batch, _ = d
        
        # ensure `y` is one-dimensional
        assert (y.dim() == 2 and y.shape[-1] == 1) or (y.dim() == 1)
        
        # make the predictive distribution
        distribution = net.condition(g_batch, *args, **kwargs)

        # make independent
        distribution = _independent(distribution)

        # calculate the log_prob
        y_batch = y[idx * batch_size : (idx + 1) * batch_size].flatten()
        log_probs[idx * batch_size : (idx + 1) * batch_size] = distribution.log_prob(y_batch).reshape(-1, 1)
    
    log_prob = log_probs.mean()

    return -log_prob
