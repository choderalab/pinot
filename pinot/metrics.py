""" Metrics to evaluate and train models.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import pinot

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def _mse(y, y_hat):
    return torch.nn.functional.mse_loss(y, y_hat)


def mse(net, g, y, sampler=None):

    y_hat = net.condition(g, sampler=sampler).mean.cpu()
    y = y.cpu()

    # gp
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    return _mse(y, y_hat)


def _rmse(y, y_hat):
    assert(y.numel() == y_hat.numel())
    return torch.sqrt(torch.nn.functional.mse_loss(y.flatten(), y_hat.flatten()))


def rmse(net, g, y, sampler=None):
    y_hat = net.condition(g, sampler=sampler).mean.cpu()
    y = y.cpu()

    # gp
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    return _rmse(y, y_hat)


def _r2(y, y_hat):
    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum()
    return 1 - torch.div(ss_res, ss_tot)


def r2(net, g, y, sampler=None):
    y_hat = net.condition(g, sampler=sampler).mean.cpu()
    y = y.cpu()

    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    return _r2(y, y_hat)

def log_sigma(net, g, y, sampler=None):
    return net.log_sigma

def avg_nll(net, g, y, sampler=None):
    
    # TODO:
    # generalize
    if isinstance(net, pinot.inference.gp.gpr.base_gpr.GPR):
        distribution = net.condition(g)
        distribution = torch.distributions.normal.Normal(
                distribution.mean,
                distribution.variance.pow(0.5))

        return -distribution.log_prob(y.flatten()).mean()

    y = y.cpu()
    return -net.condition(g, sampler=sampler).log_prob(y).mean()
