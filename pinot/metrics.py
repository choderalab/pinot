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


def mse(net, g, y):

    y_hat = net.condition(g).mean

    # gp
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    return _mse(y, y_hat)


def _rmse(y, y_hat):
    return torch.sqrt(torch.nn.functional.mse_loss(y, y_hat))


def rmse(net, g, y):
    y_hat = net.condition(g).mean

    # gp
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    return _rmse(y, y_hat)


def _r2(y, y_hat):
    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum()
    return 1 - torch.div(ss_res, ss_tot)


def r2(net, g, y):
    y_hat = net.condition(g).mean
    
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(1)

    return _r2(y, y_hat)

def avg_nll(net, g, y):
    
    # TODO:
    # generalize
    if isinstance(net, pinot.inference.gp.gpr.base_gpr.GPR):
        return net.loss(g, y).mean() / float(y.shape[0])

    return net.loss(g, y).mean()
