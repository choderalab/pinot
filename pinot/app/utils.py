# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import torch
import dgl
import numpy as np

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def train(net, ds_tr, opt, loss_fn):
    """ Train the model for one batch.
    """

    for g, y in ds_tr: # loop through the dataset
        y_hat = net(g)
        loss = loss_fn(y, y_hat)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return net, opt


def test(net, ds_te, loss_fn, batched=True):
    """ Test the model on the test set.
    """ 
    # init 
    ys = []
    ys_hat = []

    for g, y in ds_te: # loop through the dataset
        ys.append(y)
        ys_hat.append(net(g))

    if batched==True:
        ys = torch.cat(ys)
        ys_hat = torch.cat(ys_hat)

    else:
        ys = torch.stack(ys)
        ys_hat = torch.stack(ys_hat)

    return loss_fn(ys, ys_hat)


    







