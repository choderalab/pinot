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
def train_once(net, ds_tr, opt):
    """ Train the model for one batch.
    """
    for g, y in ds_tr:
        loss = torch.sum(net.loss(g, y))
        opt.zero_grad()
        loss.backward()
        opt.step()

    return net, opt

def train(net, ds_tr, ds_te, opt, reporters, n_epochs):
    [reporter.before() for reporter in reporters]

    for _ in range(n_epochs):
        net.train()
        net, opt = train_once(net, ds_tr, opt)
        net.eval()
        [reporter.during(net) for reporter in reporters]

    [reporter.after(net) for reporter in reporters]








