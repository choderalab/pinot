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
        def l():
            loss = torch.sum(net.loss(g, y))
            opt.zero_grad()
            loss.backward()
            return loss
        opt.step(l)

    return net, opt


def train(net, ds_tr, ds_te, opt, reporters, n_epochs):
    [reporter.before() for reporter in reporters]

    for _ in range(n_epochs):
        net.train()
        net, opt = train_once(net, ds_tr, opt)
        net.eval()
        [reporter.during(net) for reporter in reporters]

    [reporter.after(net) for reporter in reporters]


def optimizer_translation(opt_string, lr, *args, **kwargs):

    if opt_string.lower() == 'bbb':
        opt = lambda net: pinot.BBB(
                torch.optim.Adam(net.parameters(), lr),
                0.01,
                kl_loss_scaling=kwargs['kl_loss_scaling'])

    elif opt_string.lower() == 'sgld':
        opt = lambda net: pinot.SGLD(
                net.parameters(),
                lr)

    elif opt_string.lower() == 'adlala':
        lr = torch.tensor(lr)
        if torch.cuda.is_available():
            lr = lr.cuda()

        opt = lambda net: pinot.AdLaLa(
                [
                    {
                        'params': net.representation.parameters(),
                        'h': lr,
                        'gamma': 1e-6
                    },
                    {
                        'params': net._output_regression.parameters(),
                        'h': lr,
                        'gamma': 1e-6
                    }
                ])

    else:
        if 'weight_decay' in kwargs:
            opt = lambda net: getattr(torch.optim, opt_string
                )(net.parameters(), lr, weight_decay=kwargs['weight_decay'])
        else:
            opt = lambda net: getattr(torch.optim, opt_string
                )(net.parameters(), lr)
    return opt
