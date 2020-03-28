# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import torch
import dgl
import argparse
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def run(args):
    # get single layer
    layer = getattr(
        pinot.representation,
        args.model).GN

    # nest layers together to form the representation learning
    net_representation = pinot.representation.Sequential(
        layer,
        eval(args.config)) # output mu and sigma

    # get the last units as the input units of prediction layer
    param_in_units = list(filter(lambda x: type(x)==int, eval(args.config)))[-1]

    # construct a separated prediction net
    net_parameterization = pinot.parameterization.Linear(
        param_in_units,
        int(args.n_params))

    # get the distribution class
    distribution_class = getattr(
        getattr(
            torch.distributions,
            args.distribution.lower()),
        args.distribution.capitalize())

    net = pinot.Net(
        net_representation, 
        net_parameterization,
        distribution_class)

    # get the entire dataset
    ds= getattr(
        pinot.data,
        args.data)()

    # not normalizing for now
    # y_mean, y_std, norm, unnorm = pinot.data.utils.normalize(ds) 

    # get data specs
    batch_size = int(args.batch_size)
    partition = [int(x) for x in args.partition.split(':')]
    assert len(partition) == 2, 'only training and test here.'

    # batch
    ds = pinot.data.utils.batch(ds, batch_size)
    ds_tr, ds_te = pinot.data.utils.split(ds, partition)
    
    # get the training specs
    lr = float(args.lr)
    opt = getattr(torch.optim, args.opt)(
        net.parameters(),
        lr)
    n_epoches = int(args.n_epochs)
    
    losses_tr = []
    losses_te = []

    if args.report == True:
        from matplotlib import pyplot as plt
        plt.style.use('fivethirtyeight')
        import time
        from datetime import datetime
        import os

        now = datetime.now()

        time_str = now.strftime("%Y-%m-%d-%H%M%S%f")
        os.mkdir(time_str)

        losses = np.array([0.])
        time0 = time.time()

        f_handle = open(time_str + '/report.md', 'w')
        f_handle.write(time_str)
        f_handle.write('\n')
        f_handle.write('===========================')
        f_handle.write('\n')
        f_handle.write('# Model Summary\n')
        for arg in vars(args):
            f_handle.write(arg+ '=' + str(getattr(args, arg)))
            f_handle.write('\n')

        f_handle.write(str(net))
        f_handle.write('\n')

    # without further ado, train it
    for idx_epoch in range(n_epoches):
        net.train()
        for g, y in ds_tr:
            loss = torch.sum(net.loss(g, y))
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        
        loss_tr_this_epoch = [net.loss(g, y) for g, y in ds_tr]
        loss_te_this_epoch = [net.loss(g, y) for g, y in ds_te]
 
        # TODO: is sum right?
        losses_tr.append(torch.mean(torch.cat(loss_tr_this_epoch)).detach().numpy())
        losses_te.append(torch.mean(torch.cat(loss_te_this_epoch)).detach().numpy())

        if args.report == True:
            torch.save(net.state_dict(), time_str + '/w' + str(idx_epoch) + '.bin') 
            plt.figure()
            plt.plot(losses_tr, label='training')
            plt.plot(losses_te, label='test')
            plt.legend()
            plt.savefig(time_str + '/nll.png')


    time1 = time.time()
    f_handle.write('# Time used\n')
    f_handle.write(str(time1 - time0) + ' s\n')

    f_handle.write('# Performance \n')
    f_handle.write('{:<15}'.format('|'))
    f_handle.write('{:<15}'.format('|NLL')+ '|' + '\n')

    f_handle.write('{:<15}'.format('|' + '-' * 13))
    f_handle.write('{:<15}'.format('|' + '-' * 13))
    f_handle.write('|' + '\n')

    f_handle.write('{:<15}'.format('|TRAIN'))
    f_handle.write('{:<15}'.format('|%.2f' % losses_tr[-1]) + '|' + '\n')

    f_handle.write('{:<15}'.format('|TEST'))
    f_handle.write('{:<15}'.format('|%.2f' % losses_te[-1]) + '|' + '\n')


    f_handle.write('<div align="center"><img src="nll.jpg" width="600"></div>')
    f_handle.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='dgl_legacy')
    parser.add_argument(
        '--config', 
        default="[128, 0.1, 'tanh', 128, 0.1, 'tanh', 128, 0.1, 'tanh']")
    parser.add_argument('--distribution', default='normal')
    parser.add_argument('--n_params', default=2)
    parser.add_argument('--data', default='esol')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--opt', default='Adam')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--partition', default='4:1', type=str)
    parser.add_argument('--n_epochs', default=10)
    parser.add_argument('--report', default=True, type=bool) 
    args = parser.parse_args()
    run(args)

