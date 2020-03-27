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

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def run(args):
    # get single layer
    layer = getattr(
        pinot.models,
        args.model).GN

    # nest layers together
    net = pinot.models.Sequential(
        layer,
        eval(args.config))

    # get the entire dataset
    ds= getattr(
        pinot.data,
        args.data)()

    y_mean, y_std, norm, unnorm = pinot.data.utils.normalize(ds) 

    # get data specs
    batch_size = int(args.batch_size)
    partition = [int(x) for x in args.partition.split(':')]
    assert len(partition) == 2, 'only training and test here.'

    # batch
    ds = pinot.data.utils.batch(ds, batch_size)
    ds_tr, ds_te = pinot.data.utils.split(ds, partition)
    
    # get the training specs
    loss_fn_tr = getattr(pinot.metrics, args.loss_fn_tr)
    loss_fn_te = getattr(pinot.metrics, args.loss_fn_te)
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

    # without further ado, train it
    for idx_epoch in range(n_epoches):
        net.train()
        net, opt = pinot.app.utils.train(net, ds_tr, opt, loss_fn_te)
        net.eval()
        losses_tr.append(pinot.app.utils.test(net, ds_tr, loss_fn_te).detach().numpy())
        losses_te.append(pinot.app.utils.test(net, ds_te, loss_fn_te).detach().numpy())

        if args.report == True:
            torch.save(net.state_dict(), time_str + '/w' + str(idx_epoch) + '.bin') 
            plt.figure()
            plt.plot(losses_tr, label='training')
            plt.plot(losses_te, label='test')
            plt.legend()
            plt.savefig(time_str + '/' + args.loss_fn_te + '.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='dgl_legacy')
    parser.add_argument(
        '--config', 
        default="[128, 0.1, 'tanh', 128, 0.1, 'tanh', 128, 0.1, 'tanh']")
    parser.add_argument('--data', default='esol')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--opt', default='Adam')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--partition', default='4:1', type=str)
    parser.add_argument('--n_epochs', default=10)
    parser.add_argument('--loss_fn_tr', default='mse')
    parser.add_argument('--loss_fn_te', default='rmse')
    parser.add_argument('--report', default=True, type=bool)
    
    args = parser.parse_args()
    run(args)

