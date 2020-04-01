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
from datetime import datetime
import os

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def run(args):
    # get single layer
    layer = getattr(
        pinot.representation,
        args.model).GN

    # nest layers together to form the representation learning
    net = pinot.representation.Sequential(
        layer,
        eval(args.config)) # output mu and sigma

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
    
    # get the training specs
    lr = float(args.lr)
    opt = getattr(torch.optim, args.opt)(
        net.parameters(),
        lr)
    n_epochs = int(args.n_epochs)
    
    for _ in range(n_epochs):
        for g, y in ds:
            g = net(g, return_graph=True)
            loss = -torch.sum(pinot.metrics.semi_supervised.score(g))
            print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(net.state_dict(), 'semi-supervised.ds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='dgl_legacy')
    parser.add_argument(
        '--config', 
        default="[128, 0.1, 'tanh', 128, 0.1, 'tanh', 128, 0.1, 'sigmoid']")
    parser.add_argument('--data', default='esol')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--opt', default='Adam')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--partition', default='4:1', type=str)
    parser.add_argument('--n_epochs', default=10)
    
    args = parser.parse_args()
    run(args)

