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
    net_representation = pinot.representation.Sequential(
        layer,
        eval(args.config)) # output mu and sigma

    # get the last units as the input units of prediction layer
    param_in_units = list(filter(lambda x: type(x)==int, eval(args.config)))[-1]

    # construct a separated prediction net
    net_parameterization = pinot.regression.Linear(
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
    n_epochs = int(args.n_epochs)

    # define reporters
    now = datetime.now() 
    time_str = now.strftime("%Y-%m-%d-%H%M%S%f")
    os.mkdir(time_str)


    markdown_reporter = pinot.app.reporters.MarkdownReporter(
        time_str, ds_tr, ds_te, args=args, net=net)
    visual_reporter = pinot.app.reporters.VisualReporter(
        time_str, ds_tr, ds_te)
    weight_reporter = pinot.app.reporters.WeightReporter(
        time_str)

    reporters = [
        markdown_reporter,
        visual_reporter,
        weight_reporter]

    pinot.app.utils.train(net, ds_tr, ds_te, opt, reporters, n_epochs)

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

