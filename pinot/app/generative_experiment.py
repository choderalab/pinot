import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=64, help="Embedding dimension (dimension of the encoder's output)")
parser.add_argument('--unlabelled_data', type=str, default="zinc_tiny", help="Background data to pre-train generative model on")
parser.add_argument('--labelled_data', type=str, default="esol", help="Labelled data for supervised metrics")
parser.add_argument('--n_epochs', type=int, default=20, help="Number of epochs of generative model pre-training")
parser.add_argument('--out', type=str, default="gen_result", help="Folder to save generative training results to")
parser.add_argument('--save_model', type=str, default="generative_model.pkl", help="File to save generative model to")
parser.add_argument('--info', action="store_true", help="INFO mode with more information printing out")
args = parser.parse_args()


# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math
from datetime import datetime
import os
from abc import ABC
import copy
import pinot
import time
from pinot.generative.torch_gvae.model import GCNModelVAE
from pinot.data.utils import split, batch
from torch import optim
from pinot.app.experiment import Train, Test
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# =============================================================================
# HELPER FUNCTION
# =============================================================================

def lr_rmse(net, g, y):
    # Assess the "quality" of the representations learned
    # by measuring the mean squared error of a linear regression
    # model trained on the graph representations
    # list of N vectors
    X = net(g).detach()
    lr = LinearRegression()
    lr.fit(X, y)
    yhat = torch.tensor(lr.predict(X))
    return torch.sqrt(torch.mean((yhat-y)**2))

def gp_rmse(net, g, y):
    # list of N vectors
    X = net(g).detach()
    regressor = GaussianProcessRegressor(RBF(1.0))
    regressor.fit(X, y)
    yhat = torch.tensor(regressor.predict(X))
    return torch.sqrt(torch.mean((yhat-y)**2))

def negative_elbo_loss(net, g, y):
    return net.loss(g).detach()

class BaseRepresentation(torch.nn.Module):
    def __init__(self, in_features, out_features, forward_mode="sum"):
        super(BaseRepresentation, self).__init__()
        self.identity_transformation = torch.nn.Linear(in_features, out_features)
        self.forward_mode = forward_mode
        
    def forward(self, g):
        if self.forward_mode == "sum":
            return dgl.sum_nodes(g, "h")
        else:
            return dgl.mean_nodes(g, "h")

def plot_metrics_with_base_results(ax, metric_names, results, base_results, title):
    for metric_name in metric_names:
        metric_arr = results[metric_name]
        metric_track = [metric_arr[i] for i in list(metric_arr.keys())[:-1]]
        metric_track = [base_results[metric_name][0]] + metric_track
        ax.plot(metric_track)
    ax.legend(metric_names)
    ax.set_title(title)

# =============================================================================
# EXPERIMENT
# =============================================================================

# Config file needs to supply
# LayerType, [ConvolutionLayerArchitecture], Optimizer, StepSize

# Hidden dimensions
hidden_dimensions = [
    # [64, 64],
    # [128],
    # [256],
    # [128, 128], 
    # [256, 128],
    [256, 256],
    [128, 128, 128],
]

# Types of graph convolution layer and init arguments required
layer_type = "GraphConv"
gcn_init_args = {}


unlabelled_volume = [0.2, 0.4] #, 0.6, 0.8, 1.]


# Load data set
start = time.time()
unlabelled_data = getattr(pinot.data, args.unlabelled_data)()
labelled_data   = getattr(pinot.data, args.labelled_data)()
end = time.time()
print("Loaded {} molecules from {} and {} molecules from {} in {} seconds".format(len(labelled_data), args.labelled_data, len(unlabelled_data), args.unlabelled_data, end-start))

# Batch the unlabelled data
batch_size = 32
batched_background_data = batch(unlabelled_data, batch_size)
feat_dim = batched_background_data[0][0].ndata["h"].shape[1]


fig, ax = plt.subplots(len(hidden_dimensions), len(unlabelled_volume), sharex='col', sharey='row')
fig.set_size_inches(9 * len(hidden_dimensions), 2.5*len(unlabelled_volume))


for i, hidden_dims in enumerate(hidden_dimensions):
    # For each hidden dimensions combinations
    # For different sizes of the unlabelled data
    for j, volume in enumerate(unlabelled_volume):
        num_instances = int(volume * len(batched_background_data))

        num_atom_types = 100
        gvae = GCNModelVAE(feat_dim, layer_type, gcn_init_args,
            gcn_hidden_dims=hidden_dims,
            embedding_dim=64, num_atom_types=num_atom_types)
        optimizer = optim.Adam(gvae.parameters(), 1e-4)

        # First do unsupervised training
        start = time.time()
        train_generative = Train(gvae, batched_background_data[:num_instances], optimizer, args.n_epochs)
        train_generative.train()
        states = train_generative.states

        # Then compute the given metrics during training on the LABELLED data
        test = Test(gvae, labelled_data, states, [lr_rmse, gp_rmse])
        results = test.test()

        # Produce base line comparisons
        base_net = BaseRepresentation(1, 1, "sum")
        base_test = Test(base_net, labelled_data, {0 : base_net.state_dict()}, [lr_rmse, gp_rmse])
        base_results = base_test.test()

        end   = time.time()
        print("Finished training and evaluating for {} with hidden dims {} on {} of available unlabelled data after {} seconds".format(layer_type, hidden_dims, volume, end-start))
        print(results["lr_rmse"])
        print(results["gp_rmse"])

        title = "dims={},vol={}".format(hidden_dims, volume)
        plot_metrics_with_base_results(ax[i, j], ["gp_rmse", "lr_rmse"], results, base_results, title)


fig.savefig("test.jpg")
