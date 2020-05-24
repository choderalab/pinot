from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--split', type=list, default=[0.8, 0.2], help="train, test, validation split, default = [0.8, 0.2] (No validation)")
parser.add_argument('--batch_size', type=int, default=10, help="batch-size, i.e, how many molecules get 'merged' to form a graph per iteration during traing")
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

args = parser.parse_args()

import torch
from torch import optim
import time
import numpy as np
import scipy.sparse as sp
import dgl
import pinot
from pinot.data.utils import batch, split
from pinot.generative.torch_gvae.model import GCNModelVAE
from pinot.generative.torch_gvae.loss import negative_ELBO
from pinot.app.experiment import Train, Test, TrainAndTest, MultipleTrainAndTest

def run(args):
    # Grab some data from esol
    ds = pinot.data.esol()
    
    # Divide the molecules into train/test/val
    train_data, test_data, val_data = split(ds, [0.8, 0.1, 0.1])
    N_molecules = len(train_data)
    # "Batching" multiple molecules into groups, each groups
    # forming a "macro-molecule" (graph)
    batched_train_data = batch(train_data, args.batch_size)
    print("Training on ", N_molecules, "molecules")
    print("and batched into", len(batched_train_data), "batches")

    # Initialize the model and the optimization scheme
    feat_dim = train_data[0][0].ndata["h"].shape[1]
    model = GCNModelVAE(feat_dim, log_lik_scale=1./len(batched_train_data))
    optimizer = optim.Adam(model.parameters(), args.lr)

    # Setting up training and testing
    train_and_test = TrainAndTest(model, batched_train_data, test_data, optimizer,
                    [accuracy_edge_prediction, true_negative_edge_prediction, true_positive_edge_prediction],
                    n_epochs=1)

    results = train_and_test.run()

    print("Optimization Finished!")


################ METRICS ON EDGE PREDICTION ###################

def accuracy_edge_prediction(net, g, y):
    adj_mat = g.adjacency_matrix().to_dense()
    predicted_edges, _, _ = net.encode_and_decode(g)
    pos_pred = (predicted_edges > 0.5).int()
    return torch.mean((pos_pred == adj_mat).float())

def true_negative_edge_prediction(net, g, y):
    adj_mat = g.adjacency_matrix().to_dense()
    predicted_edges, _, _ = net.encode_and_decode(g)
    true_negatives = ((predicted_edges < 0.5) & (adj_mat==0)).int().sum()
    negatives = (adj_mat == 0).int().sum()
    return true_negatives.float()/negatives

def true_positive_edge_prediction(net, g, y):
    adj_mat = g.adjacency_matrix().to_dense()
    predicted_edges, _, _ = net.encode_and_decode(g)
    true_positives = ((predicted_edges > 0.5) & (adj_mat==1)).int().sum()
    positives = (adj_mat != 0).int().sum()
    return true_positives.float()/positives

if __name__ == '__main__':
    run(args)