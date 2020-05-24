from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--split', type=list, default=[0.8, 0.2], help="train, test, validation split, default = [0.8, 0.2] (No validation)")
parser.add_argument('--batch_size', type=int, default=1, help="batch-size, i.e, how many molecules get 'merged' to form a graph per iteration during traing")
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

def train(args):
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
    model = GCNModelVAE(feat_dim)
    optimizer = optim.Adam(model.parameters(), args.lr)

    for epoch in range(args.epochs):
        t = time.time()

        total_epoch_tr_loss = 0.
        average_accuracy = 0.

        for g_data in batched_train_data:
            ################ DATA PREPARATION #########################
            g = g_data[0]
            adj_mat = g.adjacency_matrix(True).to_dense()

            ################ Optimization ######################
            model.train()
            optimizer.zero_grad()
            n_nodes, num_features = g.ndata["h"].shape
            assert(num_features == feat_dim)
            # Compute the (sub-sampled) negative ELBO loss
            loss = model.loss(g, log_lik_scale=1./len(batched_train_data))
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            total_epoch_tr_loss += loss.detach()

            predicted_edges, _, _ = model.encode_and_decode(g)
            average_accuracy += 1/len(batched_train_data) *\
                 accuracy_edge_prediction(predicted_edges, adj_mat)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", \
            "{:.5f}".format(total_epoch_tr_loss),\
            "avg edge reconstruction accuracy = {:.5f}".format(average_accuracy)
        )

    print("Optimization Finished!")


def accuracy_edge_prediction(predicted_edges, adj_mat):
    pos_pred = (predicted_edges > 0.5).int()
    return torch.mean((pos_pred == adj_mat).float())


if __name__ == '__main__':
    train(args)