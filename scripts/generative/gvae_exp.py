from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()

import torch
from torch import optim

import pinot
import dgl
from pinot.generative.torch_gvae.model import GCNModelVAE
from pinot.generative.torch_gvae.optimizer import loss_function
from pinot.generative.torch_gvae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score



def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))

    # Grab some data from esol
    ds_tr = pinot.data.esol()[:10]

    # discard the measurement (we're doing unsupervised learning here)
    gs, _ = zip(*ds_tr)

    # Combine the molecular graphs into a large one
    g = dgl.batch(gs)

    # get the adjacency matrix for the giant graph
    adj = g.adjacency_matrix()

    features = torch.cat((g.ndata["type"], g.ndata["h0"]), dim=1) # horizontal concat
    print(features.shape)
    n_nodes, feat_dim = features.shape

    adj_orig = adj
    diag = adj_orig.to_dense().diagonal()

    # Subtract the diagonal?
    adj_orig = adj_orig - torch.diag(diag).to_sparse()

    # Some preprocessing
    adj_label = torch.eye(adj.shape[0]) + adj
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label)

    pos_weight = float(adj.shape[0] * adj.shape[0] -\
         torch.sparse.sum(adj)) / torch.sparse.sum(adj)
    
    norm = adj.shape[0] * adj.shape[0] \
        / float((adj.shape[0] * adj.shape[0] - torch.sparse.sum(adj)) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_orig)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm,
                             pos_weight=torch.from_numpy(np.array([pos_weight])))
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")


if __name__ == '__main__':
    gae_for(args)