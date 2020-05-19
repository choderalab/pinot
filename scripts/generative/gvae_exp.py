from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

import torch
from torch import optim
import time
import numpy as np
import scipy.sparse as sp
import dgl
import pinot
from pinot.generative.torch_gvae.model import GCNModelVAE
from pinot.generative.torch_gvae.loss import negative_ELBO
from pinot.generative.torch_gvae.utils import mask_test_edges, preprocess_graph, get_roc_score


def gae_for(args):
    # Grab some data from esol
    ds_tr = pinot.data.esol()[:10]

    # discard the measurement (we're doing unsupervised learning here)
    gs, _ = zip(*ds_tr)

    # Combine the molecular graphs into a large one
    g = dgl.batch(gs)

    # get the adjacency matrix for the giant graph
    # Because torch.SparseTensor doesn't interoperate too well with numpy
    # or scipy sparse matrix, we would need to work with scipy sparse matrix
    # and convert to torch.tensor where needed
    adj = sp.coo_matrix(g.adjacency_matrix().to_dense().numpy())
    features = torch.cat([g.ndata["type"], g.ndata["h0"]], dim=1)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) \
         / adj_train.sum()
    
    pos_weight = torch.FloatTensor([pos_weight])

    norm = adj_train.shape[0] * adj_train.shape[0] \
         / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)

    # Initialize the model and the optimization scheme
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model.encode_and_decode(features, adj_norm)

        # Compute the (sub-sampled) negative ELBO loss
        loss = negative_ELBO(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()

        # Compute the ROC score (with respect to link prediction task)
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)