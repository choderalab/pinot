from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')

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
from pinot.generative.torch_gvae.utils import get_roc_score, negative_shifted_laplacian, \
    prepare_train_test_val_data_from_adj_matrix, prepare_train_test_val

def gae_for(args):
    # Grab some data from esol
    ds_tr = pinot.data.esol()
    # print(ds_tr)
    graph_data = zip(*prepare_train_test_val(ds_tr))

    # # Initialize the model and the optimization scheme
    feat_dim = ds_tr[0][0].ndata["h"].shape[1]
    model = GCNModelVAE(feat_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    N_molecules = len(ds_tr)

    print("Processing ", N_molecules, "molecules")
    for epoch in range(args.epochs):
        t = time.time()

        total_epoch_tr_loss = 0.
        epoch_val_roc_score = 0.
        epoch_val_ap_score = 0.
        epoch_test_roc_score = 0.
        epoch_test_ap_score = 0.

        for molecule in graph_data:
            ################ DATA PREPARATION #########################
            # get the adjacency matrix for the molecular graph
            # Because torch.SparseTensor doesn't interoperate too well with numpy
            # or scipy sparse matrix, we would need to work with scipy sparse matrix
            # and convert to torch.tensor where needed
            (mol_graph,
                adj_orig,
                adj_norm,
                adj_label,
                adj_train,
                train_edges,
                val_edges,
                val_edges_false,
                test_edges,
                test_edges_false,
                node_features,
                norm,
                pos_weight,
                measurement) = molecule

            ################ Optimization ######################

            model.train()
            optimizer.zero_grad()
            n_nodes, num_features = node_features.shape
            assert(num_features == feat_dim)

            recovered, mu, logvar = model.encode_and_decode(mol_graph)

            # Compute the (sub-sampled) negative ELBO loss
            loss = negative_ELBO(preds=recovered, labels=adj_label,
                                mu=mu, logvar=logvar, n_nodes=n_nodes,
                                norm=norm, pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            hidden_emb = mu.data.numpy() # embedding/latent representation of the nodes

            # Compute the ROC score (with respect to link prediction task)
            roc_val, ap_val = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
            roc_test, ap_test = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)

            # Update average training/validation ROC/ validation AP score in this epoch
            total_epoch_tr_loss += 1./N_molecules * loss
            epoch_val_ap_score += 1./N_molecules * ap_val
            epoch_val_roc_score += 1./N_molecules* roc_val
            epoch_test_roc_score += 1./N_molecules * roc_test
            epoch_test_ap_score += 1./N_molecules * ap_test


        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(total_epoch_tr_loss),
            "val_ap=", "{:.5f}".format(epoch_val_ap_score),
            "val_roc={:.5f}".format(epoch_val_roc_score),
            "test_ap=", "{:.5f}".format(epoch_test_ap_score),
            "test_roc={:.5f}".format(epoch_test_roc_score),
            "time=", "{:.5f}".format(time.time() - t), 
        )

    print("Optimization Finished!")


if __name__ == '__main__':
    gae_for(args)