from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--split',nargs='+', type=float, default=[0.9, 0.1, 0.], help="train, test, validation split, default = [0.8, 0.2] (No validation)")
parser.add_argument('--batch_size', type=int, default=10, help="batch-size, i.e, how many molecules get 'merged' to form a graph per iteration during traing")
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128], help="hidden dimension 1")
parser.add_argument('--embedding_dim', type=int, default=64, help="node embedding dimension")
parser.add_argument('--html', type=str, default="results.html", help="File to save results to")

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
from pinot.app.report import html

def run(args):
    # Grab some data from esol
    ds = pinot.data.esol()
    
    # Divide the molecules into train/test/val
    train_data, test_data, _ = split(ds, args.split)
    N_molecules = len(train_data)
    # "Batching" multiple molecules into groups, each groups
    # forming a "macro-molecule" (graph)
    batched_train_data = batch(train_data, args.batch_size)
    print("Training on ", N_molecules, "molecules")
    print("and batched into", len(batched_train_data), "batches")

    # Initialize the model and the optimization scheme
    feat_dim = train_data[0][0].ndata["h"].shape[1]
    num_atom_types = 100
    model = GCNModelVAE(feat_dim, gcn_hidden_dims=args.hidden_dims,
        embedding_dim=64, num_atom_types=num_atom_types)
    
    optimizer = optim.Adam(model.parameters(), args.lr)

    # Setting up training and testing
    train_and_test = TrainAndTest(model, batched_train_data, test_data, optimizer,
                    [accuracy_edge_prediction, true_negative_edge_prediction,\
                        true_positive_edge_prediction, accuracy_node_prediction,\
                        negative_elbo_loss],
                    n_epochs=args.epochs)

    results = train_and_test.run()

    print("Optimization Finished! Now printing results to", args.html)
    html_string = html(results)
    
    f_handle = open(args.html, 'w')
    f_handle.write(html_string)
    f_handle.close()

################ METRICS ON EDGE PREDICTION ###################

def accuracy_edge_prediction(net, g, y):
    unbatched_subgraphs = dgl.unbatch(g)
    decoded_subgraphs, _, _ = net.encode_and_decode(g)
    assert(len(decoded_subgraphs) == len(unbatched_subgraphs))

    avg_acc = 0.
    for i, subg in enumerate(unbatched_subgraphs):
        edge_pred, _ = decoded_subgraphs[i]
        adj_mat = subg.adjacency_matrix(True).to_dense()
        acc = torch.mean((( (edge_pred > 0.5) & (adj_mat == 1))\
             | ((edge_pred < 0.5) & (adj_mat == 0) )).float())
        avg_acc += acc / len(unbatched_subgraphs)
    return avg_acc

def true_negative_edge_prediction(net, g, y):
    unbatched_subgraphs = dgl.unbatch(g)
    decoded_subgraphs, _, _ = net.encode_and_decode(g)
    assert(len(decoded_subgraphs) == len(unbatched_subgraphs))

    avg_tn = 0.
    for i, subg in enumerate(unbatched_subgraphs):
        edge_pred, _ = decoded_subgraphs[i]
        adj_mat = subg.adjacency_matrix(True).to_dense()
        true_negatives = ((edge_pred < 0.5) & (adj_mat==0)).int().sum()
        all_negatives = (adj_mat == 0).int().sum()
        tn =  true_negatives.float()/all_negatives
        avg_tn += tn / len(unbatched_subgraphs)

    return avg_tn


def true_positive_edge_prediction(net, g, y):
    unbatched_subgraphs = dgl.unbatch(g)
    decoded_subgraphs, _, _ = net.encode_and_decode(g)
    assert(len(decoded_subgraphs) == len(unbatched_subgraphs))

    avg_tp = 0.
    for i, subg in enumerate(unbatched_subgraphs):
        edge_pred, _ = decoded_subgraphs[i]
        adj_mat = subg.adjacency_matrix(True).to_dense()
        true_positives = ((edge_pred > 0.5) & (adj_mat==1)).int().sum()
        all_positives = (adj_mat == 1).int().sum()
        tp =  true_positives.float()/all_positives
        avg_tp += tp / len(unbatched_subgraphs)

    return avg_tp

def accuracy_node_prediction(net, g, y):
    unbatched_subgraphs = dgl.unbatch(g)
    decoded_subgraphs, _, _ = net.encode_and_decode(g)
    assert(len(decoded_subgraphs) == len(unbatched_subgraphs))

    avg_acc = 0.
    for i, subg in enumerate(unbatched_subgraphs):
        _, node_preds = decoded_subgraphs[i]
        node_types = subg.ndata["type"]
        node_type_preds = torch.argmax(node_preds, 1)
        acc = torch.mean((node_type_preds == node_types).float())
        avg_acc += acc /len(unbatched_subgraphs)

    return torch.mean((node_type_preds == node_types).float())

def negative_elbo_loss(net, g, y):
    return net.loss(g).detach()


if __name__ == '__main__':
    run(args)