import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    """ Compute ROC score (on link prediction)
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def negative_shifted_laplacian(adj):
    """ Constructs the "negative shifted Laplacian" matrix from the adjacency matrix
    D*^{-1/2}A*D*^{-1/2} in equation (2) of Kipf and Welling
    
    where 
          A* = A + I # adding self loops
          D* = sum(A*, axis=1) # row sum

    This matrix is inspired by first order approximation of localized spectral filters
    on graphs.

    To see more details, see Kipf and Welling (2016)
    Note that I don't have a good name for this matrix and not sure if people have
    come up with an "official" name for this matrix.

    But  I - D^{-1/2}AD^{-1/2} is the normalized Laplacian matrix
    And I + D^{-1/2}AD^{-1/2} = D*^{-1/2}A*D*^{-1/2}

    So one can think of D*^{-1/2}A*D*^{-1/2} as the shifted negative version
    of the normalized Laplacian

    Arg:
        adj: adjacency matrix (may not be in sparse layout)

    Returns:


    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    D_tilde = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(D_tilde, -0.5).flatten())
    neg_shifted_lap = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(neg_shifted_lap)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def prepare_train_test_val(data):
    """ Prepare training, testing and validation data for GVAE
    """
    adj_orig_arr = []
    adj_norm_arr = []
    adj_label_arr = []
    adj_train_arr = []
    train_edges_arr = []
    val_edges_arr = []
    val_edges_false_arr = []
    test_edges_arr = []
    test_edges_false_arr = []
    node_features_arr = []
    norm_arr = []
    pos_weight_arr = []
    measurement_arr = []

    for (g, measurement) in data:
        adj = sp.coo_matrix(g.adjacency_matrix().to_dense().numpy())
        features = torch.cat([g.ndata["type"], g.ndata["h0"]], dim=1)

        adj_orig, adj_norm, adj_label, adj_train, \
            train_edges, val_edges, val_edges_false, \
            test_edges, test_edges_false = \
                prepare_train_test_val_data_from_adj_matrix(adj)

        pos_weight = torch.FloatTensor([(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) \
            / adj_train.sum()])
    
        norm = adj_train.shape[0] * adj_train.shape[0] \
            / float((adj_train.shape[0] * adj_train.shape[0] \
                - adj_train.sum()) * 2)

        adj_orig_arr.append(adj_orig)
        adj_norm_arr.append(adj_norm)
        adj_label_arr.append(adj_label)
        adj_train_arr.append(adj_train)
        train_edges_arr.append(train_edges)
        val_edges_arr.append(val_edges)
        val_edges_false_arr.append(val_edges_false)
        test_edges_arr.append(test_edges)
        test_edges_false_arr.append(test_edges_false)
        node_features_arr.append(features)
        norm_arr.append(norm)
        pos_weight_arr.append(pos_weight)
        measurement_arr.append(measurement)

        return  (adj_orig_arr,
                adj_norm_arr,
                adj_label_arr,
                adj_train_arr,
                train_edges_arr,
                val_edges_arr,
                val_edges_false_arr,
                test_edges_arr,
                test_edges_false_arr,
                node_features_arr,
                norm_arr,
                pos_weight_arr,
                measurement_arr)
        


def prepare_train_test_val_data_from_adj_matrix(adj):
    """
    """
    # Store original adjacency matrix (without diagonal entries) for later use
    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, \
        val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    
    # Some preprocessing
    adj_norm = negative_shifted_laplacian(adj_train)
    adj_label = adj_train + sp.eye(adj_train.shape[0])

    adj_label = torch.FloatTensor(adj_label.toarray())

    return adj_orig, adj_norm, adj_label, adj_train, \
            train_edges, val_edges, val_edges_false, \
            test_edges, test_edges_false


def sparse_to_tuple(sparse_mx):
    """ Simply convert a sparse matrix into a 3-element tuple of
    (an array of 2d coordinates, an array of values, shape)

    Arg:
        sparse_mx: a sparse matrix (in COO layout by default)
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    """ "Split" the adjacency matrix by masking edges 
    to be used in link prediction

    Arg:
        adj: adjacency matrix of the graph

    Returns: A 6-tuple of sparse matrices
        adj_train: graph for training
        train_edges: positive edges for training
        val_edges: validation edges
        val_edges_false: negative validation edges
        test_edges: test edges
        test_edges_false: negative test edges

    """
    # Function to build test set with 10% positive links

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag sum is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj) # Get the upper triangular portion of adj
    adj_tuple = sparse_to_tuple(adj_triu) # Get (coordinates,values) tuple
    edges = adj_tuple[0] # Get the coordinates (correseponding to edges)

    # Get ALL edges (simply double count the reverse direction) that of edges
    edges_all = sparse_to_tuple(adj)[0]

    # Number of test edges, if too small, this might be 0
    num_test = max(int(np.floor(edges.shape[0] / 10.)), 2)

    # Number of validation edges
    num_val = max(int(np.floor(edges.shape[0] / 20.)), 2)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
 