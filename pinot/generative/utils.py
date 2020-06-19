# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import pinot
import numpy as np
import scipy.sparse as sp

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def graph_from_adjacency_matrix(a):
    """ Utility function to convert adjacency matrix to graph while enforcing
    that all edges are bi-directionol.

    """
    # TODO: think about other strategies
    # make graph symmetrical
    a = torch.transpose(a, 0, 1) + a

    # query indices
    idxs = torch.gt(a, 1.0).nonzero().detach().numpy().tolist()

    # add indices one by one
    g = dgl.DGLGraph()
    g.add_nodes(a.shape[0])
    g.add_edges(*list(zip(*idxs)))

    return g


def prepare_semi_supervised_data(unlabeled_data, labeled_data):
    # Mix labelled and unlabelled data together
    semi_supervised_data = []

    for (g, y) in unlabeled_data:
        semi_supervised_data.append((g, None))
    for (g, y) in labeled_data:
        semi_supervised_data.append((g, y))

    return semi_supervised_data


def batch_semi_supervised(ds, batch_size, seed=2666):
    """ Batch graphs and values after shuffling.
    """
    # get the numebr of data
    n_data_points = len(ds)
    n_batches = n_data_points // batch_size  # drop the rest

    np.random.seed(seed)
    np.random.shuffle(ds)
    gs, ys = tuple(zip(*ds))

    gs_batched = [
        dgl.batch(gs[idx * batch_size : (idx + 1) * batch_size])
        for idx in range(n_batches)
    ]

    ys_batched = [
        list(ys[idx * batch_size : (idx + 1) * batch_size]) for idx in range(n_batches)
    ]

    return list(zip(gs_batched, ys_batched))


def prepare_semi_supervised_data_from_labeled_data(labeled_data, r=0.2, seed=2666):
    """
    r is the ratio of labelled data turning into unlabelled
    """
    semi_data = []
    small_labeled_data = []

    np.random.seed(seed)
    for (g, y) in labeled_data:
        if np.random.rand() < r:
            semi_data.append((g, y))
            small_labeled_data.append((g, y))
        else:
            semi_data.append((g, None))
    return semi_data, small_labeled_data
