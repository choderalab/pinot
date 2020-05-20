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

    