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
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    D_tilde = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(D_tilde, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
