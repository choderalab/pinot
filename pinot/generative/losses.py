import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import dgl

def negative_elbo(decoded_subgraphs, mu, logvar, g):
    """Compute the negative ELBO loss function used in variational auto-encoder
    The difference between this loss function and negative_ELBO is that this
    function also computes loss term from node's identity (atom type) prediction.

    Parameters
    ----------
    decoded_subgraphs : FloatTensor
        shape (N, N): a matrix where entry (i,j) in [0,1] denotes the predicted
        probability that there is an edge between atom i and j
    mu : FloatTensor
        shape (N, hidden_dim): The mean of the approximate posterior distribution over
        the nodes' (atoms) latent representation
    logvar : FloatTensor
        shape (N, hidden_dim): The log variance of the approximate posterior distribution
        over the nodes' latent representation
    g: the batched input graph

    Returns
    -------

        loss (Float)
        The negative ELBO

    """
    # First unbatch all the graphs into individual
    # subgraphs
    gs_unbatched = dgl.unbatch(g)

    assert len(decoded_subgraphs) == len(gs_unbatched)
    loss = 0.0
    for i, subgraph in enumerate(gs_unbatched):
        # Compute decoding loss for each individual sub-graphs
        decoded_edges, decoded_nodes = decoded_subgraphs[i]
        adj_mat = subgraph.adjacency_matrix(True).to_dense()
        if torch.cuda.is_available and decoded_edges.is_cuda:
            adj_mat = adj_mat.cuda()
        node_types = subgraph.ndata["type"].flatten().long()
        edge_nll = torch.sum(
            F.binary_cross_entropy_with_logits(decoded_edges, adj_mat)
        )
        node_nll = torch.sum(F.cross_entropy(decoded_nodes, node_types))

        loss += node_nll + edge_nll
    
    KLD = (
        -0.5
        / g.number_of_nodes()
        * torch.sum(
            torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1)
        )
    )

    return loss + KLD
