""" Metrics and loss function for semi-supervised learning.

"""

# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import torch
import math
import dgl

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def get_global_attr_in_node_level(g, pooling_fn=lambda g: dgl.sum_nodes(g, 'h')):
    """ Compute the global attribute and put it in node level.
    """
    # get the global attr
    h_u = pooling_fn(g)

    # get the list of number of nodes in the subgraphs
    batch_num_nodes = g.batch_num_nodes

    # broadcast `h_u` to node level
    h_u = torch.cat(
        [
            h_u[idx][None, :].repeat(batch_num_nodes[idx], 1)\
                for idx in range(g.batch_size)
        ],
        dim=0)

    g.ndata['h_u'] = h_u

    return g

def score(g, k=10, pooling_fn=lambda g: dgl.sum_nodes(g, 'h')):
    """ Calculate the score:
    $$
    P(h_v | h_u)
    $$
    """
    g = get_global_attr_in_node_level(g, pooling_fn=pooling_fn)

    # grab the late codes
    h_v = g.ndata['h']
    h_u = g.ndata['h_u']

    # calculate
    # $$
    # h_v ^ T h_u
    # $$
    score_same = torch.nn.functional.cosine_similarity(h_v, h_u)
    score_diff = torch.nn.functional.cosine_similarity(h_v, h_u[torch.randperm(h_v.shape[0])])
    
    return score_same - k * score_diff

