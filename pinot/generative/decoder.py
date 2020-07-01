import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import scipy
from torch.autograd import Variable


class DecoderNetwork(nn.Module):
    """Simple decoder where the node identity prediction is dependent
    on the adjacency matrix.

    Parameters
    ----------

    Returns
    -------

    Attributes
    ----------
    z_to_zx : torch.nn.Module
        neural networks that map general latent space encoding $z$
        to that for node prediction $z_x$
    z_to_za : torch.nn.Module
        neural networks that map general latent space encoding $z$
        to that for edge prediction $z_a$
    zx_to_x : torch.nn.Module
        neural networks that map latent encoding of node $z_x$ to node
        predictions $\hat{x}$
    """

    def __init__(self, 
        embedding_dim=128,
        num_atom_types=100,
        Dx1=64,
        Dx2=64,
        Da1=64,
        Da2=64,
        hidden_dim=64,
    ):
        super(DecoderNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_atom_types = num_atom_types

        self.Dx1 = Dx1
        self.Dx2 = Dx2
        self.z_to_zx = nn.Sequential(
            nn.Linear(embedding_dim, self.Dx1),
            nn.ReLU(),
            nn.Linear(self.Dx1, self.Dx2),
        )

        self.Da1 = Da1
        self.Da2 = Da2
        self.z_to_za = nn.Sequential(
            nn.Linear(embedding_dim, self.Da1),
            nn.ReLU(),
            nn.Linear(self.Da1, self.Da2),
        )

        self.hidden_dim = hidden_dim
        self.zx_to_x = nn.Sequential(
            nn.Linear(self.Dx2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_atom_types),
        )

    def forward(self, g, z_sample):
        """

        Parameters
        ----------
        g :

        z_sample :


        Returns
        -------

        """
        with g.local_scope():
            # Create a new graph with sampled representations
            g.ndata["h"] = z_sample
            # Unbatch into individual subgraphs
            gs_unbatched = dgl.unbatch(g)
            # Decode each subgraph
            decoded_subgraphs = [
                self.decode(g_sample.ndata["h"]) for g_sample in gs_unbatched
            ]
            return decoded_subgraphs

    def decode_and_compute_recon_error(self, g, z_sample):
        decoded_subgraphs = self.forward(g, z_sample)
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
        return loss

    def decode(self, z):
        """

        Parameters
        ----------
        z :
            a FloatTensor of size (number_of_nodes, embedding_dim)


        Returns
        -------
            (A_tilde, x_hat)
                A_tilde is a FloatTensor of size (number_of_nodes, number_of_nodes)
                x_hat is a FloatTensor of size (number_of_nodes, num_atom_types)
        """
        # (N, Dx)
        zx = self.z_to_zx(z)
        # (N, Da)
        za = self.z_to_za(z)
        # before rounding
        # (N, N)
        A_tilde = za @ za.T
        zx = A_tilde @ zx
        # predicted x
        # (N, n_classes)
        x_hat = self.zx_to_x(zx)
        return (A_tilde, x_hat)
