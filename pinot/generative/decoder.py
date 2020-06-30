import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for edge prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, z):
        """Returns a symmetric adjacency matrix of size (N,N)
        where A[i,j] = probability there is an edge between nodes i and j

        Parameters
        ----------
        z :


        Returns
        -------

        """
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.mm(z, z.t())
        return adj


class EdgeAndNodeDecoder(nn.Module):
    """Decoder that returns both a predicted adjacency matrix
        and node identities

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, feature_dim, num_atom_types, hidden_dim=64, dropout=0):
        super(EdgeAndNodeDecoder, self).__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.decode_nodes = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_atom_types),
        )

    def forward(self, z):
        """Arg:
            z (FloatTensor):
                Shape (N, hidden_dim)

        Parameters
        ----------
        z :


        Returns
        -------

            (adj, node_preds)
            adj (FloatTensor): has shape (N, N) is a matrix where entry (i.j)
            stores the probability that there is an edge between atoms i,j
            node_preds (FloatTensor): has shape (N, num_atom_types) where each
            row i stores the probability of the identity of atom i

        """
        z_prime = F.dropout(z, self.dropout, training=self.training)
        adj = torch.mm(z_prime, z_prime.t())
        node_preds = self.decode_nodes(z_prime)
        return (adj, node_preds)


class SequentialDecoder(nn.Module):
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

    def __init__(
        self,
        embedding_dim,
        num_atom_types=100,
        Dx1=64,
        Dx2=64,
        Da1=64,
        Da2=64,
        hidden_dim=64,
    ):
        super(SequentialDecoder, self).__init__()
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

    def forward(self, z):
        """

        Parameters
        ----------
        z :


        Returns
        -------

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


class DecoderNetwork(nn.Module):
    """ """

    def __init__(self, embedding_dim, num_atom_types):
        super(DecoderNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.decoder = SequentialDecoder(embedding_dim, num_atom_types)

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
                self.decoder(g_sample.ndata["h"]) for g_sample in gs_unbatched
            ]
            return decoded_subgraphs
