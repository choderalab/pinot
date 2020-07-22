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

    def __init__(
        self,
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
            nn.ReLU(),
        )

        self.Da1 = Da1
        self.Da2 = Da2
        self.z_to_za = nn.Sequential(
            nn.Linear(embedding_dim, self.Da1),
            nn.ReLU(),
            nn.Linear(self.Da1, self.Da2),
            nn.ReLU(),
        )

        self.hidden_dim = hidden_dim
        self.zx_to_x = nn.Sequential(
            nn.Linear(self.Dx2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_atom_types),
            nn.ReLU(),
        )

    def forward(self, g, z_sample):
        """ Do the encoding

        Parameters
        ----------
        g : DGLGraph
            Input batched graph

        z_sample : FloatTensor
            Of shape (num_nodes, embedding_dim)

        Returns
        -------
        decoded_subgraphs: list[(A_tilde, x_tilde)]
            A_tilde: FloatTensor
                of shape (num_nodes, num_nodes) is the reconstructed
                adjancency matrix score

            x_tilde: FloatTensor
                of shape (num_nodes, num_atom_types) is the predicted
                node type
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
        """ Decode and compute reconstruction error

        """
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
            # Note that F.cross_entropy combines log_softmax
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


class EdgeDecoder(nn.Module):
    """Decoder where the node identity prediction is dependent
    on the adjacency matrix. And the decoder reconstructs the 
    edge tensor

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
    e_tensor_to_E_tilde: torch.nn.Module
        neural networks that map from latent encoding of node $z_x$ to
        edge tensor prediction $E_tilde$
    """

    def __init__(
        self,
        embedding_dim=128,
        num_atom_types=100,
        num_bond_types=22,
        Dx1=64,
        Dx2=64,
        Da1=64,
        Da2=64,
        hidden_dim=64,
        hidden_dim_e=64,
    ):
        super(EdgeDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types

        self.Dx1 = Dx1
        self.Dx2 = Dx2
        self.z_to_zx = nn.Sequential(
            nn.Linear(embedding_dim, self.Dx1),
            nn.ReLU(),
            nn.Linear(self.Dx1, self.Dx2),
            nn.ReLU(),
        )

        self.Da1 = Da1
        self.Da2 = Da2
        self.z_to_za = nn.Sequential(
            nn.Linear(embedding_dim, self.Da1),
            nn.ReLU(),
            nn.Linear(self.Da1, self.Da2),
            nn.ReLU(),
        )

        self.hidden_dim = hidden_dim
        self.zx_to_x = nn.Sequential(
            nn.Linear(self.Dx2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_atom_types),
            nn.ReLU(),
        )

        self.hidden_dim_e = hidden_dim_e
        self.e_tensor_to_E_tilde = nn.Sequential(
            nn.Linear(self.Dx2 * 2, self.hidden_dim_e),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_e, self.num_bond_types),
            nn.ReLU(),
        )

    def edge_tensor_from_g(self, g):
        """ Construct a E tensor of shape (num_nodes, num_nodes, num_bond_types)
        from g.

        Parameters:
        -----------
            g: the batched input graph

        Returns:
        --------
            E: FloatTensor
                Of shape (num_nodes, num_nodes, num_bond_types)
                Where E[i,j,:] is a one hot encoding of the bond type between
                nodes (atoms) i and j. There are 22 types of bonds and no bond
                (no edge) is of type 21 (0-based indexing)

        """
        n = g.number_of_nodes()

        one_hot_no_bond = torch.zeros(self.num_bond_types)
        one_hot_no_bond[-1] = 1

        # E tensor will have shape (n, n, num_bond_types)
        E = one_hot_no_bond.repeat(n, n, 1)

        # Get the indices of the edges
        indices = g.adjacency_matrix().coalesce().indices()
        # Get the bond types
        etypes = g.edata["type"]

        for e_idx in range(indices.shape[1]):
            e = torch.tensor([etypes[e_idx]])

            # Get the corresponding entry of the E_tensor
            one_hot = torch.cat((indices[:, e_idx], e.long()))

            # Flip the bit associated with the bond type
            E[list(one_hot)] = 1.0
            E[list(torch.cat((indices[:, e_idx], torch.tensor([-1]))))]
        return E

    def decode_and_compute_recon_error(self, g, z_sample):
        """ Decode and compute the reconstruction error 

        Parameters:
        -----------
            g: DGLGraph
                the input (batched) graph

            z_sample: FloatTensor
                the z variable that has previously been 
                sampled in the variational auto encoder

        Returns:
        --------
            loss: Float
                The reconstruction loss, it corresponds to the
                    negative expected likelihood term in the ELBO
        """
        decoded_subgraphs = self.forward(g, z_sample)
        gs_unbatched = dgl.unbatch(g)
        assert len(decoded_subgraphs) == len(gs_unbatched)

        if z_sample.is_cuda:
            loss = torch.tensor([0.0]).cuda()
        else:
            loss = torch.tensor([0.0])

        for i, subgraph in enumerate(gs_unbatched):
            # Compute decoding loss for each individual sub-graphs

            # First get the reconstructed E tensor and x matrix
            E_tilde, x_tilde = decoded_subgraphs[i]

            # get E_true
            E_true = self.edge_tensor_from_g(subgraph)
            if z_sample.is_cuda:
                edge_nll = torch.sum(
                    F.binary_cross_entropy_with_logits(
                        E_tilde.cuda(), E_true.cuda()
                    )
                )
            else:
                edge_nll = torch.sum(
                    F.binary_cross_entropy_with_logits(E_tilde, E_true)
                )

            node_types = subgraph.ndata["type"].flatten().long()
            # Note that F.cross_entropy combines log_softmax so we don't
            # have to do sigmoid of x_tilde before calling this function
            node_nll = torch.sum(F.cross_entropy(x_tilde, node_types))

            loss += node_nll + edge_nll
        return loss

    def decode(self, z):
        """ Decode a specific z_sample from a subgraph
        Parameters
        ----------
            z: FloatTensor
                Of shape (number of nodes, embedding_dim)
                The hidden node variable
        Returns
        -------
            (E_tilde, x_tilde)
                E_tilde: FloatTensor
                    Of shape (number of nodes, number of nodes, number of bond types)
                    Is the reconstructed E tensor where each E_tilde(i, j, :) is the scores
                    for the bond types between nodes (atoms) i and j. To be used with
                    `binary_cross_entropy_with_logits(E_true, E_tilde)`

                x_tilde: FloatTensor
                    Of shape (number of nodes, number of atom types)
                    Is the reconstructed node type scores

        """
        # z -> za
        za = self.z_to_za(z)
        # za -> Atilde
        A_tilde = za @ za.T
        # z -> zx_temp
        zx = self.z_to_zx(z)

        # Atilde, zx_temp -> zx
        # This sequence of computation corresponds to 1 layer
        # of GCN
        zx = A_tilde @ zx
        zx = torch.relu(zx)

        # zx -> x_tilde
        x_tilde = self.zx_to_x(zx)

        (n, h) = zx.shape
        assert h == self.Dx2
        # zx, Atilde -> E_tilde
        temp1 = zx.repeat(1, n).view(n * n, h)  # Shape should be (n, n, Dx2)
        temp2 = z.repeat(n, 1)  # Shape is also (n, n, Dx2)
        temp = torch.cat(
            (temp1, temp2), 1
        )  # This creates a (n, n, 2 * Dx2) tensor
        # where temp[i, j, :] is the concatenation of zx[i,:] and zx[j, :]

        # This has shape (n, n, 2*self.Dx2)
        e_tensor = temp.view(n, n, 2 * h)

        # e_tensor -> E_tilde
        E_tilde = self.e_tensor_to_E_tilde(e_tensor.view(n * n, 2 * h))
        E_tilde = E_tilde.reshape(
            n, n, self.num_bond_types
        )  # Shape should be (n, n, num_bond_types)

        return E_tilde, x_tilde

    def forward(self, g, z_sample):
        """ Do the decoding on all of the subgraphs

        Parameters
        ----------
        g : DGLGraph
            Input (batched) graph

        z_sample : FloatTensor
            Of shape (num_nodes, embedding_dim)
            Hidden node variables

        Returns
        -------
            decoded_subgraphs: list[(E_tilde, x_tilde)]
                See self.decode
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
