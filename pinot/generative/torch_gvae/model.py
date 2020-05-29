import torch
import torch.nn as nn
import torch.nn.functional as F
from pinot.representation.dgl_legacy import GN
from pinot.generative.torch_gvae.loss import negative_ELBO,\
    negative_ELBO_with_node_prediction

class GCNModelVAE(nn.Module):
    """Graph convolutional neural networks for VAE
    """
    def __init__(self, input_feat_dim, hidden_dim1=32, \
            hidden_dim2=32, hidden_dim3=16, dropout=0.0, \
            num_atom_types=100):
        """ Construct a VAE with GCN
        Args:
            input_feature_dim: Number of input features for each atom/node

            hidden_dim1: First hidden dimension, after applying one
                linear layer

            hidden_dim2: Second hidden dimension, after applying the first
                graph convolution layer

            hidden_dim_3: Third hidden dimension, after applying the second
                graph convolution layer

            num_atom_types: The number of possible atom types

        """
        super(GCNModelVAE, self).__init__()
        self.linear = nn.Linear(input_feat_dim, hidden_dim1)
        self.gc1 = GN(hidden_dim1, hidden_dim2)

        # Mapping from "latent graph" to predictive distribution parameter
        self.output_regression = nn.ModuleList([
            GN(hidden_dim2, hidden_dim3),
            GN(hidden_dim2, hidden_dim3),
        ])
        # Decoder
        self.dc = EdgeAndNodeDecoder(dropout, hidden_dim3, num_atom_types)
        self.num_atom_types = num_atom_types

    def forward(self, g):
        """ Compute the parameters of the approximate Gaussian posterior
         distribution
        
        Args:
            g (DGLGraph)
                The molecular graph
        Returns:
            z: (FloatTensor): the latent encodings of the nodes
                Shape (N, hidden_dim2)
        """
        z1 = self.linear(g.ndata["h"])
        z = self.gc1(g, z1)
        return z

    def condition(self, g):
        """ Compute the approximate Normal posterior distribution q(z|x, adj)
        and associated parameters

        Arg:
            g (DGLGraph)
                The molecular graph
        
        Returns:
            distribution, mu, logvar
                distribution (torch.Distribution) 
                    is the approximate normal distribution

                mu (FloatTensor) 
                    shape (N, hidden_dim_3) 
                    is the mean of the approximate posterior normal distribution

                logvar (FloatTensor)
                    shape (N, hidden_dim_3)
                    is the log variance of the approximate posterior normal distribution

        """
        z = self.forward(g)
        theta = [parameter.forward(g, z) for parameter in self.output_regression]
        mu  = theta[0]
        logvar = theta[1]
        distribution = torch.distributions.normal.Normal(
                    loc=theta[0],
                    scale=torch.exp(theta[1]))

        return distribution, theta[0], theta[1]

    def encode_and_decode(self, g):
        """ Forward pass through the GVAE

        Args:
            x (FloatTensor): node features
                Shape (N, D) where N is the number of nodes in the graph
            adj (FloatTensor): adjacency matrix
                Shape (N, N)

        Returns:
            adj*, mu, logvar
                adj* is the reconstructed adjacency matrix
                mu, logvar are the parameters of the approximate Gaussian
                    posterior
        """
        # Encode
        approx_posterior, mu, logvar = self.condition(g)
        # Decode
        z_sample = approx_posterior.rsample()
        return self.dc(z_sample), mu, logvar


    def loss(self, g, y=None):
        """ Compute negative ELBO loss
        """
        predicted, mu, logvar = self.encode_and_decode(g)
        (edge_preds, node_preds) = predicted

        adj_mat = g.adjacency_matrix(True).to_dense()
        node_types = g.ndata["type"].flatten().long()
        node_types_one_hot =\
            F.one_hot(node_types.flatten().long(), self.num_atom_types).float()
        loss = negative_ELBO_with_node_prediction(edge_preds, node_preds,
            adj_mat, node_types, mu, logvar) # Check one-hot
        return loss


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for edge prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        """ Returns a symmetric adjacency matrix of size (N,N)
        where A[i,j] = probability there is an edge between nodes i and j
        """
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class EdgeAndNodeDecoder(nn.Module):
    """ Decoder that returns both a predicted adjacency matrix
        and node identities
    """
    def __init__(self, dropout, feature_dim, num_atom_types, act=torch.sigmoid):
        super(EdgeAndNodeDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.linear = nn.Linear(feature_dim, num_atom_types)

    def forward(self, z):
        """
        Arg:
            z (FloatTensor):
                Shape (N, hidden_dim)
        Returns:
            (adj, node_preds)
                adj (FloatTensor): has shape (N, N) is a matrix where entry (i.j)
                    stores the probability that there is an edge between atoms i,j
                node_preds (FloatTensor): has shape (N, num_atom_types) where each
                    row i stores the probability of the identity of atom i
        """
        z_prime = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z_prime, z_prime.t()))
        node_preds = self.linear(z_prime) # Check softmax
        return (adj, node_preds)