import torch
import torch.nn as nn
import torch.nn.functional as F

from pinot.generative.torch_gvae.layers import GraphConvolution


class GCNModelVAE(nn.Module):
    """Graph convolutional neural networks for VAE
    """
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        """ Construct a VAE with GCN
        """
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x) # Decoder
        self.output_regression = torch.nn.ModuleList([
            GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x),
            GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x),
        ])

    def forward(self, x, adj):
        """ Compute the parameters of the approximate Gaussian posterior
         distribution
        
        Args:
            x (FloatTensor): node features
                Shape (N, D) where N is the number of nodes in the graph
            adj (FloatTensor): adjacency matrix
                Shape (N, N)

        Returns:
            z: (FloatTensor): latent encodings of the nodes
                Shape (N, D)
        """
        z = self.gc1(x, adj)
        return z

    def condition(self, x, adj):
        """ Compute the approximate Normal posterior distribution q(z|x, adj)
        and associated parameters
        """
        z = self.forward(x, adj)
        theta = [parameter(z, adj) for parameter in self.output_regression]
        mu  = theta[0]
        logvar = theta[1]
        distribution = torch.distributions.normal.Normal(
                    loc=theta[0],
                    scale=torch.exp(theta[1]))

        return distribution, theta[0], theta[1]

    def encode_and_decode(self, x, adj):
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
        approx_posterior, mu, logvar = self.condition(x, adj)
        z_sample = approx_posterior.rsample()
        return self.dc(z_sample), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj