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
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x) # Decoder

    def parameterization(self, x, adj):
        """ Compute the parameters of the approximate Gaussian posterior
         distribution
        
        Args:
            x (FloatTensor): node features
                Shape (N, D) where N is the number of nodes in the graph
            adj (FloatTensor): adjacency matrix
                Shape (N, N)

        Returns:
            (mu, logvar) (FloatTensor, FloatTensor): parameters of approximate posterior
        """
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def inference(self, mu, logvar):
        """ Returns a sample from the approximate posterior distribution

        Args:
            mu (FloatTensor): mean parameter
                Shape (N, d_z) where N is the number of nodes from the input
                    graph
            logvar (FloatTensor): log variance
                Shape (N, d_z)

        Returns:
            If during training, returns a sample from N(mu, exp(logvar))
            If during testing, returns mu
        """
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
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
        mu, logvar = self.parameterization(x, adj)
        z = self.inference(mu, logvar)
        return self.dc(z), mu, logvar


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