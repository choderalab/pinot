import torch
import torch.nn as nn
import torch.nn.functional as F
from pinot.representation.dgl_legacy import GN
from pinot.generative.torch_gvae.loss import negative_ELBO

class GCNModelVAE(nn.Module):
    """Graph convolutional neural networks for VAE
    """
    def __init__(self, input_feat_dim, hidden_dim1=32, \
            hidden_dim2=32, hidden_dim3=16, dropout=0.1, \
            log_lik_scale=1):
        """ Construct a VAE with GCN
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
        self.dc = InnerProductDecoder(dropout)
        # Relative weight between the KL divergence and the log likelihood term
        self.log_lik_scale = log_lik_scale

    def forward(self, g):
        """ Compute the parameters of the approximate Gaussian posterior
         distribution
        
        Args:
            x (FloatTensor): node features
                Shape (N, D) where N is the number of nodes in the graph
            adj (FloatTensor): adjacency matrix
                Shape (N, N)

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
        """
        z = self.forward(g)
        theta = [parameter.forward(g, z) for parameter in self.output_regression]
        mu  = theta[0]
        logvar = theta[1]
        distribution = torch.distributions.normal.Normal(
                    loc=theta[0],
                    scale=torch.exp(theta[1]))

        return distribution, theta[0], theta[1]

    # TODO: separate between training and testing
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
        approx_posterior, mu, logvar = self.condition(g)
        z_sample = approx_posterior.rsample()
        return self.dc(z_sample), mu, logvar


    def loss(self, g, y=None):
        """ Compute negative ELBO loss
        """
        predicted_edges, mu, logvar = self.encode_and_decode(g)
        adj_mat = g.adjacency_matrix(True).to_dense()
        # Compute the (sub-sampled) negative ELBO loss
        loss = negative_ELBO(preds=predicted_edges,
                            labels=adj_mat,
                            mu=mu, logvar=logvar,
                            norm=self.log_lik_scale)
        return loss


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