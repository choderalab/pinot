import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from pinot.representation.dgl_legacy import GN
from pinot.generative.torch_gvae.loss import negative_ELBO,\
    negative_ELBO_with_node_prediction

class GCNModelVAE(nn.Module):
    """Graph convolutional neural networks for VAE
    """
    def __init__(self, input_feat_dim, gcn_type="GraphConv", gcn_init_args ={},\
            gcn_hidden_dims=[256, 128], embedding_dim=64, dropout=0.0, \
            num_atom_types=100, \
            aggregation_function=lambda g: dgl.sum_nodes(g, "h")):
        """ Construct a VAE with GCN
        Args:
            input_feature_dim: Number of input features for each atom/node

            hidden_dims: list
                The hidden dimensions of the graph convolution layers
            
            embedding_dim: int
                The dimension of the latent representation of the nodes

            num_atom_types: The number of possible atom types

            aggregation_function: function used to aggregate the node feature
                vectors into a single graph representation

        """
        super(GCNModelVAE, self).__init__()

        # Decoder
        self.dc = EdgeAndNodeDecoder(dropout, embedding_dim, num_atom_types)
        self.num_atom_types = num_atom_types

        # Encoder

        # 1. Graph convolution layers
        # Note that we define this here because in 'Net', the model automatically
        # finds the input dimension of the output-regressor by inspecting the
        # last torch module of the representation layer
        assert(len(gcn_hidden_dims) > 0)
        self.gcn_modules = []
        for (dim_prev, dim_post) in zip([input_feat_dim] + gcn_hidden_dims[:-1],\
                gcn_hidden_dims):
            self.gcn_modules.append(GN(dim_prev, dim_post, gcn_type, gcn_init_args))
        self.gc = nn.ModuleList(self.gcn_modules)
        self.aggregator = aggregation_function
        self.embedding_dim = embedding_dim

        # 2. Mapping from node embedding to predictive distribution parameter
        self.output_regression = nn.ModuleList([
            nn.Linear(gcn_hidden_dims[-1], embedding_dim),
            nn.Linear(gcn_hidden_dims[-1], embedding_dim),
        ])

    def forward(self, g):
        """ Compute the latent representation of the input graph. This
        function is slightly different from `infer_node_representation`
        only in that it aggregates the node features into one vector.
        This is so that `GVAE` can be directly used as part of `Net`.
        
        Args:
            g (DGLGraph)
                The molecular graph
        Returns:
            z: (FloatTensor): the latent encodings of the graph
                Shape (hidden_dim2,)
        """
        # Apply the graph convolution operations
        z = self.infer_node_representation(g)
        # Find the mean of the approximate posterior distribution
        # q(z|g)
        z = self.output_regression[0](z)
        # Aggregate the nodes' representations into a graph representation
        # Note that one should use dgl.sum_nodes because this function will
        # return a tensor with a new first dimension whose size is the number
        # of subgraphs composing g
        with g.local_scope():
            g.ndata["h"] = z
            return self.aggregator(g)

    def infer_node_representation(self, g):
        """ Compute the latent representation of the nodes of input graph
        
        Args:
            g (DGLGraph)
                The molecular graph
        Returns:
            z: (FloatTensor): the latent encodings of the nodes
                Shape (N, hidden_dim2)
        """
        z = g.ndata["h"]
        for layer in self.gc:
            z = layer(g, z)
            # The output of a Graph Attention Networks is of shape
            # (N, H, D) where N is the number of nodes, H is the
            # number of attention heads and D is the output dimension of the
            # network.  Therefore, one need to "aggregate" the output
            # from multiple attention heads. Certainly one way of doing so
            # is by taking the average across the heads.
            if len(z.shape) > 2:
                z = torch.mean(z, dim=1) 
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
        z = self.infer_node_representation(g)
        theta = [parameter.forward(z) for parameter in self.output_regression]
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

    def get_encoder(self):
        return self.gc


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for edge prediction."""

    def __init__(self, dropout, act=lambda x: x):
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
    def __init__(self, dropout, feature_dim, num_atom_types, act=lambda x: x):
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
        node_preds = self.linear(z_prime)
        return (adj, node_preds)