import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from pinot.representation.dgl_legacy import GN
from pinot.generative.torch_gvae.loss import negative_ELBO,\
    negative_ELBO_with_node_prediction, negative_elbo
from pinot.generative.torch_gvae.decoder import *

class GCNModelVAE(nn.Module):
    """Graph convolutional neural networks for VAE
    """
    def __init__(self, input_feat_dim, gcn_type="GraphConv", gcn_init_args ={},\
            gcn_hidden_dims=[256, 128], embedding_dim=64, dropout=0.0, \
            num_atom_types=100, \
            aggregation_function="sum"):
        """ Construct a VAE with GCN
        Args:
            input_feature_dim: Number of input features for each atom/node

            hidden_dims: list
                The hidden dimensions of the graph convolution layers
            
            embedding_dim: int
                The dimension of the latent representation of the nodes

            num_atom_types: The number of possible atom types

            aggregation_function: str ["sum", "mean"]
                function used to aggregate the node feature
                    vectors into a single graph representation

        """
        super(GCNModelVAE, self).__init__()

        # 2. Decoder
        self.dc = SequentialDecoder(embedding_dim, num_atom_types)
        self.num_atom_types = num_atom_types

        # Encoder
        # 1b. Mapping from node embedding to predictive distribution parameter
        self.gcn_hidden_dims = gcn_hidden_dims
        self.embedding_dim = embedding_dim
        self.output_regression = nn.ModuleList([
            nn.Linear(gcn_hidden_dims[-1], embedding_dim),
            nn.Linear(gcn_hidden_dims[-1], embedding_dim),
        ])

        # 1a. Graph convolution layers
        # Note that we define this here because in 'Net', the model automatically
        # finds the input dimension of the output-regressor by inspecting the
        # last torch module of the representation layer. And the dimension of the 
        # output of `GCNModelVAE.forward` should match the input dimension to `Net`

        if aggregation_function == "sum":
            self.aggregator = self._sum_aggregate
        elif aggregation_function == "mean":
            self.aggregator = self._mean_aggregate

        assert(len(gcn_hidden_dims) > 0)
        self.gcn_modules = []
        for (dim_prev, dim_post) in zip([input_feat_dim] + gcn_hidden_dims[:-1],\
                gcn_hidden_dims):
            self.gcn_modules.append(GN(dim_prev, dim_post, gcn_type, gcn_init_args))
        self.gc = nn.ModuleList(self.gcn_modules)

    def forward(self, g):
        """ Compute the latent representation of the input graph. This
        function is slightly different from `infer_node_representation`
        only in that it aggregates the node features into one vector.
        This is so that `GVAE` can be directly used as part of `Net`.
        
        Args:
            g (DGLGraph)
                The molecular graph
        Returns:
            z: List(FloatTensor): the latent encodings of the sub-graphs
                composing batch g. Each latent encoding in this list has 
                dimension gcn_hidden_dims[-1]
        """
        # Apply the graph convolution operations
        z = self.infer_node_representation(g)
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
        
        # Then decode
        # First sample latent node representations
        z_sample = approx_posterior.rsample()
        # z_sample = mu

        # Create a local scope so as not to modify the original input graph
        with g.local_scope():
            # Create a new graph with sampled representations
            g.ndata["h"] = z_sample
            # Unbatch into individual subgraphs
            gs_unbatched = dgl.unbatch(g)
            # Decode each subgraph
            decoded_subgraphs = [self.dc(g_sample.ndata["h"]) \
                for g_sample in gs_unbatched]
            return decoded_subgraphs, mu, logvar


    def loss(self, g, y=None):
        """ Compute negative ELBO loss
        """
        decoded_subgraphs, mu, logvar = self.encode_and_decode(g)
        loss = negative_elbo(decoded_subgraphs, mu, logvar, g)
        return loss

    def _sum_aggregate(self, g):
        return dgl.sum_nodes(g, "h")

    def _mean_aggregate(self, g):
        return dgl.mean_nodes(g, "h")

    def infer_graph_representation(self, g):
        h = self.infer_node_representation(g)
        with g.local_scope():
            g.ndata["h"] = h
            return self.aggregator(g)