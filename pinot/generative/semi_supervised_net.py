# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import pinot
from pinot.generative.losses import negative_elbo

# =============================================================================
# MODULE CLASSES
# =============================================================================
class SemiSupervisedNet(pinot.Net):
    r""" Net object with semisupervised learning.

    """

    def __init__(self, output_regression_generative, decoder, *args, **kwargs):
        super(SemiSupervisedNet, self).__init__(*args, **kwargs)

        # bookkeeping
        self.output_regression_generative = output_regression_generative
        self.decoder = decoder

    def forward_no_pool(self, g):
        """ Forward pass for semisupervised training.
        """
        # (n_nodes, hidden_dim)
        return self.representation.forward(g, pool=None)

    @staticmethod
    def _condition_no_pool(h):
        theta = self.output_regression_generative(h)
        mu, log_var = theta
        distribution = torch.distributions.normal.Normal(loc=mu, scale=log_var)

        return distribution, mu, log_var

    def condition_no_pool(self, g):
        h = self.forward_no_pool(g)
        distribution, mu, log_var = self._condition_no_pool(h)
        return distribution

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
            decoded_subgraphs = [
                self.dc(g_sample.ndata["h"]) for g_sample in gs_unbatched
            ]
            return decoded_subgraphs, mu, logvar

    def loss_semisupervised(self, g):
        """ ELBO loss.
        """
        # encode and decode
        decoded_subgraphs, mu, logvar = encode_and_decode(g)

        # loss
        loss = negative_elbo(decoded_subgraphs, mu, logvar, g)

        return loss
