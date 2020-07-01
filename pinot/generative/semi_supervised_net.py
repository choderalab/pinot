# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import torch.nn as nn
import pinot
from pinot.generative.losses import negative_elbo
from pinot.generative.decoder import DecoderNetwork
from pinot.regressors.neural_network_regressor import NeuralNetworkRegressor

# =============================================================================
# MODULE CLASSES
# =============================================================================


class SemiSupervisedNet(pinot.Net):
    """ """

    def __init__(
        self,
        representation,
        output_regressor=NeuralNetworkRegressor,
        decoder=DecoderNetwork,
        unsup_scale=1.0,
        embedding_dim=64,
        generative_hidden_dim=64,
    ):

        super(SemiSupervisedNet, self).__init__(
            representation=representation, output_regressor=output_regressor
        )

        # Recommended class: pinot.representation.sequential.Sequential
        # Representation needs to have these functions
        # representation.forward(g, pool) -> h_graph or h_node depending on pool
        # representation.pool(h_node) -> h_graph
        assert hasattr(self.representation, "forward")
        assert hasattr(self.representation, "pool")

        # grab the last dimension of `representation`
        self.representation_dim = self.representation_out_features

        # pass in decoder as class
        self.decoder_cls = decoder
        # Recommended class: pinot.generative.decoder.DecoderNetwork
        # Decoder needs to satisfy:
        # decoder.loss(g, z_sample) -> compute reconstruction loss
        # Embedding_dim is the dimension of the z_sample vector, or the
        # input of the decoder
        self.embedding_dim = embedding_dim
        self.decoder = decoder(embedding_dim=embedding_dim, num_atom_types=100)

        # Output_regressor_generative:
        assert hasattr(self.decoder, "forward")
        assert hasattr(self.decoder, "embedding_dim")
        # Generative hidden dim is the size of the hidden layer of the
        # generative output regressor network
        self.generative_hidden_dim = generative_hidden_dim

        self.output_regressor_generative = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.representation_dim, self.generative_hidden_dim
                    ),
                    nn.Tanh(),
                    nn.Linear(
                        self.generative_hidden_dim, self.decoder.embedding_dim
                    ),
                )
                for _ in range(2)  # mean and logvar
            ]
        )

        # Output regressor needs to satisfy
        # output_regressor.loss(h_graph, y) -> supervised loss
        # output_regressor.condition(h_graph) -> pred distribution
        assert hasattr(self.output_regressor, "loss") or hasattr(
            self.output_regressor, "condition"
        )

        # Zookeeping
        self.unsup_scale = unsup_scale

    def loss(self, g, y):
        """Compute the loss function

        Parameters
        ----------
        g :
            
        y :
            

        Returns
        -------

        """
        # Compute the node representation
        # Call this function to compute the nodes representations
        h = self.representation.forward(g, pool=None)  # We always call this
        # Compute unsupervised loss
        total_loss = self.loss_unsupervised(g, h) * self.unsup_scale
        # Compute the graph representation from node representation
        # Then compute graph representation, by pooling
        h = self.representation.pool(g, h)

        # Then compute supervised loss
        if len(y[~torch.isnan(y)]) != 0:
            # Only compute supervised loss for the labeled data
            h_labeled = h[~torch.isnan(y).flatten(), :]
            y_labeled = y[~torch.isnan(y)].unsqueeze(1)

            if (
                self.has_exact_gp is True
            ):  # Save the last graph batch + ys if exact GP
                self.y_last = y_labeled
                self.g_last = dgl.batch(
                    [
                        g
                        for i, g in enumerate(dgl.unbatch(g))
                        if ~torch.isnan(y[i])
                    ]
                )

            # The output-regressor needs to implement a loss function
            supervised_loss = self.loss_supervised(h_labeled, y_labeled)
            total_loss += supervised_loss.sum()
        return total_loss

    def loss_supervised(self, h, y):
        """

        Parameters
        ----------
        h :
            
        y :
            

        Returns
        -------

        """
        # If output regressor has loss function implemented
        return self._loss(h, y)

    def loss_unsupervised(self, g, h):
        """

        Parameters
        ----------
        g :
            
        h :
            

        Returns
        -------

        """
        # h = (number of nodes, embedding_dim)
        theta = [
            parameter.forward(h)
            for parameter in self.output_regressor_generative
        ]
        mu, logvar = theta[0], theta[1]
        # (number of nodes, z_dimension)
        distribution = torch.distributions.normal.Normal(
            loc=mu, scale=torch.exp(logvar)
        )
        z_sample = distribution.rsample()
        # Compute the ELBO loss
        # First the reconstruction loss (~~ expected log likelihood)
        recon_loss = self.decoder.decode_and_compute_recon_error(g, z_sample)
        # KL-divergence term
        KLD = (
            -0.5
            / g.number_of_nodes()
            * torch.sum(
                torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1)
            )
        )
        return recon_loss + KLD
