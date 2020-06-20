# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import torch.nn as nn
import pinot
from pinot.generative.losses import negative_elbo
from pinot.net import Net

# =============================================================================
# MODULE CLASSES
# =============================================================================

class SemiSupervisedNet(Net):
    def __init__(self, representation, output_regressor,
            decoder,
            unsup_scale=1.,
            cuda=True,
            generative_hidden_dim=64):

        super(SemiSupervisedNet, self).__init__(
            representation=representation,
            output_regressor=output_regressor
        )

        # Recommended class: pinot.representation.sequential.Sequential
        # Representation needs to have these functions
        # representation.forward(g, pool) -> h_graph or h_node depending on pool
        # representation.pool(h_node) -> h_graph
        assert(hasattr(representation, "forward"))
        assert(hasattr(representation, "pool"))

        # grab the last dimension of `representation`
        self.representation_dim = [
                layer for layer in list(self.representation.modules())\
                        if hasattr(layer, 'out_features')][-1].out_features

        # Output_regressor_generative:
        assert(hasattr(decoder, "embedding_dim"))
        self.generative_hidden_dim = generative_hidden_dim
        self.output_regressor_generative = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(self.representation_dim, self.generative_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.generative_hidden_dim, decoder.embedding_dim),
            ) for _ in range(2) # mean and logvar
        ])

        # Recommended class: pinot.generative.decoder.DecoderNetwork
        # Decoder needs to satisfy:
        # decoder.loss(g, z_sample) -> compute reconstruction loss
        assert(hasattr(decoder, "forward"))
        self.decoder = decoder

        # Output regressor needs to satisfy
        # output_regressor.loss(h_graph, y) -> supervised loss
        # output_regressor.condition(h_graph) -> pred distribution
        assert(hasattr(output_regressor, "loss"))
        assert(hasattr(output_regressor, "condition"))
        # self.output_regressor = output_regressor

        # Move to CUDA if available
        self.cuda = cuda
        self.device = torch.device("cuda:0" if cuda else "cpu:0")
        self.representation.to(self.device)
        self.output_regressor_generative.to(self.device)
        self.decoder.to(self.device)
        self.output_regressor.to(self.device)

        # Zookeeping
        self.unsup_scale = unsup_scale


    def loss(self, g, y):
        """ Compute the loss function
        """
        # Move to CUDA if available
        g.to(self.device)
        # Compute the node representation
        # print(g.ndata["h"].shape)
        # Call this function to compute the nodes representations
        h = self.representation.forward(g, pool=None) # We always call this
        # print(g.ndata["h"].shape)
        # Compute unsupervised loss
        unsup_loss = self.loss_unsupervised(g, h)
        # Compute the graph representation from node representation
        # print(g.ndata["h"].shape)
        # Then compute graph representation, by pooling
        h = self.representation.pool(g, h)
        # print(g.ndata["h"].shape)

        supervised_loss = torch.tensor(0.)

        # Then compute supervised loss
        if len(y[~torch.isnan(y)]) != 0:
            # Only compute supervised loss for the labeled data
            h_not_none = h[~torch.isnan(y).flatten(), :]
            y_not_none = y[~torch.isnan(y)].unsqueeze(1)
            # Convert to cuda if available
            y_not_none = y_not_none.to(self.device)
            # The output-regressor needs to implement a loss function
            supervised_loss = self.loss_supervised(h_not_none, y_not_none)

        total_loss = supervised_loss.sum() + unsup_loss*self.unsup_scale
        return total_loss

    def loss_supervised(self, h, y):
        return self.output_regressor.loss(h, y)

    def loss_unsupervised(self, g, h):
        """

        """
        # h = (number of nodes, embedding_dim)
        theta = [parameter.forward(h) for parameter in self.output_regressor_generative]
        mu, logvar = theta[0], theta[1]
        # (number of nodes, z_dimension)
        distribution = torch.distributions.normal.Normal(
                    loc=mu,
                    scale=torch.exp(logvar)
        )
        z_sample = distribution.rsample()
        decoded_subgraphs = self.decoder(g, z_sample)
        loss = negative_elbo(decoded_subgraphs, mu, logvar, g)
        return loss
