import pinot
import torch
import dgl
from pinot.generative.torch_gvae.loss import negative_elbo
import numpy as np
import sys


class SemiSupervisedNet(pinot.Net):
    def __init__(self, representation,
            output_regression=None,
            measurement_dimension=1,
            noise_model='normal-heteroschedastic',
            hidden_dim=64,
            unsup_scale=1):

        if not hasattr(representation, "infer_node_representation"):
            print("Representation needs to have infer_node_representation function")
            sys.exit(1)

        
        if not hasattr(representation, "graph_h_from_node_h"):
            print("Representation needs to have graph_h_from_node_h function")
            sys.exit(1)
    
    
        super(SemiSupervisedNet, self).__init__(
            representation, output_regression, measurement_dimension, noise_model)
        
        # grab the last dimension of `representation`
        self.representation_hidden_units = [
                layer for layer in list(representation.modules())\
                        if hasattr(layer, 'out_features')][-1].out_features

        if output_regression is None:
            # make the output regression as a 2-layer network
            # if nothing is specified
            self._output_regression = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(self.representation_hidden_units, hidden_dim),
                            torch.nn.Tanh(),
                            torch.nn.Linear(hidden_dim, measurement_dimension),
                        ) for _ in range(2) # mean and logvar
                    ])

            def output_regression(theta):
                return [f(theta) for f in self._output_regression]

        self.output_regression = output_regression
        self.hidden_dim = hidden_dim
        # unsupervised scale is to balance between the supervised and unsupervised
        # loss term. It should be r if one synthesizes the semi-supervised data
        # using prepare_semi_supeprvised_data_from_labeled_data
        self.unsup_scale = unsup_scale
    
    def loss(self, g, y):
        """ Compute the loss function

        """
        # Compute the node representation
        h = self.representation.infer_node_representation(g) # We always call this
        # Compute unsupervised loss
        unsup_loss = self.representation.decode_and_compute_loss(g, h)  
        # Compute the graph representation from node representation
        h    = self.representation.graph_h_from_node_h(g, h)
        
        not_none_ind =[i for i in range(len(y)) if y[i] is not None]
        supervised_loss = torch.tensor(0.)

        if len(not_none_ind) != 0:
            # Only compute supervised loss for the labeled data
            h_not_none = h[not_none_ind, :]
            y_not_none = [y[idx] for idx in not_none_ind]
            y_not_none = torch.tensor(y_not_none).unsqueeze(1)
            supervised_loss = self.compute_supervised_loss(h_not_none, y_not_none)

        total_loss = supervised_loss.sum() + unsup_loss*self.unsup_scale 

        return total_loss

    def compute_supervised_loss(self, h, y):
        """ Compute supervised loss

        Args:
            h (FloatTensor)

            y (FloatTensor)

        Returns:
            Compute the negative log likelihood
        
        """
        theta = self.output_regression(h)
        distribution = None
        if self.noise_model == 'normal-heteroschedastic':
            mu, log_sigma = theta
            distribution = torch.distributions.normal.Normal(
                    loc=mu,
                    scale=torch.exp(log_sigma))

        elif self.noise_model == 'normal-homoschedastic':
            mu, _ = theta

            # initialize a `LOG_SIMGA` if there isn't one
            if not hasattr(self, 'LOG_SIGMA'):
                self.LOG_SIGMA = torch.zeros((1, self.measurement_dimension))
                self.LOG_SIGMA.requires_grad = True

            distribution = torch.distributions.normal.Normal(
                    loc=mu,
                    scale=torch.exp(self.LOG_SIGMA))

        elif self.noise_model == 'normal-homoschedastic-fixed':
            mu, _ = theta
            distribution = torch.distributions.normal.Normal(
                    loc=mu, 
                    scale=torch.ones((1, self.measurement_dimension)))

        # negative log likelihood
        return -distribution.log_prob(y)