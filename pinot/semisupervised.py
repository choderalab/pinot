import pinot
import torch
import dgl
from pinot.generative.torch_gvae.loss import negative_elbo
import numpy as np

class SemiSupervisedNet(pinot.Net):
    def __init__(self, representation,         
            output_regression=None,
            measurement_dimension=1,
            noise_model='normal-heteroschedastic',
            unsup_scale=1):

        super(SemiSupervisedNet, self).__init__(
            representation, output_regression, measurement_dimension, noise_model)
        self.unsup_scale = unsup_scale
    
    def loss(self, g, y):
        h = self.representation.infer_node_representation(g) # We always call this

        # Do decode and elbo loss
        theta = [parameter.forward(h) for parameter in self.representation.output_regression]
        mu  = theta[0]
        logvar = theta[1]
        approx_posterior = torch.distributions.normal.Normal(
                    loc=theta[0],
                    scale=torch.exp(theta[1]))

        z_sample = approx_posterior.rsample()
        # Compute unsupervised loss
        # Create a local scope so as not to modify the original input graph
        unsup_loss = self.compute_unsupervised_loss(g, z_sample, mu, logvar)     
        h = self.compute_graph_representation_from_node_representation(g, h)

        # Get the indices of the labelled data
        not_none_ind =[i for i in range(len(y)) if y[i] != None]
        supervised_loss = torch.tensor(0)
        if sum(not_none_ind) != 0:
            # Only compute supervised loss for the labelled data
            h_not_none = h[not_none_ind, :]
            y_not_none = [y[idx] for idx in not_none_ind]
            y_not_none = torch.tensor(y_not_none).unsqueeze(1)
            dist_y = self.compute_pred_distribution_from_rep(h_not_none)
            supervised_loss = -dist_y.log_prob(y_not_none)
            
        return unsup_loss*self.unsup_scale + supervised_loss.sum() 


    def compute_pred_distribution_from_rep(self, h):
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

        return distribution

    def compute_unsupervised_loss(self, g, z_sample, mu, logvar):
        with g.local_scope():
            # Create a new graph with sampled representations
            g.ndata["h"] = z_sample
            # Unbatch into individual subgraphs
            gs_unbatched = dgl.unbatch(g)
            # Decode each subgraph
            decoded_subgraphs = [self.representation.dc(g_sample.ndata["h"]) \
                for g_sample in gs_unbatched]
            unsup_loss = negative_elbo(decoded_subgraphs, mu, logvar, g)
            return unsup_loss

    def compute_graph_representation_from_node_representation(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            h = self.representation.aggregator(g)
            return h


def batch_semi_supervised(ds, batch_size, seed=2666):
    """ Batch graphs and values after shuffling.
    """
    # get the numebr of data
    n_data_points = len(ds)
    n_batches = n_data_points // batch_size  # drop the rest

    np.random.seed(seed)
    np.random.shuffle(ds)
    gs, ys = tuple(zip(*ds))

    gs_batched = [
        dgl.batch(gs[idx * batch_size : (idx + 1) * batch_size])
        for idx in range(n_batches)
    ]

    ys_batched = [
        list(ys[idx * batch_size : (idx + 1) * batch_size])
        for idx in range(n_batches)
    ]

    return list(zip(gs_batched, ys_batched))


def prepare_semi_supervised_training_data(unlabelled_data, labelled_data, batch_size=32):
    # Mix t
    semi_supervised_data = []
    
    for (g, y) in unlabelled_data:
        semi_supervised_data.append((g, None))
    for (g, y) in labelled_data:
        semi_supervised_data.append((g, y))
        
    semi_supervised_data = batch_semi_supervised(semi_supervised_data, batch_size)
    return semi_supervised_data


def prepare_semi_supervised_data_from_labelled_data(labelled_data, r=0.2, seed=2666):
    """
    r is the ratio of labelled data turning into unlabelled
    """
    semi_data = []
    small_labelled_data = []
    
    np.random.seed(seed)
    for (g,y) in labelled_data:
        if np.random.rand() < r:
            semi_data.append((g, y))
            small_labelled_data.append((g,y))
        else:
            semi_data.append((g, None))
    return semi_data, small_labelled_data
