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
            hidden_dim=64,
            unsup_scale=1):

        super(SemiSupervisedNet, self).__init__(
            representation, output_regression, measurement_dimension, noise_model)

        self.hidden_dim = hidden_dim
        # grab the last dimension of `representation`
        representation_hidden_units = [
                layer for layer in list(representation.modules())\
                        if hasattr(layer, 'out_features')][-1].out_features
        
        if output_regression is None:
            # make the output regression as simple as a linear one
            # if nothing is specified
            self._output_regression = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(
                            torch.nn.Linear(representation_hidden_units, self.hidden_dim),
                            torch.nn.Linear(self.hidden_dim, measurement_dimension)
                        ) for _ in range(2) # now we hard code # of parameters
                    ])

            def output_regression(theta):
                return [f(theta) for f in self._output_regression]
                
        self.output_regression = output_regression
        # unsupervised scale is to balance between the supervised and unsupervised
        # loss term. It should be r if one synthesizes the semi-supervised data
        # using prepare_semi_supeprvised_data_from_labelled_data
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
        not_none_ind = [i for i in range(len(y)) if y[i] != None]
        supervised_loss = torch.tensor(0)
        if sum(not_none_ind) != 0:
            # Only compute supervised loss for the labelled data
            h_not_none = h[not_none_ind, :]
            y_not_none = [y[idx] for idx in not_none_ind]
            y_not_none = torch.tensor(y_not_none).unsqueeze(1).to(y_not_none[0].device)
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
            decoded_subgraphs = [self.representation.dc(g_sample.ndata["h"])
                                 for g_sample in gs_unbatched]
            # print(mu, logvar) # .to(g.ndata["h"].device)
            unsup_loss = negative_elbo(decoded_subgraphs, mu, logvar, g)
            return unsup_loss

    def compute_graph_representation_from_node_representation(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            h = self.representation.aggregator(g)
            return h

def prepare_semi_supervised_data(unlabeled_data, labeled_data):
    # Mix labelled and unlabelled data together
    semi_supervised_data = []
    
    for (g, y) in unlabeled_data:
        semi_supervised_data.append((g, None))
    for (g, y) in labeled_data:
        semi_supervised_data.append((g, y))

    return semi_supervised_data


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

 
def prepare_semi_supervised_data_from_labeled_data(labeled_data, r=0.2, seed=2666):
    """
    r is the ratio of labelled data turning into unlabelled
    """
    semi_data = []
    small_labeled_data = []
    
    np.random.seed(seed)
    for (g,y) in labeled_data:
        if np.random.rand() < r:
            semi_data.append((g, y))
            small_labeled_data.append((g,y))
        else:
            semi_data.append((g, None))
    return semi_data, small_labeled_data


def train_and_test_semisupervised(model, optimizer, semi_train_data,
                                  train_labeled, test_labeled,
                                  n_epochs=100):
    semi_train = Train(net=model, data=semi_train_data, optimizer=optimizer, n_epochs=n_epochs)
    semi_train.train()

    # Measure trained labelled data
    train_metrics = Test(net=model, data=train_labelled,
                         states=semi_train.states,
                         metrics=[pinot.rmse, pinot.r2, pinot.avg_nll])
    semi_supervised_train_results = train_metrics.test()

    # Measure metrics on labelled data
    test_metrics = Test(net=model, data=test_labeled,
                        states=semi_train.states,
                        metrics=[pinot.rmse, pinot.r2, pinot.avg_nll])
    semi_supervised_test_results = test_metrics.test()
    
    return semi_supervised_train_results, semi_supervised_test_results, semi_train.states