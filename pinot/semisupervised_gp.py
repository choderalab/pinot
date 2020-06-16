import pinot
import torch
import dgl
from pinot.generative.torch_gvae.loss import negative_elbo
import numpy as np
import math
from pinot.inference.gp.kernels.rbf import RBF
from pinot.semisupervised import SemiSupervisedNet

class SemiSupervisedGaussianProcess(SemiSupervisedNet):
    def __init__(self, representation,
            kernel=None,
            output_regression=None,
            measurement_dimension=1,
            noise_model='normal-heteroschedastic',
            log_sigma=-3.0,
            hidden_dim=64,
            unsup_scale=1):

        super(SemiSupervisedGaussianProcess, self).__init__(
            representation, output_regression, measurement_dimension, noise_model, hidden_dim, unsup_scale)

        if not hasattr(representation, "infer_node_representation"):
            print("The current implementation requires representation to have a infer_node_representation function")
            assert(False)

        self.hidden_dim = hidden_dim
        self.log_sigma = torch.nn.Parameter(
                torch.tensor(
                    log_sigma))

        if kernel is None:
            self.kernel = RBF()
        else:
            self.kernel = kernel
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

    def forward(self, g):
        # Copy exact_GPR for now
        # Store all the h's whose y is not None
        # Store the y's that is not None
        h_node = self.representation.infer_node_representation(g) # We always call this

        # Feed h -> into baseKernel

        # Get ys that are not None, and the corresponding h
        # Feed that into GP
        # Store all the gs,

        # Do decode and elbo loss
        theta = [parameter.forward(h_node) for parameter in self.representation.output_regression]
        return theta


    def loss(self, g, y):
        """ Compute the loss with a input graph and a set of parameters.
        """
        # Copy exact_GPR for now
        # Store all the h's whose y is not None
        # Store the y's that is not None
        h_node = self.representation.infer_node_representation(g) # We always call this

        # Feed h -> into baseKernel

        # Get ys that are not None, and the corresponding h
        # Feed that into GP
        # Store all the gs,

        # Do decode and elbo loss
        theta = [parameter.forward(h_node) for parameter in self.representation.output_regression]

        mu, logvar = theta
        approx_posterior = torch.distributions.normal.Normal(
                    loc=mu,
                    scale=torch.exp(logvar))

        z_sample = approx_posterior.rsample()
        # Compute unsupervised loss
        # Create a local scope so as not to modify the original input graph
        unsup_loss = self.compute_unsupervised_loss(g, z_sample, mu, logvar)     
        h_graph = self.compute_graph_representation_from_node_representation(g, h_node)

        # Get the indices of the labelled data
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        not_none_ind = [i for i in range(len(y)) if y[i] != None]
        supervised_loss = torch.tensor(0.)

        # Suppose that the first loop of semi-supervised bayesian optimization presents
        # no labels, the loss_GP will not be invoked, next call to condition will
        # result in error because `self._x_tr` is not set without calls
        # to `loss_GP`
        if not hasattr(self, "_x_tr"):
            self._x_tr = h_graph
            self._y_tr = y

        if sum(not_none_ind) != 0:
            # Only compute supervised loss for the labelled data
            # Using GPs
            print(h_graph)
            h_not_none = h_graph[not_none_ind, :]
            y_not_none = [y[idx] for idx in not_none_ind]
            y_not_none = torch.tensor(y_not_none).unsqueeze(1).to(y_not_none[0].device)
            # dist_y = self.compute_pred_distribution_from_rep(h_not_none)
            # supervised_loss = -dist_y.log_prob(y_not_none)
            gp_loss = self.loss_GP(h_not_none, y_not_none)
            supervised_loss += gp_loss.sum()
            
        return unsup_loss*self.unsup_scale + supervised_loss.sum()

    def loss_GP(self, x_tr, y_tr):
        r""" Compute the loss.
        Note
        ----
        Defined to be negative Gaussian likelihood.
        Parameters
        ----------
        x_tr : torch.tensor, shape=(batch_size, ...)
            training data.
        y_tr : torch.tensor, shape=(batch_size, 1)
            training data measurement.
        """
        # point data to object
        self._x_tr = x_tr
        self._y_tr = y_tr

        # get the parameters

        # This function needs to return `h` for unsupervised loss
        k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha\
            = self._get_kernel_and_auxiliary_variables(x_tr, y_tr)

        # we return the exact nll with constant
        nll = 0.5 * (y_tr.t() @ alpha) + torch.trace(l_low)\
            + 0.5 * y_tr.shape[0] * math.log(2.0 * math.pi)

        return nll

    def _get_kernel_and_auxiliary_variables(
            self, x_tr, y_tr, x_te=None,
        ):
        
        # grab sigma
        sigma = torch.exp(self.log_sigma)

        # compute the kernels
        k_tr_tr = self._perturb(self.kernel.forward(x_tr, x_tr))

        # This function needs to return `h` as well
        if x_te is not None: # during test
            k_te_te = self._perturb(self.kernel(x_te, x_te))
            k_te_tr = self._perturb(self.kernel(x_te, x_tr))
            # k_tr_te = self.forward(x_tr, x_te)
            k_tr_te = k_te_tr.t() # save time

        else: # during train
            k_te_te = k_te_tr = k_tr_te = k_tr_tr

        # (batch_size_tr, batch_size_tr)
        k_plus_sigma = k_tr_tr + torch.exp(self.log_sigma) * torch.eye(
                k_tr_tr.shape[0],
                device=k_tr_tr.device)

        # (batch_size_tr, batch_size_tr)
        l_low = torch.cholesky(k_plus_sigma)
        l_up = l_low.t()

        # (batch_size_tr. 1)
        l_low_over_y, _ = torch.triangular_solve(
            input=y_tr,
            A=l_low,
            upper=False)

        # (batch_size_tr, 1)
        alpha, _ = torch.triangular_solve(
            input=l_low_over_y,
            A=l_up,
            upper=True)

        return k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha


    def condition(self, g, x_tr=None, y_tr=None, sampler=None):
        x_te = self.representation(g)
        return self._condition(x_te, x_tr, y_tr, sampler)


    def _condition(self, x_te, x_tr=None, y_tr=None, sampler=None):
        r""" Calculate the predictive distribution given `x_te`.

        Note
        ----
        Here we allow the speicifaction of sampler but won't actually
        use it here.

        Parameters
        ----------
        x_tr : torch.tensor, shape=(batch_size, ...)
            training data.
        y_tr : torch.tensor, shape=(batch_size, 1)
            training data measurement.
        x_te : torch.tensor, shape=(batch_size, ...)
            test data.
        sigma : float or torch.tensor, shape=(), default=1.0
            noise parameter.
        """
        if x_tr is None:
            x_tr = self._x_tr
        if y_tr is None:
            y_tr = self._y_tr

        # get parameters
        k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha\
            = self._get_kernel_and_auxiliary_variables(
                x_tr, y_tr, x_te)

        # compute mean
        # (batch_size_te, 1)
        mean = k_te_tr @ alpha

        # (batch_size_tr, batch_size_te)
        v, _ = torch.triangular_solve(
            input=k_tr_te,
            A=l_low,
            upper=False)

        # (batch_size_te, batch_size_te)
        variance = k_te_te - v.t() @ v

        # ensure symetric
        variance = 0.5 * (variance + variance.t())

        # $ p(y|X) = \int p(y|f)p(f|x) df $
        # variance += torch.exp(self.log_sigma) * torch.eye(
        #         *variance.shape,
        #         device=variance.device)

        # construct noise predictive distribution
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            mean.flatten(),
            variance)

        return distribution


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

    def _perturb(self, k):
        """ Add small noise `epsilon` to the diagnol of kernels.

        Parameters
        ----------
        k : torch.tensor
            kernel to be perturbed.
        """

        # introduce noise along the diagnol
        noise = 1e-5 * torch.eye(
                *k.shape,
                device=k.device)

        return k + noise