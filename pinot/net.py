""" Combine representation, parameterization, and distribution class
to construct a model.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


class Net(torch.nn.Module):
    """ An object that combines the representation and parameter
    learning, puts into a predicted distribution and calculates the
    corresponding divergence.
    Attributes
    ----------
    representation: a `pinot.representation` module
        the model that translates graphs to latent representations
    output_regression: a `torch.nn.Module` or None,
        if None, this will be set as a simple `Linear` layer that inputs
        the latent dimension and output the number of parameters for
        `self.distribution_class`
    noise_model: either a string (
        one of 
            'normal-homoschedastic',
            'normal-heteroschedastic',
            'normal-homoschedastic-fixed') 
        or a function that transforms a set of parameters.
    """

    def __init__(
        self,
        representation,
        output_regression=None,
        measurement_dimension=1,
        noise_model='normal-heteroschedastic',
    ):
        
        super(Net, self).__init__()
        self.representation = representation

        # grab the last dimension of `representation`
        representation_hidden_units = [
                layer for layer in list(self.representation.modules())\
                        if hasattr(layer, 'out_features')][-1].out_features


        if output_regression is None:
            # make the output regression as simple as a linear one
            # if nothing is specified
            self._output_regression = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(representation_hidden_units, measurement_dimension)\
                                for _ in range(2) # now we hard code # of parameters
                    ])

            def output_regression(theta):
                return [f(theta) for f in self._output_regression]

        self.output_regression = output_regression
        self.measurement_dimension=measurement_dimension 
        self.noise_model = noise_model
        self.representation_hidden_units = representation_hidden_units

    def forward(self, g):
        """ Forward pass.
        """
        # graph representation $\mathcal{G}$
        # ->
        # latent representation $h$
        h = self.representation(g)

        # latent representation $h$
        # ->
        # parameters $\theta$
        theta = self.output_regression(h)
       
        return theta

    def _condition(self, g):
        """ Compute the output distribution.
        """

        if self.noise_model == 'normal-heteroschedastic':
            mu, log_sigma = self.forward(g)
            distribution = torch.distributions.normal.Normal(
                    loc=mu,
                    scale=torch.exp(log_sigma))

        elif self.noise_model == 'normal-homoschedastic':
            mu, _ = self.forward(g)

            # initialize a `LOG_SIMGA` if there isn't one
            if not hasattr(self, 'LOG_SIGMA'):
                self.LOG_SIGMA = torch.zeros((1, self.measurement_dimension))
                self.LOG_SIGMA.requires_grad = True

            distribution = torch.distributions.normal.Normal(
                    loc=mu,
                    scale=torch.exp(self.LOG_SIGMA))

        elif self.noise_model == 'normal-homoschedastic-fixed':
            mu, _ = self.forward(g)
            distribution = torch.distributions.normal.Normal(
                    loc=mu, 
                    scale=torch.ones((1, self.measurement_dimension)))

        else:
            assert isinstance(
                    self.noise_model,
                    dict)

            distribution = self.noise_model[distribution](
                    self.noise_model[kwargs])


        return distribution

    def condition(self, g, sampler=None, n_samples=64):
        """ Compute the output distribution with sampled weights.
        """
        if sampler is None:
            return self._condition(g)

        if not hasattr(sampler, 'sample_params'):
            return self._condition(g)

        # initialize a list of distributions
        distributions = []

        for _ in range(n_samples):
            sampler.sample_params()
            distributions.append(self._condition(g))

        # get the parameter of these distributions
        # NOTE: this is not necessarily the most efficienct solution
        # since we don't know the memory footprint of 
        # torch.distributions
        mus, sigmas = zip(*[
                (distribution.loc, distribution.scale)
                for distribution in distributions])

        # concat parameters together
        # (n_samples, batch_size, measurement_dimension)
        mu = torch.stack(mus).cpu() # distribution no cuda
        sigma = torch.stack(sigmas).cpu()

        # construct the distribution
        distribution = torch.distributions.normal.Normal(
                loc=mu,
                scale=sigma)

        # make it mixture
        distribution = torch.distributions.mixture_same_family\
                .MixtureSameFamily(
                        torch.distributions.Categorical(
                            torch.ones(mu.shape[0],)),
                        torch.distributions.Independent(distribution, 2))
       
        return distribution

    def loss(self, g, y):
        """ Compute the loss with a input graph and a set of parameters.
        """

        distribution = self.condition(g)

        return -distribution.log_prob(y)


class MultiTaskNet(nn.Module):
    """ An object that combines the representation and parameter
    learning, puts into a predicted distribution and calculates the
    corresponding divergence.
    Attributes
    ----------
    representation: a `pinot.representation` module
        the model that translates graphs to latent representations
    output_regression: a `torch.nn.Module` or None,
        Prototype of the network to use for the head; inputs
        the latent dimension and output the number of parameter;s for
        `self.distribution_class`
    noise_model: either a string (
        one of 
            'normal-homoschedastic',
            'normal-heteroschedastic',
            'normal-homoschedastic-fixed')
        or a function that transforms a set of parameters.
    """
    def __init__(self, net_class, representation, num_nets,
                 noise_model='normal-heteroschedastic'):
        """
        Parameters
        ----------
        net_class : pinot.Net class
        """
        super(MultiTaskNet, self).__init__()
        
        self.representation = representation
        self.num_nets = num_nets
        self.nets = nn.ModuleList([net_class(self.representation) for _ in range(self.num_nets)])
        self.noise_model = noise_model
        
    def forward(self, g, mask):
        """ Forward pass.
        """
        return self.apply('forward', mask, g)

    def condition(self, g, sampler=None):
        """ Compute the output distribution with sampled weights.
        """
        self.eval()
        mask = torch.ones(self.num_nets, dtype=torch.bool)
        return self.apply('condition', mask, g, sampler=sampler)

    def loss(self, g, y, mask):
        """ Compute the loss with a input graph and a set of parameters.
        """
        losses = torch.stack(self.apply('loss', mask, g, y))
        return losses.mean()

    def apply(self, func, mask, *args, **kwargs):
        """ Splits function across each task.
        """
        results = []
        
        # for each task
        for i in range(self.num_nets):
            
            # retrieve mask
            task_mask = mask[:,i]
            
            # mask the arguments
            task_args = self._mask_task_args(task_mask, i, *args)
            task_kwargs = self._mask_task_kwargs(task_mask, i, **kwargs)
            
            # run the function
            result = getattr(self.nets[i], func)(*task_args, **task_kwargs)
            results.append(result)

        return results
    
    def _mask_task_kwargs(self, task_mask, i, **kwargs):
        task_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, dgl.DGLGraph):
                v = self._mask_graph(v, task_mask)
                task_kwargs[k] = v
            elif isinstance(v, torch.Tensor):
                v = self._mask_tensor(v, task_mask, i)
                task_kwargs[k] = v
        return task_kwargs
        
    def _mask_task_args(self, task_mask, i, *args):
        task_args = []
        for v in args:
            if isinstance(v, dgl.DGLGraph):
                v = self._mask_graph(v, task_mask)
                task_args.append(v)
            elif isinstance(v, torch.Tensor):
                v = self._mask_tensor(v, task_mask, i)
                task_args.append(v)
        return task_args

    def _mask_graph(self, x, mask):
        if x.batch_size > 1:
            x = dgl.unbatch(x)
        return dgl.batch([x[i] for i, _ in enumerate(x) if mask[i]])

    def _mask_tensor(self, x, mask, idx):
        return x[mask, idx].unsqueeze(-1)


class GPyTorchNet(torch.nn.Module):
    """ An object that combines the representation and parameter
    learning, puts into a predicted distribution and calculates the
    corresponding divergence.

    Attributes
    ----------
    representation: a `pinot.representation` module
        the model that translates graphs to latent representations
    output_regression: a `torch.nn.Module` or None,
        if None, this will be set as a simple `Linear` layer that inputs
        the latent dimension and output the number of parameters for
        `self.distribution_class`
    noise_model: either a string (
        one of 
            'normal-homoschedastic',
            'normal-heteroschedastic',
            'normal-homoschedastic-fixed') 
        or a function that transforms a set of parameters.
    """

    def __init__(
        self,
        representation,
        output_regression=None,
        measurement_dimension=1,
        noise_model='normal-heteroschedastic',
    ):
        super(GPyTorchNet, self).__init__()
        self.representation = representation

        # grab the last dimension of `representation`
        representation_hidden_units = [
                layer for layer in list(self.representation.modules())\
                        if hasattr(layer, 'out_features')][-1].out_features


        if output_regression is None:
            # make the output regression as simple as a linear one
            # if nothing is specified
            self._output_regression = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(representation_hidden_units, measurement_dimension)\
                                for _ in range(2) # now we hard code # of parameters
                    ])

            def output_regression(theta):
                return [f(theta) for f in self._output_regression]

        self.output_regression = output_regression
        self.measurement_dimension=measurement_dimension 
        self.noise_model = noise_model
        self.representation_hidden_units = representation_hidden_units

    def forward(self, g):
        """ Forward pass.
        """
        # graph representation $\mathcal{G}$
        # ->
        # latent representation $h$
        h = self.scale(self.representation.forward(g))

        # latent representation $h$
        # ->
        # parameters $\theta$
        theta = self.output_regression.forward(h)
       
        return theta

    def scale(self, x):
        x = x - x.min(0)[0]
        x = 2 * (x / x.max(0)[0]) - 1
        return x


    def posterior(self, g):
        """ Compute the output distribution.
        """
        # put output_regression in eval mode
        self.output_regression.eval()
        # graph representation $\mathcal{G}$
        # ->
        # latent representation $h$
        h = self.scale(self.representation.forward(g))

        # latent representation $h$
        # ->
        # parameters $\theta$ using __call__
        distribution = self.output_regression(h)

        if self.noise_model == 'normal-heteroschedastic':
            pass

        elif self.noise_model == 'normal-homoschedastic':

            # initialize a `SIGMA` if there isn't one
            if not hasattr(self, 'SIGMA'):
                self.SIGMA = torch.zeros((1, self.measurement_dimension))
                self.SIGMA.requires_grad = True

            distribution = torch.distributions.normal.Normal(
                    loc=distribution.mean,
                    scale=self.SIGMA)

        elif self.noise_model == 'normal-homoschedastic-fixed':
            distribution = torch.distributions.normal.Normal(
                    loc=distribution.mean,
                    scale=torch.ones((1, self.measurement_dimension)))
        else:
            assert isinstance(
                    self.noise_model,
                    dict)

            distribution = self.noise_model[distribution](
                    self.noise_model[kwargs])

        return distribution

    def loss(self, g, y):
        """ Compute the loss with a input graph and a set of parameters.
        """
        # loss for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.output_regression.likelihood,
                                                       self.output_regression)
        distribution = self.forward(g)
        return -mll(distribution, y)