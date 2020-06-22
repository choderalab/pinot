import torch
from pinot import Net
from pinot.regressors import ExactGaussianProcessRegressor, NeuralNetworkRegressor

class MultitaskNet(Net):
    """ An object that combines the representation and parameter
    learning, puts into a predicted distribution and calculates the
    corresponding divergence.

    Attributes
    ----------
    representation: a `pinot.representation` module
        the model that translates graphs to latent representations
    """
    def __init__(
        self,
        representation,
        output_regressor=NeuralNetworkRegressor,
        **kwargs
    ):
        super(MultitaskNet, self).__init__(
            representation,
            output_regressor,
            **kwargs)
        
        self.output_regressors = torch.nn.ModuleDict()
        self.kwargs = kwargs

    def condition(self, g, task):
        """ Compute the output distribution with sampled weights.
        """
        task = str(task)

        # get representation
        h = self.representation(g)
        
        # get output regressor for a particular task
        self.output_regressor = self._get_regressor(task)

        # get distribution for input
        distribution = self.output_regressor.condition(h)

        return distribution

    def loss(self, g, y):
        """ Compute the loss from input graph and corresponding y.
        """
        loss = 0.0
        
        # for each task in the data, split up data
        h = self.representation(g)
        l = self._generate_mask(y)

        for task, mask in enumerate(l.T):
            
            # switch to regressor for that task
            self.output_regressor = self._get_regressor(task)

            if isinstance(self.output_regressor, ExactGaussianProcessRegressor):
                # mask input if ExactGP
                h_task = self._mask_tensor(h, mask)
                y_task = self._mask_tensor(y, mask, task)
                loss += self.output_regressor.loss(h_task, y_task).mean()
            else:
                # mask output if VariationalGP
                distribution = self.output_regressor.condition(h)
                y_dummy = self._generate_y_dummy(y, task)
                loss += -distribution.log_prob(y_dummy)[mask].mean()

        return loss
    
    def _get_regressor(self, task):
        """ Returns regressor for a task.
        """
        # ModuleDict needs str
        task = str(task)
        
        # if we already instantiated the output_regressor
        if task not in self.output_regressors:

            # get the type of self.output_regressor, and instantiate it
            self.output_regressors[task] = self.output_regressor_cls(
                self.representation_out_features,
                **self.kwargs
                )

            # move to cuda if the parent net is
            if next(self.parameters()).is_cuda:
                self.output_regressors[task].cuda()
        
        return self.output_regressors[task]
    
    def _generate_y_dummy(self, y, task):
        """ Generates y dummy - fill nans with zeros.
        """
        y_dummy = y[:, task]
        y_dummy[torch.isnan(y_dummy)] = 0
        return y_dummy.view(-1, 1)

    def _mask_tensor(self, x, mask, task=None):
        """ Subsets data given mask for particular task.
        """
        if task != None:
            ret = x[mask, task].unsqueeze(-1)
        else:
            ret = x[mask]
        return ret

    def _generate_mask(self, y):
        """ Creates a boolean mask where y is nan.
        """
        return ~torch.isnan(y)