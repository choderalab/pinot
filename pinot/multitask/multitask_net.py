import pinot
import gpytorch
import torch

class MultiTaskNet(pinot.Net):
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
        output_regressor=pinot.inference.output_regressors.NeuralNetworkOutputRegressor,
        **kwargs
    ):

        super(MultiTaskNet, self).__init__(representation, output_regressor, **kwargs)
        self.output_regressors = torch.nn.ModuleDict()

    def condition_train(self, g, l=None, sampler=None):
        """ Compute the output distribution with sampled weights.
        """
        if l is None:
            l = torch.ones((g.batch_size, len(self.output_regressors)), dtype=torch.bool)

        h = self.representation(g)

        # find which assays are being used
        assays = torch.arange(l.shape[1])[l.any(axis=0)]
        distributions = []
        
        for assay in assays:
            
            # if we already instantiated the head
            if str(assay) not in self.output_regressors:
                
                # get the type of self.head, and instantiate it
                self.output_regressors[str(assay)] = type(self.output_regressor)(self.representation_out_features)
                
                # move to cuda if the parent net is
                if next(self.parameters()).is_cuda:
                    self.output_regressors[str(assay)].cuda()
                
            # switch to head for that assay
            self.output_regressor = self.output_regressors[str(assay)]

            # get distribution for each input
            distribution = self.output_regressor.condition(h, sampler=sampler)            
            distributions.append(distribution)
        return distributions
            
    def loss(self, g, y):
        """ Compute the loss with a input graph and a set of parameters.
        """
        l = self._generate_mask(y)
        loss = 0.0
        distributions = self.condition(g, l)
        for idx, assay_mask in enumerate(l.T):
            if assay_mask.any():
                assay_y = y[assay_mask, idx].view(-1, 1)
                loss += -distributions[idx].log_prob(assay_y).mean()
        return loss

    def _generate_mask(self, y):
        return ~torch.isnan(y)