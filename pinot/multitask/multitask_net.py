import pinot

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
        output_regressor=pinot.inference.heads.mle_head.MaximumLikelihoodEstimationHead,
        **kwargs
    ):

        super(MultiTaskNet, self).__init__(representation, output_regressor, **kwargs)
        self.output_regressors = torch.nn.ModuleDict()

    def condition(self, g, l, sampler=None):
        """ Compute the output distribution with sampled weights.
        """
        # find which assays are being used
        assays = torch.arange(l.shape[1])[l.any(axis=0)]
        assay_distributions = []
        
        for assay in assays:
            
            # if we already instantiated the head
            if str(assay) not in self.heads:
                
                # get the type of self.head, and instantiate it
                self.output_regressors[str(assay)] = type(self.output_regressor)(self.representation_out_features)
                
                # move to cuda if the parent net is
                if next(self.parameters()).is_cuda:
                    self.output_regressors[str(assay)].cuda()
                
            # switch to head for that assay
            self.head = self.heads[str(assay)]

            # get distribution for each input
            assay_y = y[:,assay].view(-1, 1)
            distribution = super(MultiTaskNet, self).condition(g, sampler=sampler)

            # mask distribution
            assay_mask = l[:,assay]
            assay_distribution = torch.distributions.normal.Normal(
                distribution.loc[assay_mask],
                distribution.scale[assay_mask])
            
            assay_distributions.append(assay_distribution)
        return assay_distributions
            
    def loss(self, g, l, y):
        """ Compute the loss with a input graph and a set of parameters.
        """
        loss = 0.0
        distributions = self.condition(g, l)
        for idx, assay_mask in enumerate(l.T):
            if assay_mask.any():
                assay_y = y[assay_mask, idx].view(-1, 1)
                loss += -distributions[idx].log_prob(assay_y).mean()
        return loss