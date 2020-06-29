import pinot
import gpytorch
import torch


class MultiTaskNet(pinot.Net):
    """An object that combines the representation and parameter
    learning, puts into a predicted distribution and calculates the
    corresponding divergence.

    Parameters
    ----------

    Returns
    -------

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

        super(MultiTaskNet, self).__init__(
            representation, output_regressor, **kwargs
        )
        self.output_regressors = torch.nn.ModuleDict()

    def condition(self, g, assay):
        """Compute the output distribution with sampled weights.

        Parameters
        ----------
        g :
            
        assay :
            

        Returns
        -------

        """
        self.eval()
        h = self.representation(g)
        # nn.ModuleDict needs string keys
        assay = str(assay)

        # if we already instantiated the output_regressor
        if assay not in self.output_regressors:

            # get the type of self.output_regressor, and instantiate it
            self.output_regressors[assay] = type(self.output_regressor)(
                self.representation_out_features
            )

            # move to cuda if the parent net is
            if next(self.parameters()).is_cuda:
                self.output_regressors[assay].cuda()

        # switch to head for that assay
        self.output_regressor = self.output_regressors[assay]

        # get distribution for each input
        distribution = self.output_regressor.condition(h)
        return distribution

    def condition_train(self, g, l, sampler=None):
        """Compute the output distribution with sampled weights.

        Parameters
        ----------
        g :
            
        l :
            
        sampler :
             (Default value = None)

        Returns
        -------

        """
        h = self.representation(g)

        # find which assays are being used
        assays = [str(assay) for assay in range(l.shape[1])]
        distributions = []

        for assay in assays:

            # if we already instantiated the output_regressor
            if assay not in self.output_regressors:

                # get the type of self.output_regressor, and instantiate it
                self.output_regressors[assay] = type(self.output_regressor)(
                    self.representation_out_features
                )

                # move to cuda if the parent net is
                if next(self.parameters()).is_cuda:
                    self.output_regressors[assay].cuda()

            # switch to head for that assay
            self.output_regressor = self.output_regressors[assay]

            # get distribution for each input
            distribution = self.output_regressor.condition(h)
            distributions.append(distribution)

        return distributions

    def loss(self, g, y):
        """Compute the loss with a input graph and a set of parameters.

        Parameters
        ----------
        g :
            
        y :
            

        Returns
        -------

        """
        loss = 0.0
        l = self._generate_mask(y)
        distributions = self.condition_train(g, l)
        print(distributions[0].loc)

        for idx, assay_mask in enumerate(l.T):
            if assay_mask.any():
                # create dummy ys if unlabeled
                y_dummy = torch.zeros(y.shape[0], device=y.get_device()).view(
                    -1, 1
                )
                y_dummy[assay_mask] = y[assay_mask, idx].view(-1, 1)
                # compute log probs
                log_probs = distributions[idx].log_prob(y_dummy)
                # mask log probs
                loss += -log_probs[assay_mask].mean()
        return loss

    def _generate_mask(self, y):
        """

        Parameters
        ----------
        y :
            

        Returns
        -------

        """
        return ~torch.isnan(y)
