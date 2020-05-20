# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
import copy
from abc import abstractmethod

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Sampler(torch.optim.Optimizer, abc.ABC):
    """ The base class for samplers.

    """
    def __init__(self):
        super(Sampler, self).__init__()
   
    @abstractmethod
    def sample_params(self,  *args, **kwargs):
        pass

    @abstractmethod
    def expectation_params(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def condition(self, net, g, n_samples=1, *args, **kwargs):
        """ Get the predicted distribution of measurement
        conditioned on input.

        Parameters
        ----------
        net : `pinot.Net` module
        g : input of net
        n_samples : int, default=1,
            number of copies of distributions to be mixed.

        """
        # initialize a list of distributions
        distributions = []
        
        for _ in range(n_samples):
            self.sample_params()
            distributions.append(net.condition(g))

        # get the parameter of these distributions
        # NOTE: this is not necessarily the most efficienct solution
        # since we don't know the memory footprint of 
        # torch.distributions
        mus, sigmas = zip(*[
                (distribution.loc, distribution.scale)
                for distribution in distributions])

        # concat parameters together
        # (n_samples, batch_size, measurement_dimension)
        mu = torch.stack(mus)
        sigma = torch.stack(sigmas)

        # construct distribution            
        distribution = torch.distributions.Normal.normal(
                loc=mu,
                scale=sigma)
         
        # make it mixture
        distribution = torch.distributions.mixture_same_family\
                .MixtureSameFamily(
                        torch.distributions.Categorical(
                            torch.ones(mu.shape[0])),
                        distribution)

        return distribution


        
