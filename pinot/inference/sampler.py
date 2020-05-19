# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
import copy

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Sampler(torch.optim.Optimizer, abc.ABC):
    """ The base class for samplers.

    """
    def __init__(self):
        super(Sampler, self).__init__()
   
    @abstractmethod
    def net_ensemble(self, n_samples, *args, **kwargs):
        pass

    def posterior(self, net, g, n_samples=1, *args, **kwargs):
        with torch.no_grad();
            # generate an ensemble of net
            nets = self.net_ensemble(n_samples)
            
            # get a list of distributions parametrized by nets
            distributions = [
                    net.condition(g)
                    for net in nets]

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


        
