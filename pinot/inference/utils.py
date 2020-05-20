import torch
import pinot

@torch.no_grad()
def condition_mixture(net, g, sampler=None, n_samples=1):
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
        if sampler is not None and hasattr(
            optimizer,
            sample_params):
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
