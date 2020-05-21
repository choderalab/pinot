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
            sampler,
            'sample_params'):
            sampler.sample_params()
            
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

def confidence_interval(distribution, percentage=0.95, n_samples=1000):
    """ Calculate the confidence interval of a distribution.

    Parameters
    ----------
    distribution : `torch.distributions.Distribution` object
        the distribution to be characterized.
        the event dimension has to be [-1, 1].
    percentage : float, default=0.95
        percentage of confidence interval.
    n_samples : int, default=100
        number of samples to be drawn for confidence interval to be 
        calculated.

    """
    percentage = torch.tensor(percentage)

    try:
        low = distribution.icdf((1 - percentage) / 2)
        high = distribution.icdf(1 - (1 - percentage) / 2)

    except:
        samples = distribution.sample([n_samples])

        samples_sorted, idxs = torch.sort(samples, dim=0)

        low_idx = int(n_samples * (1 - percentage) / 2) 
        high_idx = int(n_samples * (1 - (1 - percentage) / 2))

        low = samples_sorted[low_idx, :, :]
        high = samples_sorted[high_idx, :, :]
        
    
    return low, high
