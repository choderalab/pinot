import torch
import pinot


@torch.no_grad()
def condition_mixture(net, g, sampler=None, n_samples=1):
    """Get the predicted distribution of measurement
    conditioned on input.

    Parameters
    ----------
    net :
        
    g :
        
    sampler :
         (Default value = None)
    n_samples :
         (Default value = 1)

    Returns
    -------

    
    """
    # initialize a list of distributions
    distributions = []

    for _ in range(n_samples):
        if sampler is not None and hasattr(sampler, "sample_params"):
            sampler.sample_params()

        distributions.append(net.condition(g))

    # get the parameter of these distributions
    # NOTE: this is not necessarily the most efficienct solution
    # since we don't know the memory footprint of
    # torch.distributions
    mus, sigmas = zip(
        *[
            (distribution.loc, distribution.scale)
            for distribution in distributions
        ]
    )

    # concat parameters together
    # (n_samples, batch_size, measurement_dimension)
    mu = torch.stack(mus)
    sigma = torch.stack(sigmas)

    # construct the distribution
    distribution = torch.distributions.normal.Normal(loc=mu, scale=sigma)

    # make it mixture
    distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
        torch.distributions.Categorical(torch.ones(mu.shape[0],)),
        torch.distributions.Independent(distribution, 2),
    )

    return distribution


def confidence_interval(distribution, percentage=0.95, n_samples=1000):
    """Calculate the confidence interval of a distribution.

    Parameters
    ----------
    distribution :
        
    percentage :
         (Default value = 0.95)
    n_samples :
         (Default value = 1000)

    Returns
    -------

    
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

        if samples_sorted.dim() == 3:
            low = samples_sorted[low_idx, :, :]
            high = samples_sorted[high_idx, :, :]

        elif samples_sorted.dim() == 2:
            low = samples_sorted[low_idx, :]
            high = samples_sorted[high_idx, :]

        else:
            raise Exception("sorry, either two or three dimensions")

    return low, high
