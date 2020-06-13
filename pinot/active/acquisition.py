# =============================================================================
# IMPORTS
# =============================================================================
import torch
from botorch.acquisition.objective import IdentityMCObjective
from botorch.sampling.samplers import SobolQMCNormalSampler
from torch.quasirandom import SobolEngine
import math

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def probability_of_improvement(distribution, y_best=0.0):
    r""" Probability of Improvement (PI).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution` object
        the predictive distribution.
    y_best : float or float tensor, default=0.0
        the best value so far.

    """

    return 1.0 - distribution.cdf(y_best)

def expected_improvement(distribution, y_best=0.0):
    r""" Expected Improvement (EI).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution` object
        the predictive distribution.
    y_best : float or float tensor, default=0.0
        the best value so far.

    """
    return distribution.mean - y_best

def upper_confidence_bound(distribution, y_best=0.0, kappa=0.5):
    r""" Upper Confidence Bound (UCB).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution` object
        the predictive distribution.
    y_best : float or float tensor, default=0.0
        the best value so far.

    """

    from pinot.inference.utils import confidence_interval
    _, high = confidence_interval(distribution, kappa)
    return high

def random(distribution, y_best=0.0, seed=2666):
    # torch.manual_seed(seed)
    return torch.rand(distribution.batch_shape)

def monte_carlo_acq(posterior, batch_size, sequential_acq=None, q=10,
                    num_samples=1000,
                    sampler_fn=SobolQMCNormalSampler,
                    objective=IdentityMCObjective(),
                    **kwargs):
    """
    Runs Monte Carlo acquisition over provided `sequential_fn`
    
    Parameters
    ----------
    posterior : GPyTorchPosterior
        Output of running net.posterior(gs).
    batch_size : int
        Batch size (i.e., gs.batch_size).
    sequential_acq : Python function
        Sequential acquisition function applied to Monte Carlo samples.
    q : int
        Size of the batch samples to be acquired.
    num_samples : int
        Number of times to sample from posterior.
    sampler_fn : BoTorch Sampler
        The sampler used to draw base samples. Defaults to SobolQMCNormalSampler.
    objective : BoTorch MCAquisitionObjective
        Monte Carlo objective under which the samples are evaluated.
        Default is IdentityMCObjective.
    **kwargs 
        Parameters to pass through to the `sequential_acq`
    
    Returns
    -------
    indices : Torch Tensor of int
        The Tensor containing the indices of each sampled batch.
    q_samples : Torch Tensor of float
        The values associated with the `sequential_acq` from the Monte Carlo samples.
    """
    # establish sampler
    SobolEngine.MAXDIM = 5000
    sampler = sampler_fn(num_samples, collapse_batch_dims=True)
    
    # perform monte carlo sampling
    seq_samples = torch.zeros((num_samples, batch_size, q))
    for q_idx in range(q):
        samples = sampler(posterior)
        obj = objective(samples)
        # evaluate samples using inner acq function
        seq_samples[:,:,q_idx] = sequential_acq(obj, **kwargs)

    # form batch with monte carlo samples
    indices = torch.randint(batch_size, seq_samples.shape[-2:])
    q_samples = torch.gather(seq_samples, 1, indices.expand(seq_samples.shape))
    q_samples = q_samples.max(dim=-1)[0].mean(dim=0)

    return indices, q_samples