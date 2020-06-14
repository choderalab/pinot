# =============================================================================
# IMPORTS
# =============================================================================
import math

import torch
from torch import nn

from botorch.acquisition.objective import IdentityMCObjective
from botorch.sampling.samplers import SobolQMCNormalSampler
from torch.quasirandom import SobolEngine

from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

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

def uncertainty(distribution, y_best=0.0):
    r""" Uncertainty.

    Parameters
    ----------
    distribution : `torch.distributions.Distribution` object
        the predictive distribution.
    y_best : float or float tensor, default=0.0
        the best value so far.

    """
    return distribution.variance

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

# =============================================================================
# MODULE CLASSES
# =============================================================================

class MCAcquire:
    """ Implements Monte Carlo acquisition.

    """
    def __init__(
            self,
            sequential_acq,
            batch_size, q=10,
            num_samples=1000,
            sampler_fn=SobolQMCNormalSampler,
            objective=IdentityMCObjective
        ):
        """
        Runs Monte Carlo acquisition over provided `sequential_fn`
        
        Parameters
        ----------
        sequential_acq : Python function
            Sequential acquisition function applied to Monte Carlo samples.
        batch_size : int
            Batch size (i.e., gs.batch_size).
        q : int
            Size of the batch samples to be acquired.
        num_samples : int
            Number of times to sample from posterior.
        sampler_fn : BoTorch Sampler
            The sampler used to draw base samples. Defaults to SobolQMCNormalSampler.
        objective : BoTorch MCAquisitionObjective
            Monte Carlo objective under which the samples are evaluated.
            Default is IdentityMCObjective.
        """
        super(MCAcquire, self).__init__()

        self.sequential_acq = sequential_acq
        self.q = q
        self.num_samples = num_samples
        self.sampler = sampler_fn(self.num_samples,
                                  collapse_batch_dims=True,
                                  resample=True)
        self.objective = objective()

    def __call__(self, posterior, batch_size, y_best):
        """
        Parameters
        ----------
        posterior : GPyTorchPosterior
            Output of running net.posterior(gs).        

        Returns
        -------
        indices : Torch Tensor of int
            The Tensor containing the indices of each sampled batch.
        y_best : float
            The best y output that's been seen so far.
        q_samples : Torch Tensor of float
            The values associated with the `sequential_acq` from the Monte Carlo samples.
        """
        # establish sampler
        SobolEngine.MAXDIM = 5000
        
        # perform monte carlo sampling
        seq_samples = torch.zeros((self.num_samples,
                                   batch_size,
                                   self.q))
        for q_idx in range(self.q):
            
            samples = self.sampler(posterior)
            obj = self.objective(samples)

            # evaluate samples using inner acq function
            seq_samples[:,:,q_idx] = self.sequential_acq(obj=obj,
                                                         y_best=y_best)

        # form batch with monte carlo samples
        indices = torch.randint(batch_size, seq_samples.shape[-2:])
        q_samples = torch.gather(seq_samples, 1, indices.expand(seq_samples.shape))
        q_samples = q_samples.max(dim=-1)[0].mean(dim=0)

        return indices, q_samples


class SeqAcquire:
    """ Wraps sequential to keep track of hyperparameters.
    """
    def __init__(self, acq_fn, **kwargs):
        """
        Parameters
        ----------
        acq_fn : str or function
            If string, must be one of 'ei', 'ucb', 'pi', 'random'.
            If a function, it will be used directly as acq function.
        **kwargs
            Any state parameters to pass to acq_fn.
        """
        if isinstance(acq_fn, str):
            if acq_fn == 'ei':
                self.acq_fn = self.EI
            elif acq_fn == 'pi':
                self.acq_fn = self.PI
            elif acq_fn == 'ucb':
                self.acq_fn = self.UCB
            elif acq_fn == 'random':
                self.acq_fn = self.RAND
            else:
                raise ValueError(f"""{acq_fn} is not an available function.
                    Only available functions are 'ei', 'pi', 'ucb', 'variance', or 'random'.""")
        else:
            self.acq_fn = acq_fn

        self.kwargs = kwargs

    def __call__(self, obj, y_best=0.0):
        return self.acq_fn(obj=obj, y_best=y_best, **self.kwargs)

    def EI(self, obj, y_best):
        return obj - y_best

    def UCB(self, obj, beta, y_best=0.0):
        beta_prime = math.sqrt(beta * math.pi / 2)
        mean = obj.mean(dim=0)
        return mean + beta_prime * (obj - mean).abs()

    def PI(self, obj, tau, y_best=0.0):
        max_obj = obj.max(dim=-1)[0]
        return torch.sigmoid((max_obj - y_best) / tau).mean(dim=0)

    def RAND(self, obj, y_best=0.0):
        return torch.rand(obj.shape)


class BTModel(nn.Module):
    r"""Wrapping the pinot gpr model for BoTorch."""
    
    def __init__(self, gpr):
        super(BTModel, self).__init__()
        self.gpr = gpr
        self.num_outputs = 1 # needs to be 1 for most of these functions

    def condition(self, X):
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `b x q x d`-dim Tensor, where `d` is the dimension of the
                feature space, `q` is the number of points considered jointly,
                and `b` is the batch dimension.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add observation noise to the posterior.

        Returns:
            A `Posterior` object, representing a batch of `b` joint distributions
            over `q` points and `m` outputs each.
        """
        self.gpr.eval()
        out = self.gpr.condition(X)
        gptorch_mvn = MultivariateNormal(out.mean, out.covariance_matrix)
        return GPyTorchPosterior(gptorch_mvn)
    
    def loss(self, g, y):
        return self.gpr.loss(g, y)