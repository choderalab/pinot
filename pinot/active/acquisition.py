# =============================================================================
# IMPORTS
# =============================================================================
import math

import torch
from torch import nn

from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def dummy(distribution, y_best=0.0):
    r""" Dummy acquisition function that always picks the first in sequence.
    Designed to be used with temporal datasets to compare with baselines.

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.


    """
    score = torch.range(
        start=0,
        end=len(distribution.mean.flatten()) - 1
        ).flip(0)
    if torch.cuda.is_available():
        score = score.cuda()
    return score


def probability_of_improvement(distribution, y_best=0.0):
    r""" Probability of Improvement (PI).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.
    """
    return 1.0 - distribution.cdf(y_best)


def uncertainty(distribution, y_best=0.0):
    r""" Uncertainty.

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.
    """
    return distribution.variance


def expected_improvement(distribution, y_best=0.0):
    r""" Expected Improvement (EI).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.
    """
    return distribution.mean - y_best


def upper_confidence_bound(distribution, y_best=0.0, kappa=0.95):
    r""" Upper Confidence Bound (UCB).

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.
    """

    from pinot.samplers.utils import confidence_interval

    _, high = confidence_interval(distribution, kappa)
    return high


def random(distribution, y_best=0.0, seed=2666):
    """ Random assignment of scores under normal distribution.

    Parameters
    ----------
    distribution : `torch.distributions.Distribution`
        Predictive distribution.

    y_best : float
        The score for best candidate so far.
         (Default value = 0.0)

    Returns
    -------
    score : `torch.Tensor`, `shape=(n_candidates, )`
        Score for candidates under predictive distribution.

    Note
    ----
    Random seed set to `2666`, the title of the single greatest novel in
    human literary history by Roberto Bolano.
    This needs to be set to `None` if parallel experiments were to be performed.

    """
    # torch.manual_seed(seed)
    return torch.rand(distribution.batch_shape)


# =============================================================================
# MODULE CLASSES
# =============================================================================


class MCAcquire:
    """Implements Monte Carlo acquisition."""

    def __init__(
        self,
        sequential_acq,
        batch_size,
        q=10,
        num_samples=1000,
        marginalize_batch=False,
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
        self.marginalize_batch = marginalize_batch

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
        # sample from posterior
        seq_samples = posterior.sample(
            torch.Size([self.q, self.num_samples])
        ).squeeze(-1)

        # shuffle within outer dimension of qth row to obtain random baches
        indices = torch.randint(batch_size, (self.q, batch_size))
        q_samples = torch.stack(
            [
                row[:, indices[idx]]
                for idx, row in enumerate(torch.unbind(seq_samples))
            ]
        )

        if self.marginalize_batch:
            # collapse individuals within each batch
            q_samples = q_samples.reshape(
                q_samples.shape[0] * q_samples.shape[1], q_samples.shape[2]
            )
            # apply sequential acquisition function on joint distribution over batch
            scores = self.sequential_acq(q_samples, axis=0)

        else:
            # apply sequential acquisition function on in parallel over batch
            scores = self.sequential_acq(q_samples, axis=1)
            # find max expectation of acquisition function within batch
            scores = scores.max(axis=0).values

        return indices, scores


class SeqAcquire:
    """Wraps sequential to keep track of hyperparameters."""

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
            if acq_fn == "ei":
                self.acq_fn = self.EI
            elif acq_fn == "pi":
                self.acq_fn = self.PI
            elif acq_fn == "ucb":
                self.acq_fn = self.UCB
            elif acq_fn == "uncertainty":
                self.acq_fn = self.VAR
            elif acq_fn == "random":
                self.acq_fn = self.RAND
            else:
                raise ValueError(
                    f"""{acq_fn} is not an available function.
                    Only available functions are 'ei', 'pi', 'ucb', 'uncertainty', or 'random'."""
                )
        else:
            self.acq_fn = acq_fn

        self.kwargs = kwargs

    def __call__(self, samples, axis, y_best=0.0):
        return self.acq_fn(
            samples=samples, axis=axis, y_best=y_best, **self.kwargs
        )

    def PI(self, samples, axis=0, y_best=0.0):
        """

        Parameters
        ----------
        samples :

        axis :
             (Default value = 0)
        y_best :
             (Default value = 0.0)

        Returns
        -------

        """
        return (samples > y_best).float().mean(axis=axis)

    def EI(self, samples, axis=0, y_best=0.0):
        """

        Parameters
        ----------
        samples :

        axis :
             (Default value = 0)
        y_best :
             (Default value = 0.0)

        Returns
        -------

        """
        return (samples - y_best).mean(axis=axis)

    def VAR(self, samples, axis=0, y_best=0.0):
        """

        Parameters
        ----------
        samples :

        axis :
             (Default value = 0)
        y_best :
             (Default value = 0.0)

        Returns
        -------

        """
        return samples.var(axis=axis)

    def RAND(self, samples, axis=0, y_best=0.0):
        """

        Parameters
        ----------
        samples :

        axis :
             (Default value = 0)
        y_best :
             (Default value = 0.0)

        Returns
        -------

        """
        return torch.rand(samples.shape[1])

    def UCB(self, samples, beta, axis=0, y_best=0.0):
        """

        Parameters
        ----------
        samples :

        beta :

        axis :
             (Default value = 0)
        y_best :
             (Default value = 0.0)

        Returns
        -------

        """
        samples_sorted, idxs = torch.sort(samples, dim=0)
        high_idx = int(len(samples) * (1 - (1 - beta) / 2))
        if axis == 0:
            high = samples_sorted[high_idx, :]
        else:
            high = samples_sorted[:, high_idx, :]
        return high
