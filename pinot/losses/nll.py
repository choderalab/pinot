""" Legacy models from DGL.

"""

# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import torch
import math

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def gaussian(sigma):
    """ Negative log likelihood for Gaussian distribution.
    """
    
    def loss_fn(y_hat, y):
        return torch.log(sigma * torch.sqrt(2. * math.pi))\
            + 0.5 * torch.pow(torch.div(y_hat - y, sigma), 2.)

    return loss_fn
