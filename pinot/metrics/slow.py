""" Slow metrics.

"""

# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import torch
import math
import sklearn
from sklearn.metrics import r2_score

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def rmse(y, y_hat):
    return torch.sqrt(
        torch.nn.functional.mse_loss(
            y,
            y_hat))


def r2(y, y_hat):
    return r2_score(y, y_hat)



