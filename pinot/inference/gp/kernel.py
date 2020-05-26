#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Kernel(torch.nn.Module, abc.ABC):
    r""" A Gaussian Process Kernel that hosts parameters.


    """

    @abc.abstractmethod
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def loss(self, x, y, sigma=1.0):
        r""" Calculate the negative log likelihood of data with GP regression.

        $$

        -\log p(y | x) \propto
        y^T (K + \sigma ^ 2 I) ^ {-1} y
        + \log | K + \sigma ^ 2 I|


        $$
        """
        # calculate the covariance matrix
        # for this single batch of data
        # (batch_size, batch_size)
        k = self.forward(x)

        # $ K + \sigma ^ 2 I $
        # (batch_size, batch_size)
        k_plus_sigma = k + sigma * torch.eye(k.shape[0])

        # (batch_size, batch_size)
        k_plus_sigma_inv = torch.inverse(k_plus_sigma)

        # ()
        k_plus_sigma_log_det = torch.log(
            torch.abs(
                torch.det(
                    k_plus_sigma)))

        # ()
        nll = torch.transpose(y, 0, 1) @  k_plus_sigma_inv @ y + k_plus_sigma_log_det

        return nll

    def inference(x_tr, y_tr, x_te, sigma=1.0):
        r""" Calculate the predictive distribution given `x_te`.


        """

