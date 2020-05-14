# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import pinot

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

class GraphVAE(torch.nn.module):
    r""" Variational Graph Auto-Encoders. (VGAE)
    arXiv: 1611.07308

    Inference:


    $$

    p(A|z) = \prod\limits_{i=1}^N \prod\limits_{i=1}^N
            p(A_{ij} | z_i, z_j)

    p(A_{ij} = 1 | z_i, z_j) = \operatorname{sigmoid}(z_i^T z_j)

    $$

    """
    def __init__(self):
        self.linear = torch.nn.Linear(1, 2)

    def parametrization(self, x):
        r""" Generate the parameters for a Gaussian distribution
        from the latent representation of nodes.

        Parameters
        ----------
        x : torch.tensor, shape=(`N`, `D`)
            where `N` is the number of nodes and `D` is the hidden dimension.
            latent representation of nodes.

        Returns
        -------
        mu, log_sigma : tuple of torch.tensor, with each shape=(`N`, `D`)
            where `N` is the number of nodes and `D` is the hidden dimension.
            parameters of distribution.

        """
        # expand the dimension of the input signal
        x = x[:, :, None]

        # compute the parameters
        theta = self.linear(x)

        # unbind to two parameters and return
        return torch.unbind(theta, dim=2)

    @staticmethod
    def inference(mu, log_sigma):
        r""" Construct a distribution of latent code from parameters.


        $$
        
        q(Z|X, A) = \prod\limits_{i=1}^N q(z_i | X, A)
        
        q(z_i | X, A) = \mathcal{N}(z_i | \mu_i, \operatorname{\diag}(\sigma^2)

        $$

        Parameters
        ----------
        mu : torch.tensor, shape=(`N`, `D`)
            where `N` is the number of nodes and `D` is the hidden dimension
        log_sigma  : torch.tensor, shape=(`N`, `D`)
            where `N` is the number of nodes and `D` is the hidden dimension
        
        Returns
        -------
        q_z_i : latent distribution conditioned on nodes of a graph.
        
        """
        # construct latent distribution conditioned on _node_ of a graph.
        # (N, D)
        q_z_i = torch.distributions.normal.Normal(
                loc=mu,
                scale=torch.exp(log_sigma))

        return q_z_i

    @staticmethod
    def p_a_given_z(a, z):
        """ Calculate the probability of an adjacency matrix given the latent code.
        
        TODO: better naming
        
        Parameters
        ----------
        a : torch.tensor, shape=(`N`, `N`)
            where `N` is the number of nodes.
            the adjacency map
        z : torch.tensor, shape=(`N`, `D`)
            where `N` is the number of nodes and `D` is the hidden dimension
            latent code.
        """
        # compute the dot product of latent code
        # (N, N)
        z_i_t_z_j = torch.sum(
                z[:, None, :] * z[None, :, :],
                dim=2)



