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

    def inference(self, mu, log_sigma):
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

    def log_p_a_given_z(self, a, z):
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
        
        Returns
        -------
        log_p_a_z : torch.tensor, shape=(),
            log probability of adjacency map given laten

        """
        # compute the dot product of latent code
        # (N, N)
        z_i_t_z_j = torch.sum(
                z[:, None, :] * z[None, :, :],
                dim=2)
        
        # compute the joint probability of edges given latent code
        log_p_a_given_z = torch.where(
                torch.gt(a, 0.5),
                torch.log(torch.nn.functional.sigmoid(z_i_t_z_j)),
                torch.log(1.0 - torch.nn.function.sigmoid(z_i_t_z_j))).sum()

        return log_p_a_given_z

    def loss(self, x, a):
        """ Calculate the ELBO of the learning and inference processes.

        """
        # get distribution parameters
        theta = self.linear(x)

        # inference pass
        q_z_i = self.inference(theta)

        # sample from q_z_i with reparametrization trick
        z = q_z_i.rsample()

        # compute loss function
        # reconstruct term
        log_p_a_given_z = self.log_p_a_given_z(a, z)

        # kl-divergence term
        kl = q_z_i.log_prob(z) - torch.distributions.normal.Normal(
                loc=torch.zeros_like(z),
                scale=torch.ones_like(z)).log_prob(z)

        return log_p_a_given_z - kl



