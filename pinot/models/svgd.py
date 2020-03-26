""" Stein variational gradient descent.

arXiv:1608.04471
"""
import torch
import numpy as np

def rbf_kernel(x):
    """ Compute the pairwise RBF kernel for a stack of vectors.
    """
    # compute pairwise distance
    pairwise_distance_square = torch.pow(x[:, None, :] - x[None, :, :], 2).sum(dim=-1)

    # compute bandwidth
    h = torch.div(
        torch.median(torch.reshape(pairwise_distance_square, [-1])),
        np.log(x.shape[0]))

    k_xx = torch.exp(
        torch.div(
            pairwise_distance_square,
            -h))

    return k_xx


def svgd_grad(theta, grad):
    # theta and grad here is stacked among particles

    # compute kernel
    # (n, n)
    k_xx = rbf_kernel(theta)

    # compute the grad of k_xx w.r.t. theta
    # (n, d)
    d_k_xx_d_theta = torch.autograd.grad(k_xx.sum(), theta)[0]

    # compute the modified grad
    phi = (k_xx.detach().matmul(grad) - d_k_xx_d_theta) / theta.size(0)
    
    return phi
