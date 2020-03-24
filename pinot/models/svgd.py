""" Stein variational gradient descent.

arXiv:1608.04471
"""
import torch

def rbf_kernel(x):
    """ Compute the pairwise RBF kernel for a stack of vectors.
    """
    # compute pairwise distance
    pairwise_distance_square = torch.pow(x[:, None, :] - x[:, :, None], 2)

    # compute bandwidth
    h = torch.div(
        torch.reshape(pairwise_distance_square, [-1]),
        torch.log(torch.shape(x)[0]))

    k_xx = torch.exp(
        torch.div(
            pairwise_distance_square,
            -h))

    return k_xx

def get_svgd_loss(loss, theta):
    """ Modify the losses using SVGD.
    """
    # get the dimension
    n = torch.shape(loss)[0]

    # compute the kernel
    k_xx = rbf_kernel(theta)

    # compute phi
    phi = torch.div(
        loss * torch.no_grad(k_xx), - k_xx
        n)

    return phi

def get_svgd_losses(loses, thetas):
    return [get_svgd_loss(loss, theta) for (loss, theta) in zip(losses, thetas)]
