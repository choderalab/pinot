import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def negative_ELBO(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    """Compute negative ELBO loss for Graph Variational Auto Encoders
    Args:
        preds (tensor):
            Link prediction given by the decoder
        labels (tensor):
            Labelled edge data (present or absent) from adjacency matrix
        mu (float):
            Mean parameter of the approximate normal distribution
        logvar (float):
            Log variance parameter of the approximate normal distribution
        n_nodes (int):
            Number of nodes in the graph
        norm (float):
            Normalizing constant on the likelihood term (since we're 
            optimizing a stochastic ELBO)
        pos_weight (float):
            Weights for binary cross entropy

    Returns:
        The negative ELBO
    """
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD