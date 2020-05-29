import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def negative_ELBO(edge_preds, adj, mu, logvar, norm):
    """Compute negative ELBO loss for Graph Variational Auto Encoders
    Args:
        preds (FloatTensor):
            shape (N, N): a matrix where entry (i,j) in [0,1] denotes the predicted
            probability that there is an edge between atom i and j

        labels (FloatTensor):
            shape (N, N): the adjacency matrix of the molecular graph

        mu (FloatTensor):
            shape (N, hidden_dim): The mean of the approximate posterior distribution over
            the nodes' (atoms) latent representation

        logvar (FloatTensor):
            shape (N, hidden_dim): The log variance of the approximate posterior distribution
            over the nodes' latent representation
    
        norm (Float):
            Normalizing factor for the log likelihood term in the ELBO
    Returns:
        loss (Float)
            The negative ELBO

    """
    n_nodes = edge_preds.shape[0]
    cost = torch.sum(norm * (F.binary_cross_entropy_with_logits(edge_preds, 
        adj)))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    return cost + KLD


def negative_ELBO_with_node_prediction(edge_preds, node_preds, adj, node_types, mu, logvar):
    """ Compute the negative ELBO loss function used in variational auto-encoder
    The difference between this loss function and negative_ELBO is that this 
    function also computes loss term from node's identity (atom type) prediction.

    Args:
        edge_preds (FloatTensor):
            shape (N, N): a matrix where entry (i,j) in [0,1] denotes the predicted
            probability that there is an edge between atom i and j
        
        node_preds (FloatTensor):
            shape (N, num_atom_types): Each row i stores the predicted probability
            of the type for atom i

        adj (FloatTensor):
            shape (N, N): the adjacency matrix of the molecular graph

        node_types:
            shape (N, num_atom_types): the true atom types in 1-hot form

        mu (FloatTensor):
            shape (N, hidden_dim): The mean of the approximate posterior distribution over
            the nodes' (atoms) latent representation

        logvar (FloatTensor):
            shape (N, hidden_dim): The log variance of the approximate posterior distribution
            over the nodes' latent representation

        norm (Float):
            Normalizing factor for the log likelihood term in the ELBO
    
    Returns:
        loss (Float)
            The negative ELBO

    """
    n_nodes = edge_preds.shape[0]

    node_preds_masked = node_preds.clone()
    node_preds_masked[torch.isnan(node_preds_masked)] = 0

    edge_preds_masked = edge_preds.clone()
    edge_preds_masked[torch.isnan(edge_preds_masked)] = 0

    edge_nll = torch.sum(F.binary_cross_entropy(edge_preds_masked, adj))
    node_nll = torch.sum(F.cross_entropy(node_preds_masked, node_types))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 / n_nodes * torch.sum(torch.sum(
    #     1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    return node_nll # + edge_nll #+ KLD