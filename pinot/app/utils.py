# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import torch

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def train_once(net, ds_tr, opt):
    """Train the model for one batch.

    Parameters
    ----------
    net :

    ds_tr :

    opt :


    Returns
    -------

    """
    for g, y in ds_tr:

        def l():
            """ """
            loss = torch.sum(net.loss(g, y))
            opt.zero_grad()
            loss.backward()
            return loss

        opt.step(l)

    return net, opt


def train(net, ds_tr, ds_te, opt, reporters, n_epochs):
    """

    Parameters
    ----------
    net :

    ds_tr :

    ds_te :

    opt :

    reporters :

    n_epochs :


    Returns
    -------

    """
    [reporter.before() for reporter in reporters]

    for _ in range(n_epochs):
        net.train()
        net, opt = train_once(net, ds_tr, opt)
        net.eval()
        [reporter.during(net) for reporter in reporters]

    [reporter.after(net) for reporter in reporters]


def optimizer_translation(opt_string, lr, *args, **kwargs):
    """

    Parameters
    ----------
    opt_string :

    lr :

    *args :

    **kwargs :


    Returns
    -------

    """

    if opt_string.lower() == "bbb":
        opt = lambda net: pinot.BBB(
            torch.optim.Adam(net.parameters(), lr),
            0.01,
            kl_loss_scaling=kwargs["kl_loss_scaling"],
        )

    elif opt_string.lower() == "sgld":
        opt = lambda net: pinot.SGLD(net.parameters(), lr)

    elif opt_string.lower() == "adlala":
        lr = torch.tensor(lr)
        if torch.cuda.is_available():
            lr = lr.cuda()

        opt = lambda net: pinot.AdLaLa(
            [
                {
                    "params": net.representation.parameters(),
                    "h": lr,
                    "gamma": 1e-6,
                },
                {
                    "params": net._output_regression.parameters(),
                    "h": lr,
                    "gamma": 1e-6,
                },
            ]
        )

    else:
        if "weight_decay" in kwargs:
            opt = lambda net: getattr(torch.optim, opt_string)(
                net.parameters(), lr, weight_decay=kwargs["weight_decay"]
            )
        else:
            opt = lambda net: getattr(torch.optim, opt_string)(
                net.parameters(), lr
            )
    return opt


# =============================================================================
# VARIATIONAL GP UTILS
# =============================================================================

from sklearn import cluster
import torch

def _initial_values_for_GP(train_dataset, feature_extractor, n_inducing_points):
    """ Assumes that both dataset and feature extractor
        are either cuda or not cuda.
        Also assumes the train_dataset is unbatched
    """
    steps = 10
    indices = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    f_X_samples = []

    with torch.no_grad():
        for i in range(steps):
            f_X_sample = torch.cat([
                feature_extractor(train_dataset[j.item()][0])
                for j in indices[i]
            ])
            f_X_samples.append(f_X_sample)
    
    return torch.cat(f_X_samples)
            
def _get_kmeans(f_X_sample, n_inducing_points):
    """ Get k means for multidimensional input.
    """
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    kmeans.fit(f_X_sample.cpu().numpy())
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_)

    return cluster_centers

def initialize_inducing_points(train_dataset, feature_extractor, n_inducing_points):
    """ Get initial inducing points for variational GP model.
    """
    f_X_sample = _initial_values_for_GP(
        train_dataset,
        feature_extractor,
        n_inducing_points
    )
    
    initial_inducing_points = _get_kmeans(f_X_sample, n_inducing_points)
    return initial_inducing_points