import pytest


def test_import():
    """ """
    import pinot.net


@pytest.fixture
def net():
    """ """
    import pinot

    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"],
    )

    net = pinot.Net(
        net_representation,
        output_regressor_class=pinot.regressors.ExactGaussianProcessRegressor,
    )

    return net


def test_init(net):
    """

    Parameters
    ----------
    net :


    Returns
    -------

    """
    net


def test_condition(net):
    """

    Parameters
    ----------
    net :


    Returns
    -------

    """
    import torch
    import pinot

    ds = pinot.data.esol()[:8]
    ds = pinot.data.utils.batch(ds, 8)
    g, y = ds[0]

    loss = net.loss(g, y)
