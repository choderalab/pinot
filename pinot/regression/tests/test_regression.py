import pytest


def test_import():
    import pinot.regression
    import pinot.regression.models


def test_initialize():
    import pinot

    g, y = pinot.data.esol()[0]
    layer = pinot.representation.dgl_legacy.GN
    config = [32, "tanh", 32, "tanh", 32, "tanh"]
    
    net_representation = pinot.representation.Sequential(
        layer, config,
    )

    from pinot.regression.models import ExactGPModel
    exactgp = ExactGPModel(g.batch_size, config[0])


def test_forward():
    import pinot
    
    g, y = pinot.data.esol()[0]
    layer = pinot.representation.dgl_legacy.GN
    config = [32, "tanh", 32, "tanh", 32, "tanh"]
    
    net_representation = pinot.representation.Sequential(
        layer, config,
    )

    from pinot.regression.models import ExactGPModel
    exactgp = ExactGPModel(g.batch_size, config[0])
    
    h = net_representation(g)
    theta = exactgp.forward(h)