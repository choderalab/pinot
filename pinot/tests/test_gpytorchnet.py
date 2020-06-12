import pytest


def test_import():
    import pinot.net


def test_initialize():
    import pinot

    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"],
    )
    net = pinot.net.GPyTorchNet(net_representation, exactgp)


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
    
    net = pinot.net.GPyTorchNet(net_representation, exactgp)

    net.posterior(g)