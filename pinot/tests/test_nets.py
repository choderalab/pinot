import pytest


def test_import():
    import pinot.net


def test_initialize():
    import pinot

    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"],
    )
    net = pinot.Net(net_representation)


def test_forward():
    import pinot
    g, y = pinot.data.esol()[0]
    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"],
    )
    net = pinot.Net(net_representation)

    net.condition(g)


   
