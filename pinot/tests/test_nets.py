import pytest


def test_import():
    import pinot.net


def test_initialize():
    import pinot

    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"]
    )
    net_regression = pinot.regression.Linear(32, 1)
    net = pinot.Net(net_representation, net_regression)
