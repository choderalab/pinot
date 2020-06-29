import pytest
import pinot


def test_batched_graph_shape():
    """ """
    ds = pinot.data.esol()
    ds = pinot.data.utils.batch(ds, 8)
    g, y = ds[0]

    layer = pinot.representation.dgl_legacy.gn()
    net = pinot.representation.Sequential(layer, [32, "tanh", 32, "tanh"])

    h = net(g)

    assert h.dim() == 2
    assert h.shape[0] == 8
