import pytest


def test_initialize():
    from pinot.generative.torch_gvae.model import GCNModelVAE
    from pinot.net import Net
    gvae = GCNModelVAE(117)
    net = Net(gvae)

def test_condition():
    import pinot
    from pinot.generative.torch_gvae.model import GCNModelVAE
    from pinot.net import Net
    g, y = pinot.data.esol()[0]
    layer = pinot.representation.dgl_legacy.GN

    gvae = GCNModelVAE(117)
    net = Net(gvae)
    dis = net.condition(g)
    assert (hasattr(dis, "mean"))
    assert (hasattr(dis, "sample"))


def test_loss():
    import pinot
    from pinot.generative.torch_gvae.model import GCNModelVAE
    from pinot.net import Net
    g, y = pinot.data.esol()[0]
    layer = pinot.representation.dgl_legacy.GN

    gvae = GCNModelVAE(117)
    net = Net(gvae)
    loss = net.loss(g, y)
    assert (loss.numel() == y.numel())
