import pytest
import torch
import numpy as np
import numpy.testing as npt

def test_import():
    import pinot
    from pinot.generative.torch_gvae.model import GCNModelVAE
    gvae = GCNModelVAE(16)

@pytest.fixture
def gvae():
    import pinot
    from pinot.generative.torch_gvae.model import GCNModelVAE
    return GCNModelVAE(16)

def test_infer_node_representation():
    import pinot
    import dgl
    from pinot.data import esol
    from pinot.generative.torch_gvae.model import GCNModelVAE
    ds = esol()[:10]
    gs, _ = zip(*ds)
    g = dgl.batch(gs)
    ndims = g.ndata["h"].shape[1]
    gvae = GCNModelVAE(ndims)
    z = gvae.infer_node_representation(g)
    assert z.shape[0] == g.ndata["h"].shape[0]

def test_forward():
    import pinot
    import dgl
    from pinot.data import esol
    from pinot.generative.torch_gvae.model import GCNModelVAE
    ds = esol()[:10]
    gs, _ = zip(*ds)
    g = dgl.batch(gs)
    ndims = g.ndata["h"].shape[1]
    gvae = GCNModelVAE(ndims)
    z = gvae.forward(g)
    assert len(z.shape) == 1
    assert z.shape[0] == gvae.embedding_dim

def test_inference():
    import pinot
    import dgl
    from pinot.data import esol
    from pinot.generative.torch_gvae.model import GCNModelVAE
    ds = esol()[:10]
    gs, _ = zip(*ds)
    g = dgl.batch(gs)
    ndims = g.ndata["h"].shape[1]
    gvae = GCNModelVAE(ndims)
    dis, mu, logvar = gvae.condition(g)
    assert (mu.shape[0] == g.ndata["h"].shape[0])
    assert (logvar.shape[0] == g.ndata["h"].shape[0])
    z = dis.rsample()
    assert (z.shape[0] == g.ndata["h"].shape[0])