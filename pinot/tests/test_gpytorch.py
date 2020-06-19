import pytest

def test_import():
    import pinot.net

@pytest.fixture
def net():
    import pinot

    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"],
    )

    net = pinot.Net(
        net_representation,
        output_regressor=pinot.inference.output_regressors\
            .gaussian_process_output_regressor\
            .VariationalGaussianProcessOuputRegressor)

    return net


@pytest.fixture
def ds0():
    import torch
    import pinot
    ds = pinot.data.esol()[:8]
    ds = pinot.data.utils.batch(ds, 8)
    return ds[0]

def test_init(net):
    net

def test_condition(net):
    import torch
    import pinot
    ds = pinot.data.esol()[:8]
    ds = pinot.data.utils.batch(ds, 8)
    g, y = ds[0]

    loss = net.loss(g, y)

def test_train_eval(net):
    net.train()
    net.eval()

def test_train_one_step(net, ds0):
    import torch
    g, y = ds0
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    opt.zero_grad()
    loss = net.loss(g, y).mean()
    loss.backward()
    opt.step()
