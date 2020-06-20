import pytest
import torch
import pinot


def test_variational():
    import pinot
    ds = pinot.data.esol()
    ds_batched = pinot.data.utils.batch(ds, 32)

    representation = pinot.representation.Sequential(pinot.representation.dgl_legacy.gn(),
                                                 [32, 'tanh', 32, 'tanh', 32, 'tanh'])

    from pinot.regressors import VariationalGaussianProcessRegressor
    net = pinot.Net(
        representation,
        VariationalGaussianProcessRegressor
    ) # .to(torch.device('cuda:0'))

    from pinot.app.experiment import Train
    net = Train(net=net,
            data=ds_batched,
            n_epochs=10,
            optimizer=torch.optim.Adam(net.parameters(), lr=1e-4)).train()
