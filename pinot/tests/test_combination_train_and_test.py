import pytest
import pinot
import torch
import numpy as np

@pytest.fixture
def representation():
    layer = pinot.representation.dgl_legacy.gn()
    representation = pinot.representation.Sequential(
        layer,
        [32, 'tanh', 32, 'tanh', 32, 'tanh'])

    return representation

@pytest.fixture
def ds():
    ds = pinot.data.esol()[:8]
    ds = pinot.data.utils.batch(ds, 4)
    return ds


@pytest.mark.parametrize(
    'net',
    [
        pinot.Net,
])
@pytest.mark.parametrize(
    'regressor',
    [
        pinot.regressors.ExactGaussianProcessRegressor,
        pinot.regressors.VariationalGaussianProcessRegressor,
        pinot.regressors.NeuralNetworkRegressor
    ]
)
def test_train(net, regressor, representation, ds):
    net = net(
        output_regressor=regressor,
        representation=representation
    )

    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    exp = pinot.app.experiment.TrainAndTest(
        optimizer=optimizer,
        net=net,
        data_tr=ds,
        data_te=ds,
        n_epochs=1
    )

    results = exp.run()

    for key, subdict in results.items():
        for metric, value in subdict.items():
            for idx, res in value.items():
                assert np.isnan(res).any() == False
