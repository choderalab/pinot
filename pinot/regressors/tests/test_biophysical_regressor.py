import pytest

def test_import():
    import pinot
    pinot.regressors.biophysical_regressor

def test_init():
    import pinot
    regressor = pinot.regressors.biophysical_regressor.BiophysicalRegressor(
        base_regressor=pinot.regressors.NeuralNetworkRegressor,
    )

@pytest.fixture
def net():
    import pinot
    regressor = pinot.regressors.biophysical_regressor.BiophysicalRegressor(
        base_regressor_class=pinot.regressors.NeuralNetworkRegressor,
        in_features=32,    
    )

    representation = pinot.representation.Sequential(
        config=[32, 'tanh', 32, 'tanh', 32, 'tanh'],
        layer=pinot.representation.dgl_legacy.gn()
    )

    net = pinot.Net(
        representation,
        output_regressor=regressor,
    )

    return net

def test_init(net):
    net

def test_forward(net):
    import pinot
    from rdkit import Chem
    g = pinot.graph.from_rdkit_mol(
        Chem.MolFromSmiles(
            'c1ccccc1'
        )
    )
    distribution = net.condition(g)

    import torch
    # assert distribution.mean.shape == torch.Size([1, 1])
