import pytest

def test_import():
    import pinot
    pinot.regressors.biophysical_regressor

@pytest.fixture
def net():
    import pinot

    representation = pinot.representation.Sequential(
        config=[32, 'tanh', 32, 'tanh', 32, 'tanh'],
        layer=pinot.representation.dgl_legacy.gn()
    )

    net = pinot.Net(
        representation,
        output_regressor_class=pinot.regressors.gaussian_process_regressor\
            .BiophysicalVariationalGaussianProcessRegressor
    )

    return net

def test_init(net):
    net

def test_forward(net):
    import pinot
    import torch
    from rdkit import Chem
    g = pinot.graph.from_rdkit_mol(
        Chem.MolFromSmiles(
            'c1ccccc1'
        )
    )
    distribution, _, _ = net.condition(g, test_ligand_concentration=0.001)

    assert distribution.mean.shape == torch.Size([1, 1])

def test_loss(net):
    import pinot
    import torch
    from rdkit import Chem

    g = pinot.graph.from_rdkit_mol(
        Chem.MolFromSmiles(
            'c1ccccc1'
        )
    )

    net.loss(g, torch.zeros(1, 1), 0.001)

def test_train(net):
    import pinot
    import torch
    ds = pinot.data.moonshot_mixed()
    view = ds.view('fixed_size_batch', batch_size=32)
    train = pinot.Train(
        net=net,
        data=view,
        optimizer=torch.optim.Adam(net.parameters(), 1e-3),
        n_epochs=10,
    )
    train.train()
