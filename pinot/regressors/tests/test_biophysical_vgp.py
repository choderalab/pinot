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

    import dgl
    g = dgl.batch([g, g])

    distribution = net.condition(g, test_ligand_concentration=0.001)

    assert distribution.mean.shape == torch.Size([2, 1])

def test_loss(net):
    import pinot
    import torch
    from rdkit import Chem

    g = pinot.graph.from_rdkit_mol(
        Chem.MolFromSmiles(
            'c1ccccc1'
        )
    )

    import dgl
    g = dgl.batch([g, g])

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
        n_epochs=2,
    )
    train.train()


'''
def test_train_and_test(net):
    import pinot
    import torch
    ds = pinot.data.moonshot_mixed()
    view = ds.view('fixed_size_batch', batch_size=32)
    experiment = pinot.TrainAndTest(
        net=net,
        data_tr=view,
        data_te=view,
        optimizer=torch.optim.Adam(net.parameters(), 1e-3),
        n_epochs=2,
    )

    experiment.run()
'''
